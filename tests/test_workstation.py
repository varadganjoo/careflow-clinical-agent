"""Opt-in rendered journeys against the real API with a deterministic test scribe.

Run with CAREFLOW_BROWSER_TESTS=1 after installing Playwright and Chromium.
"""

import json
import os
from pathlib import Path
import re
import socket
from threading import Thread
import time

import pytest

pytestmark = pytest.mark.skipif(os.getenv("CAREFLOW_BROWSER_TESTS") != "1", reason="Opt-in Playwright browser checks")


@pytest.fixture(scope="module")
def workstation():
    import uvicorn
    from app.main import app
    from app.llm import SOAPNote

    note = SOAPNote(subjective="Follow-up visit.", objective="See sample chart.",
                    assessment="Medication review required.", plan=["Review current medications."],
                    icd10_codes=["I10"], cpt_codes=["99214"])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("app.graph.generate_structured", lambda *args: (note, 1, 1))
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert server.started, "Test server failed to start"
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            sock.close()


@pytest.fixture
def page(workstation):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=os.getenv("CAREFLOW_CHROMIUM_PATH"))
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(workstation)
        page.wait_for_load_state("networkidle")
        page.get_by_role("heading", name="Eleanor Vance").wait_for()
        yield page
        browser.close()
        assert errors == [], errors


def select_patient(page, name):
    page.get_by_role("button", name=re.compile(name)).click()
    page.get_by_role("heading", name=name).wait_for()


def generate(page):
    page.get_by_role("button", name="Generate draft", exact=True).click()
    page.locator("#f_assessment").wait_for()


def test_critical_review_preserves_edits_and_downloads_receipt(page, tmp_path):
    select_patient(page, "Robert Chen")
    page.get_by_role("button", name="Scribe", exact=True).click()
    page.get_by_label("Additional encounter notes").fill("Robert Chen reports dizziness.")
    select_patient(page, "Eleanor Vance")
    select_patient(page, "Robert Chen")
    assert page.get_by_label("Additional encounter notes").input_value() == "Robert Chen reports dizziness."
    generate(page)
    revised = "Physician revised <script>window.__xss = 1</script>"
    page.get_by_label("Assessment", exact=True).fill(revised)
    page.get_by_label("Plan (one item per line)").fill("Review medication orders.\nDiscuss alerts with physician.")
    page.get_by_role("button", name="Continue to sign-off").click()
    assert page.locator("#reviewBody").inner_text().count(revised) == 1
    assert page.evaluate("window.__xss") is None
    page.get_by_role("button", name="Sign and commit", exact=True).click()
    assert page.locator("#signError").is_visible()
    assert page.locator("#ackCritical").evaluate("element => element === document.activeElement")
    page.keyboard.press("Space")
    page.get_by_label(re.compile("Sign-off notes")).fill("Critical alerts reviewed; medication orders addressed separately.")
    select_patient(page, "Eleanor Vance")
    select_patient(page, "Robert Chen")
    assert page.locator("#ackCritical").is_checked()
    assert "orders addressed" in page.locator("#signNotes").input_value()
    page.get_by_role("button", name="Scribe", exact=True).click()
    assert page.get_by_label("Assessment", exact=True).input_value() == revised
    page.get_by_role("button", name="Continue to sign-off").click()

    page.route("**/api/consult/*/resume", lambda route: route.fulfill(
        status=503, content_type="application/json", body=json.dumps({"detail": "Temporary review failure"})), times=1)
    page.get_by_role("button", name="Sign and commit", exact=True).click()
    page.get_by_text("Temporary review failure", exact=True).wait_for()
    assert page.locator("#ackCritical").is_checked()
    assert "orders addressed" in page.locator("#signNotes").input_value()
    page.get_by_role("button", name="Sign and commit", exact=True).click()
    link = page.get_by_role("link", name="Download review receipt")
    link.wait_for()
    with page.expect_download() as download:
        link.click()
    receipt = json.loads(Path(download.value.path()).read_text())
    assert receipt["status"] == "EDITED" and receipt["ehr_committed"] is True
    assert receipt["final_soap"]["assessment"] == revised
    assert receipt["review_receipt"]["critical_alerts_acknowledged"]
    assert set(receipt["review_receipt"]["changes"]) == {"assessment", "plan"}
    assert "phi_vault" not in receipt and "raw_bundle" not in receipt
    screenshots = Path(os.getenv("CAREFLOW_SCREENSHOT_DIR", str(tmp_path)))
    screenshots.mkdir(exist_ok=True, parents=True)
    page.screenshot(path=str(screenshots / "careflow-review-desktop.png"), full_page=True)


def test_mobile_reject_remains_available_without_acknowledgement(page, tmp_path):
    page.set_viewport_size({"width": 390, "height": 844})
    page.emulate_media(reduced_motion="reduce")
    select_patient(page, "Robert Chen")
    generate(page)
    page.get_by_role("button", name="Continue to sign-off").click()
    assert not page.locator("#ackCritical").is_checked()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.get_by_role("button", name="Reject draft", exact=True).click()
    page.get_by_text(re.compile("Draft rejected. Nothing was committed.")).wait_for()
    assert page.get_by_role("link", name="Download review receipt").is_visible()
    screenshots = Path(os.getenv("CAREFLOW_SCREENSHOT_DIR", str(tmp_path)))
    screenshots.mkdir(exist_ok=True, parents=True)
    page.screenshot(path=str(screenshots / "careflow-review-mobile.png"), full_page=True)


@pytest.mark.parametrize("failure", [False, True])
def test_notes_stay_with_rendered_patient_during_chart_loading(page, failure):
    page.get_by_role("button", name="Scribe", exact=True).click()
    pending = []
    page.route("**/api/patients/pat-002", lambda route: pending.append(route))
    page.get_by_role("button", name=re.compile("Marcus Holloway")).click()
    page.wait_for_function("state.active === 'pat-002'")
    page.get_by_label("Additional encounter notes").fill("Eleanor reports dizziness.")
    page.wait_for_timeout(100)
    assert pending
    if failure:
        pending[0].fulfill(status=503, content_type="application/json", body='{"detail":"Chart unavailable"}')
        page.get_by_role("alert").filter(has_text="Chart unavailable").wait_for()
    else:
        pending[0].continue_()
        page.get_by_role("heading", name="Marcus Holloway").wait_for()
        assert page.get_by_label("Additional encounter notes").input_value() == ""
    assert page.evaluate("state.encounterNotes['pat-002'] || ''") == ""
    select_patient(page, "Eleanor Vance")
    assert page.get_by_label("Additional encounter notes").input_value() == "Eleanor reports dizziness."
