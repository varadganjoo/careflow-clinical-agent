"""Automated Screenshot Capture Script for CareFlow Clinical Agent Documentation.
Uses Playwright with native Edge to capture high-res production screenshots.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def capture_all():
    print("Launching Playwright with msedge...")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1050}, device_scale_factor=1.5)
        page = context.new_page()

        print("Navigating to http://127.0.0.1:8010...")
        page.goto("http://127.0.0.1:8010", wait_until="networkidle")
        time.sleep(3)  # Allow consultation to complete

        # 1. Patient 1: Eleanor Vance (Stage 2 Hypertension)
        print("Capturing 01_physician_portal_hypertension.png...")
        page.screenshot(path=str(OUTPUT_DIR / "01_physician_portal_hypertension.png"), full_page=False)

        # 2. Patient 3: Robert Chen (Drug Interaction & Hyperkalemia)
        print("Switching to Patient 3 (Robert Chen)...")
        cards = page.query_selector_all(".patient-card")
        if len(cards) >= 3:
            cards[2].click()
            time.sleep(3)
            print("Capturing 02_safety_guardrail_critical_alert.png...")
            page.screenshot(path=str(OUTPUT_DIR / "02_safety_guardrail_critical_alert.png"), full_page=False)

        # 3. Patient 2: Marcus Holloway (Diabetes & CKD Metformin Contraindication)
        print("Switching to Patient 2 (Marcus Holloway)...")
        cards = page.query_selector_all(".patient-card")
        if len(cards) >= 2:
            cards[1].click()
            time.sleep(3)
            print("Capturing 03_metformin_contraindication_ada.png...")
            page.screenshot(path=str(OUTPUT_DIR / "03_metformin_contraindication_ada.png"), full_page=False)

        # 4. Return to Patient 1 and perform Physician Sign-Off
        print("Testing Physician Sign-Off on Patient 1...")
        cards = page.query_selector_all(".patient-card")
        if len(cards) >= 1:
            cards[0].click()
            time.sleep(2)
            page.click("#btn-approve")
            time.sleep(2)
            print("Capturing 04_physician_signoff_ehr_commit.png...")
            page.screenshot(path=str(OUTPUT_DIR / "04_physician_signoff_ehr_commit.png"), full_page=False)

        browser.close()
        print("All screenshots captured successfully in:", OUTPUT_DIR)


if __name__ == "__main__":
    capture_all()
