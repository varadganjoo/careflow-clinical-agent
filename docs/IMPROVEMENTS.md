# Safety-aware review and review receipts

## Scope and acceptance

Improve the existing fictional-patient workstation using FastAPI, Pydantic,
LangGraph and the existing static interface. No new runtime dependency or clinical
rule is needed. The selected strategy makes human review observable: critical
alerts require acknowledgement and a rationale, every decision produces a receipt,
and unsent work stays attached to the patient while navigating.

Acceptance criteria:

- Approve/edit requires an explicit action; edits are bounded SOAP fields only.
- Critical alerts require explicit acknowledgement and nonblank sign-off notes.
  Rejection always remains available. Both HTTP and direct graph resumes enforce this.
- A receipt records the UTC decision time, actual before/after changes, rationale
  and critical alerts acknowledged; it is available only after a decision.
- Receipts/status omit the raw chart and PHI token vault; responses are not cached.
- Encounter notes pass through de-identification before reaching the model.
- Switching patients preserves SOAP edits and review inputs in page memory.
- Tests cover invalid input, direct graph bypass, decision replay, failure paths,
  patient switching, mobile layout and keyboard acknowledgement.

## Implementation plan

1. Add regression tests and run them against the existing implementation.
2. Share review request validation between API and graph; create review receipts
   at the physician gate and serialize HTTP sign-off per encounter.
3. Add the receipt endpoint and remove raw checkpoint data from status responses.
4. Add acknowledgement, rationale guidance, receipt download and in-memory editing
   to the existing interface; preserve the current visual system.
5. Run full tests, local browser journeys, dependency and secret checks; inspect
   the entire diff, record release evidence and push the verified revision.

Connected paths: `app/main.py`, `app/graph.py`, new `app/review.py`, the static UI,
API/graph regressions, browser checks, CI and README. MCP does not resume graphs
and retains its existing tools. Existing safe-patient approve/edit calls remain
valid; critical-patient calls must supply the new acknowledgement and rationale.

## Security model

Assets are fictional charts, generated notes, clinician edits and review receipts.
Actors are public demo visitors, local MCP callers and external model providers.
There is no authenticated clinician identity or tenant boundary in this demo;
receipts document a workflow decision and are not authenticated signatures.
Do not store real patient data. A production system needs identity, resource
authorization, durable encrypted checkpoints and retention policy.

Untrusted request/model content must never decide whether a note is committed.
ASVS 2.2.1/2.2.2: validate review inputs at both API and graph boundaries.
ASVS 2.3.1/2.3.4: explicit ordered decision and per-encounter HTTP locking.
Keep existing deterministic safety rules authoritative; acknowledgement records
review and does not resolve an alert or establish clinical appropriateness.

## Procedures and exclusions

Applied enterprise-app-dev routing/release gate, Minimalist MVP scope, Ponytail
full/review, Superpowers design/planning/TDD/verification, Spellbook FastAPI and
accessibility, OWASP guidance/code review/secrets/dependency and focused AI review,
Sentry diff review and Anthropic local browser testing. Implementation proceeds
under the user's explicit instruction to choose, implement, test and push.

Design Taste is N/A: this is a dense workstation using its existing design system.
Mobile apps, databases/migrations and infrastructure deployment are N/A. No full
audit or compliance certification was requested. Cloudflare guidance was read;
its isolated audit execution is BLOCKED because the full resource-limited,
read-only toolchain sandbox is unavailable. Ordinary authorized application tests
are recorded separately and do not substitute for an independent security audit.

## Release evidence — November 5, 2026

Source: the commit containing this document, based on `b09b324`. Environment:
Windows, Python 3.12.12, isolated `.venv`, locked runtime dependencies, Playwright
1.64.1 with local Chromium. No clinical thresholds or model selection changed.

| Control | Status | Evidence |
| --- | --- | --- |
| Unit/integration and rendered journeys | PASS | `python -m pytest -q --tb=short` with `CAREFLOW_BROWSER_TESTS=1`: **73 passed in 13.14s**. Default run: 69 passed, 4 opt-in browser tests skipped. |
| Human decision and input limits | PASS | HTTP/direct graph rejection of invalid actions, extra fields, oversized edits and missing critical acknowledgement/rationale; valid edit/reject recovery after invalid direct resumes. |
| Concurrent decisions | PASS | Simultaneous approve/reject requests produce one success and one 409; subsequent decisions cannot overwrite the receipt. Locking is per process, consistent with the existing in-memory demo. |
| Receipt and privacy projection | PASS | Pre-decision download returns 409; completed receipt contains actual edits and UTC time, with `no-store` and attachment headers. Status/receipt omit raw bundle and token vault. |
| Model boundary | PASS (application control) | Encounter notes are de-identified; hostile model text cannot commit a note or suppress critical-review requirements. Browser test renders script-shaped note text inertly. This does not establish model robustness or clinical accuracy. |
| Browser experience | PASS | Real local API with deterministic test scribe: edit/patient-switch/retry/download, keyboard checkbox, mobile rejection, reduced motion, delayed and failed chart loading. No browser JavaScript errors; no mobile page overflow. Desktop/mobile screenshots were inspected. |
| Dependency advisories | PASS | `pip-audit 2.10.1 -r requirements.lock --no-deps --disable-pip`: 68 dependencies checked, zero known advisories. Fresh online advisory query on the date above; Windows-applicable markers only locally. CI repeats the scan on Linux. |
| Secret review | PASS (bounded) | Manual high-confidence provider-key/private-key/credential-URL patterns across 40 current tracked/nonignored new text files: zero findings. `.env` stays ignored. No historical scan, specialized secret scanner or credential-validity probes. |
| Syntax/configuration | PASS | Python `compileall`, Node syntax check for both inline scripts, workflow YAML parsing and `git diff --check`. No standalone build/lint/type command is configured. |
| Independent diff review | PASS after fixes | Reviewer reproduced two issues: patient-note ownership during chart loading and invalid graph-resume recovery. Both fixed, with passing regressions. |
| Continuous testing | PASS (configured) | Python 3.11/3.12 unit jobs, Chromium browser job, locked-dependency advisory job; workflow token has read-only repository contents permission. Remote job results must be checked after push. |
| Live provider generation | BLOCKED | Neither `GEMINI_API_KEY` nor `GROQ_API_KEY` is configured in this checkout. No paid quota was spent; model failures/fallback are tested with controlled responses. |
| Clinical/production approval | BLOCKED | Fictional demo only: no authenticated clinician identity, tenant isolation, durable checkpoint/receipt storage, clinical validation or external EHR integration. Owner: repository maintainer. |
| Full audit/compliance | N/A | Not requested. Cloudflare isolated audit execution remains unavailable; ordinary application tests are not an audit certification. |
| Database migration, new infrastructure, design redesign | N/A | No database/infrastructure change or supported Design Taste page redesign. The workstation retains its existing light-only design. |

The receipt is an in-memory record, not a tamper-proof audit log. Page reload
loses unsent edits; server recycle loses sessions/receipts. Download a receipt
before recycling. Existing regex de-identification still does not cover all
patient identifiers. Public demo endpoints and provider quota controls require
separate identity/abuse work before any real-patient deployment. No production
readiness or medical-device compliance is claimed.
