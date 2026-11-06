"""Shared validation for HTTP and direct LangGraph physician decisions."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NoteText = Annotated[str, Field(max_length=5000)]
CodeText = Annotated[str, Field(max_length=100)]


class SOAPEdits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subjective: NoteText | None = None
    objective: NoteText | None = None
    assessment: NoteText | None = None
    plan: list[NoteText] | None = Field(None, max_length=30)
    icd10_codes: list[CodeText] | None = Field(None, max_length=30)
    cpt_codes: list[CodeText] | None = Field(None, max_length=30)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "edit", "reject"]
    edits: SOAPEdits = Field(default_factory=SOAPEdits)
    notes: str = Field("", max_length=2000)
    acknowledge_critical_alerts: bool = Field(False, strict=True)

    @model_validator(mode="after")
    def validate_edits(self) -> Self:
        edits = self.edits.model_dump(exclude_none=True)
        if self.action == "edit" and not edits:
            raise ValueError("Action 'edit' requires at least one edited SOAP field.")
        if self.action != "edit" and edits:
            raise ValueError("SOAP edits require action 'edit'.")
        return self

    def validate_alerts(self, alerts: list[dict]) -> None:
        # ASVS 2.2.2: this check also runs inside the graph, beyond HTTP validation.
        if self.action != "reject" and any(a["severity"] == "CRITICAL" for a in alerts):
            if not self.acknowledge_critical_alerts or not self.notes.strip():
                raise ValueError("Acknowledge critical alerts and record a sign-off rationale before committing.")
