from __future__ import annotations

import base64
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

InputModality = Literal["text", "image", "multimodal"]
InteractionMode = Literal["education", "companion", "parent"]
ActType = Literal["direct", "retrieve_knowledge", "read_memory"]
ConfidenceLevel = Literal["high", "medium", "low"]


def _max_upload_image_bytes() -> int:
    return int(os.getenv("MAX_UPLOAD_IMAGE_BYTES", str(4 * 1024 * 1024)))


class ChatRequest(BaseModel):
    text: str | None = Field(
        default=None,
        description="Short child input text.",
        max_length=1000,
    )
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image data without line breaks.",
    )
    image_url: HttpUrl | None = Field(
        default=None,
        description="Publicly accessible image URL.",
    )
    image_mime_type: str | None = Field(
        default=None,
        description="Image mime type, such as image/jpeg or image/png.",
    )
    age_hint: str | None = Field(
        default="3-8",
        description="Approximate child age range.",
        max_length=20,
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for short-term memory.",
        max_length=100,
    )
    mode: InteractionMode = Field(
        default="education",
        description="Interaction mode selected by frontend.",
    )
    profile_id: str | None = Field(
        default=None,
        description="Optional profile identifier for lightweight profile memory.",
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_input(self) -> "ChatRequest":
        has_text = bool(self.text and self.text.strip())
        has_base64_image = bool(self.image_base64 and self.image_base64.strip())
        has_url_image = self.image_url is not None
        has_image = has_base64_image or has_url_image
        if not has_text and not has_image:
            raise ValueError("Either text, image_base64, or image_url must be provided.")
        if has_base64_image and not self.image_mime_type:
            raise ValueError("image_mime_type is required when image_base64 is provided.")
        if has_base64_image:
            try:
                decoded_image = base64.b64decode(self.image_base64 or "", validate=True)
            except Exception as exc:
                raise ValueError(
                    "image_base64 must be valid base64 data, not a placeholder string."
                ) from exc
            if len(decoded_image) > _max_upload_image_bytes():
                raise ValueError(
                    "image_base64 payload is too large. Please upload a smaller image or use image_url."
                )
            if self.image_mime_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError(
                    "image_mime_type must be one of image/png, image/jpeg, image/webp."
                )
        if has_url_image and self.image_mime_type:
            raise ValueError("image_mime_type should not be provided when using image_url.")
        return self


class ReactInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str
    selected_act: ActType
    tool_name: str | None = None
    tool_success: bool
    reason: str


class GroundingSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    score: float
    snippet: str


class GroundingInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    used_rag: bool
    sources: list[GroundingSource] = Field(default_factory=list)


class MemoryInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_updated: bool
    profile_updated: bool
    written_types: list[str] = Field(default_factory=list)
    consolidated_count: int = 0
    forgotten_count: int = 0


class ChatMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_mode: Literal["llm"]
    confidence: ConfidenceLevel
    safety_notes: str = ""
    used_image: bool
    dialogue_stage: Literal["responded"]
    workflow_trace: list[str]
    input_modality: InputModality
    route_reason: str = ""
    rag_enabled: bool


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    mode: InteractionMode
    message: str
    follow_up_question: str | None = None
    topic: str | None = None
    react: ReactInfo
    grounding: GroundingInfo
    memory: MemoryInfo
    metadata: ChatMetadata
