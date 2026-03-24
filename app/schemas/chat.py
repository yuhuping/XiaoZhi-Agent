import base64
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


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
        description="Optional session identifier for future extensibility.",
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
                base64.b64decode(self.image_base64 or "", validate=True)
            except Exception as exc:
                raise ValueError(
                    "image_base64 must be valid base64 data, not a placeholder string."
                ) from exc
            if self.image_mime_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError(
                    "image_mime_type must be one of image/png, image/jpeg, image/webp."
                )
        if has_url_image and self.image_mime_type:
            raise ValueError("image_mime_type should not be provided when using image_url.")
        return self


class ChatMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_mode: Literal["mock", "openai"]
    confidence: Literal["high", "medium", "low"]
    safety_notes: str = ""
    used_image: bool


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str
    explanation: str
    follow_up_question: str
    metadata: ChatMetadata
