"""Experiment options specific to OpenCode's optional reviewer."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OpenCodeSubagentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_name: str = Field(min_length=1)
    server_url: str | None = None
    model_max_len: int = Field(default=262000, ge=4, strict=True)
    description: str = Field(
        default="Consult this stronger reviewer when stuck, uncertain about a solution, "
        "or needing a code review before finishing.",
        min_length=1,
    )
    prompt: str = Field(
        default="Review the primary agent's question and code. Identify mistakes and "
        "suggest concrete next steps. Return advice without modifying files.",
        min_length=1,
    )

    @field_validator("server_url")
    @classmethod
    def validate_endpoint(cls, value: str | None) -> str | None:
        if value is not None and value != "openrouter":
            from urllib.parse import urlsplit

            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("subagent server_url must be HTTP(S) or 'openrouter'")
        return value
