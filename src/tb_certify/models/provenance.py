"""Provenance gate for pretrained encoders used in cross-site experiments."""

from __future__ import annotations

from dataclasses import dataclass


class EncoderEligibilityError(ValueError):
    """Raised when model license or pretraining overlap is not auditable."""


@dataclass(frozen=True)
class EncoderProvenance:
    model_name: str
    revision: str
    license_name: str | None
    license_verified: bool
    pretraining_sites: tuple[str, ...] | None
    pretraining_data_reviewed: bool

    def validate_for_target(self, target_site: str) -> None:
        if not self.model_name.strip() or not self.revision.strip():
            raise EncoderEligibilityError("Model name and immutable revision are required.")
        if not self.license_name or not self.license_verified:
            raise EncoderEligibilityError("Encoder license must be documented and verified.")
        if self.pretraining_sites is None or not self.pretraining_data_reviewed:
            raise EncoderEligibilityError(
                "Encoder pretraining data must be reviewed before cross-site evaluation."
            )
        if not self.pretraining_sites:
            raise EncoderEligibilityError(
                "At least one reviewed pretraining source must be documented."
            )
        target_key = self._normalize_site(target_site)
        training_keys = {self._normalize_site(site) for site in self.pretraining_sites}
        if target_key in training_keys:
            raise EncoderEligibilityError(
                f"Encoder pretraining includes held-out target site {target_site!r}."
            )

    @staticmethod
    def _normalize_site(site: str) -> str:
        return "".join(character for character in site.casefold() if character.isalnum())
