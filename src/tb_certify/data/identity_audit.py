"""Read conservative patient-linkage status from the generated audit."""

from __future__ import annotations

import json
from pathlib import Path


def patient_level_supported(project_root: Path) -> bool:
    """Return the audit's patient-level support decision; missing audits fail closed."""
    audit_path = project_root / "artifacts" / "audits" / "image_identity.json"
    if not audit_path.is_file():
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return audit.get("patient_level_supported") is True
