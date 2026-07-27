"""Verify the public, claim-linked paper artifact manifest without network access.

The manifest is deliberately separate from the result registry. The registry says what the
study concluded; this file checks that every selected public evidence receipt has an exact byte
identity, an auditable producer, and an explicit registration/provenance tier. Local receipts
must be committed below ``paper_artifacts/``. Remote receipts must name a versioned HTTPS object;
their hashes and sizes are recorded but cannot be rechecked without downloading, so this command
only validates their metadata and URL syntax.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Collection, Mapping
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geoprobe.registry import load_registry  # noqa: E402

SCHEMA_VERSION = 2
DEFAULT_MANIFEST = REPO_ROOT / "paper_artifacts" / "manifest.json"
DEFAULT_REGISTRY = REPO_ROOT / "docs" / "results_registry.yaml"

REGISTRATION_TIERS = {
    "prospective",
    "frozen_exploratory",
    "post_hoc_registered",
    "retrospective_synthesis",
    "unregistered_descriptive",
}
EXACT_IDENTITY_REQUIRED_CLAIMS = {"C10", "C11", "C13"}
CONFIRMATORY_CLAIMS = {"C5"}
LOCATION_KINDS = {"tracked_path", "immutable_url"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMMUTABLE_ID_RE = re.compile(r"^[a-z][a-z0-9._-]*:([^\s]+)$")
FORBIDDEN_PATH_PARTS = {
    ".private",
    "private",
    "ops",
    "operations",
    "runpod_results",
    "secrets",
    "credentials",
}
FORBIDDEN_COMMAND_MARKERS = (
    ".private",
    "../",
    "runpod",
    "ssh ",
    "/users/",
    "/home/",
    "/workspace/",
    "/private/",
    "/tmp/",
    "credentials",
    "approval.json",
)
ARCHIVE_EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
MUTABLE_MODEL_REVISIONS = {"", "main", "master", "head", "latest", "default"}
MUTABLE_URL_PARTS = {"latest", "main", "master", "head", "current"}

REQUIRED_ENTRY_FIELDS = {
    "artifact_id",
    "result_id",
    "claim_id",
    "location",
    "sha256",
    "byte_size",
    "producer",
    "producer_command",
    "input_hashes",
    "model_dependent",
    "model_identity",
    "provenance",
}
TOP_LEVEL_FIELDS = {"schema_version", "registry_path", "artifacts"}
REQUIRED_PROVENANCE_FIELDS = {
    "registration_tier",
    "confirmatory",
    "lifecycle_status",
    "source_commit",
    "source_tree_dirty",
    "note",
}
MODEL_IDENTITY_FIELDS = {
    "model_id",
    "revision",
    "weights_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "note",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of ``path`` using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("manifest root must be a JSON object")
    return value


def _git_tracked_paths(repo_root: Path) -> tuple[set[str], str | None]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        return set(), f"could not enumerate Git-tracked files: {detail or 'git ls-files failed'}"
    return {
        item.decode(errors="surrogateescape")
        for item in completed.stdout.split(b"\0")
        if item
    }, None


def _archive_regular_paths(repo_root: Path) -> set[str]:
    """Return regular, non-symlink files available in an extracted archive.

    This is only an archive-membership view. It deliberately does not claim that the files have
    Git provenance; the manifest's byte, producer, path-safety, and provenance checks still run.
    """
    paths: set[str] = set()
    for path in repo_root.rglob("*"):
        relative = path.relative_to(repo_root)
        if any(part in ARCHIVE_EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.is_file() and not path.is_symlink():
            paths.add(relative.as_posix())
    return paths


def _safe_relative_path(value: object, *, prefix: str | None = None) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "must be a non-empty repository-relative POSIX path"
    if "\\" in value:
        return None, "must use POSIX separators"
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        return None, "must be a normalized repository-relative POSIX path"
    if any(part in {"", ".", ".."} for part in path.parts):
        return None, "must not contain '.', '..', or empty path components"
    lowered = [part.lower() for part in path.parts]
    if any(part in FORBIDDEN_PATH_PARTS or part.startswith("runpod") for part in lowered):
        return None, "references a private or operational path"
    if prefix is not None and (not path.parts or path.parts[0] != prefix):
        return None, f"must be below {prefix}/"
    return path.as_posix(), None


def _validate_url(location: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    url = location.get("url")
    immutable_id = location.get("immutable_id")
    if not isinstance(url, str) or not url:
        return ["location.url must be a non-empty string"]
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        errors.append("location.url must be an absolute HTTPS URL")
    if parsed.username or parsed.password:
        errors.append("location.url must not contain credentials")
    if parsed.fragment:
        errors.append("location.url must not contain a fragment")
    if parsed.query:
        errors.append("location.url must not contain a mutable or signed query string")
    url_parts = [part.lower() for part in PurePosixPath(parsed.path).parts]
    if any(part in FORBIDDEN_PATH_PARTS or part.startswith("runpod") for part in url_parts):
        errors.append("location.url references a private or operational path")
    if any(part in MUTABLE_URL_PARTS for part in url_parts):
        errors.append("location.url contains a mutable path component")
    if not isinstance(immutable_id, str) or not (match := IMMUTABLE_ID_RE.fullmatch(immutable_id)):
        errors.append("location.immutable_id must be a namespaced identifier such as doi:10.x/y")
    elif match.group(1) not in url:
        errors.append("the immutable identifier value must appear in location.url")
    return errors


def _validate_model(entry: Mapping[str, Any], *, claim_id: object) -> list[str]:
    errors: list[str] = []
    dependent = entry.get("model_dependent")
    identity = entry.get("model_identity")
    if not isinstance(dependent, bool):
        return ["model_dependent must be a boolean"]
    if not dependent:
        errors.append("model_dependent must be true for every paper evidence receipt")
        if identity is not None:
            errors.append("model_identity must be null when model_dependent is false")
        return errors
    if not isinstance(identity, Mapping):
        return ["model_identity must be an object when model_dependent is true"]
    extra = set(identity) - MODEL_IDENTITY_FIELDS
    if extra:
        errors.append(f"model_identity has unknown fields: {sorted(extra)}")
    missing = MODEL_IDENTITY_FIELDS - set(identity)
    if missing:
        errors.append(f"model_identity missing required fields: {sorted(missing)}")
    model_id = identity.get("model_id")
    revision = identity.get("revision")
    note = identity.get("note")
    if not isinstance(model_id, str) or not model_id.strip():
        errors.append("model_identity.model_id must be a non-empty string")
    if note is not None and not isinstance(note, str):
        errors.append("model_identity.note must be a string")

    identity_is_exact = claim_id in EXACT_IDENTITY_REQUIRED_CLAIMS
    missing_identity: list[str] = []
    if revision is None:
        missing_identity.append("revision")
    elif not isinstance(revision, str) or revision.strip().lower() in MUTABLE_MODEL_REVISIONS:
        errors.append("model_identity.revision must be an immutable, non-default revision")
    for field in ("weights_sha256", "config_sha256", "tokenizer_sha256"):
        value = identity.get(field)
        if value is None:
            missing_identity.append(field)
        elif not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            errors.append(f"model_identity.{field} must be a lowercase SHA-256 or null")
    if identity_is_exact and missing_identity:
        errors.append(
            f"claim {claim_id} requires exact model identity; missing: "
            f"{', '.join(missing_identity)}"
        )
    elif missing_identity and (not isinstance(note, str) or not note.strip()):
        errors.append(
            "model_identity.note must explain missing identity fields: "
            f"{', '.join(missing_identity)}"
        )
    return errors


def _validate_provenance(entry: Mapping[str, Any], *, claim_id: object) -> list[str]:
    provenance = entry.get("provenance")
    if not isinstance(provenance, Mapping):
        return ["provenance must be an object"]
    errors: list[str] = []
    missing = sorted(REQUIRED_PROVENANCE_FIELDS - set(provenance))
    if missing:
        errors.append(f"provenance missing required fields: {', '.join(missing)}")
    extra = sorted(set(provenance) - REQUIRED_PROVENANCE_FIELDS)
    if extra:
        errors.append(f"provenance has unknown fields: {', '.join(extra)}")
    tier = provenance.get("registration_tier")
    if tier not in REGISTRATION_TIERS:
        errors.append(f"provenance.registration_tier must be one of {sorted(REGISTRATION_TIERS)}")
    confirmatory = provenance.get("confirmatory")
    if not isinstance(confirmatory, bool):
        errors.append("provenance.confirmatory must be a boolean")
    elif confirmatory:
        if tier != "prospective":
            errors.append("confirmatory evidence must use the prospective registration tier")
        if claim_id not in CONFIRMATORY_CLAIMS:
            errors.append(f"claim {claim_id!r} cannot be marked confirmatory")
    lifecycle = provenance.get("lifecycle_status")
    if not isinstance(lifecycle, str) or not lifecycle.strip():
        errors.append("provenance.lifecycle_status must be a non-empty string")
    source_commit = provenance.get("source_commit")
    if source_commit is not None and (
        not isinstance(source_commit, str) or not COMMIT_RE.fullmatch(source_commit)
    ):
        errors.append("provenance.source_commit must be a lowercase 40-character Git hash or null")
    source_dirty = provenance.get("source_tree_dirty")
    if source_dirty is not None and not isinstance(source_dirty, bool):
        errors.append("provenance.source_tree_dirty must be a boolean or null")
    note = provenance.get("note")
    if not isinstance(note, str):
        errors.append("provenance.note must be a string")
    elif (source_commit is None or source_dirty is None) and not note.strip():
        errors.append("provenance.note must explain missing source provenance")
    return errors


def _validate_input_hashes(entry: Mapping[str, Any]) -> list[str]:
    inputs = entry.get("input_hashes")
    if not isinstance(inputs, Mapping):
        return ["input_hashes must be an object (use {} only when there are no hashed inputs)"]
    errors: list[str] = []
    for name, digest in inputs.items():
        if not isinstance(name, str) or not name.strip():
            errors.append("input_hashes keys must be non-empty strings")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            errors.append(f"input_hashes[{name!r}] must be a lowercase SHA-256")
    return errors


def _command_references_producer(command: str, producer: str) -> tuple[bool, str | None]:
    """Return whether a shell-like command names the declared producer as one token."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        return False, f"producer_command is not valid shell token syntax: {exc}"
    return producer in tokens, None


def _validate_embedded_receipt(
    artifact_path: Path,
    *,
    label: str,
    claim_id: object,
    producer: str | None,
    producer_digest: str | None,
) -> list[str]:
    """Bind a tracked JSON receipt to its claim and current producer bytes."""
    try:
        receipt = json.loads(artifact_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"{label}: tracked JSON receipt is not valid JSON: {exc}"]
    if not isinstance(receipt, Mapping):
        return [f"{label}: tracked JSON receipt root must be an object"]

    errors: list[str] = []
    if receipt.get("claim_id") != claim_id:
        errors.append(
            f"{label}: embedded claim_id {receipt.get('claim_id')!r} "
            f"does not match manifest claim_id {claim_id!r}"
        )
    if producer is not None and receipt.get("producer") != producer:
        errors.append(
            f"{label}: embedded producer {receipt.get('producer')!r} "
            f"does not match manifest producer {producer!r}"
        )
    embedded_digest = receipt.get("producer_sha256")
    if producer_digest is not None and embedded_digest != producer_digest:
        errors.append(
            f"{label}: embedded producer_sha256 {embedded_digest!r} does not match "
            f"current producer SHA-256 {producer_digest}"
        )
    return errors


def _validate_entry(
    entry: object,
    *,
    index: int,
    repo_root: Path,
    tracked_paths: Collection[str],
    result_claims: Mapping[str, str | None],
    claim_ids: Collection[str],
) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"artifacts[{index}] must be an object"]
    artifact_id = entry.get("artifact_id")
    label = f"artifact {artifact_id!r}" if artifact_id else f"artifacts[{index}]"
    errors: list[str] = []
    missing = sorted(REQUIRED_ENTRY_FIELDS - set(entry))
    if missing:
        errors.append(f"{label}: missing required fields: {', '.join(missing)}")
    extra = sorted(set(entry) - REQUIRED_ENTRY_FIELDS)
    if extra:
        errors.append(f"{label}: unknown fields: {', '.join(extra)}")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        errors.append(f"{label}: artifact_id must be a non-empty string")

    result_id = entry.get("result_id")
    claim_id = entry.get("claim_id")
    if not isinstance(result_id, str) or not result_id.strip():
        errors.append(f"{label}: result_id must be a non-empty string")
    elif result_id not in result_claims:
        errors.append(f"{label}: result_id {result_id!r} is not in the result registry")
    if not isinstance(claim_id, str) or not claim_id.strip():
        errors.append(f"{label}: claim_id must be a non-empty string")
    elif claim_id not in claim_ids:
        errors.append(f"{label}: claim_id {claim_id!r} is not in the result registry")
    if isinstance(result_id, str) and result_id in result_claims and (
        result_claims[result_id] != claim_id
    ):
        errors.append(
            f"{label}: result {result_id!r} belongs to claim "
            f"{result_claims[result_id]!r}, not {claim_id!r}"
        )

    digest = entry.get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        errors.append(f"{label}: sha256 must be a lowercase 64-character digest")
    byte_size = entry.get("byte_size")
    if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size <= 0:
        errors.append(f"{label}: byte_size must be a positive integer")

    producer, producer_error = _safe_relative_path(entry.get("producer"))
    producer_digest: str | None = None
    if producer_error:
        errors.append(f"{label}: producer {producer_error}")
    elif producer is not None:
        producer_path = repo_root / producer
        if producer not in tracked_paths:
            errors.append(f"{label}: producer is not Git-tracked: {producer}")
        if not producer_path.is_file() or producer_path.is_symlink():
            errors.append(f"{label}: producer is not a regular repository file: {producer}")
        else:
            producer_digest = sha256_file(producer_path)

    command = entry.get("producer_command")
    if not isinstance(command, str) or not command.strip():
        errors.append(f"{label}: producer_command must be a non-empty string")
    elif any(marker in command.lower() for marker in FORBIDDEN_COMMAND_MARKERS):
        errors.append(f"{label}: producer_command contains a private or operational reference")
    elif producer is not None:
        references_producer, command_error = _command_references_producer(command, producer)
        if command_error:
            errors.append(f"{label}: {command_error}")
        elif not references_producer:
            errors.append(
                f"{label}: producer_command must reference declared producer {producer!r}"
            )

    location = entry.get("location")
    if not isinstance(location, Mapping):
        errors.append(f"{label}: location must be an object")
    else:
        kind = location.get("kind")
        if kind not in LOCATION_KINDS:
            errors.append(f"{label}: location.kind must be one of {sorted(LOCATION_KINDS)}")
        elif kind == "tracked_path":
            extra = set(location) - {"kind", "path"}
            if extra:
                errors.append(f"{label}: tracked_path location has unknown fields: {sorted(extra)}")
            path, path_error = _safe_relative_path(location.get("path"), prefix="paper_artifacts")
            if path_error:
                errors.append(f"{label}: location.path {path_error}")
            elif path is not None:
                artifact_path = repo_root / path
                if path not in tracked_paths:
                    errors.append(f"{label}: artifact is not Git-tracked: {path}")
                if not artifact_path.is_file() or artifact_path.is_symlink():
                    errors.append(f"{label}: artifact is not a regular repository file: {path}")
                else:
                    if isinstance(byte_size, int) and not isinstance(byte_size, bool):
                        actual_size = artifact_path.stat().st_size
                        if actual_size != byte_size:
                            errors.append(
                                f"{label}: byte_size mismatch for {path}: "
                                f"manifest {byte_size}, actual {actual_size}"
                            )
                    if isinstance(digest, str) and SHA256_RE.fullmatch(digest):
                        actual_digest = sha256_file(artifact_path)
                        if actual_digest != digest:
                            errors.append(
                                f"{label}: sha256 mismatch for {path}: "
                                f"manifest {digest}, actual {actual_digest}"
                            )
                    if artifact_path.suffix.lower() == ".json":
                        errors.extend(
                            _validate_embedded_receipt(
                                artifact_path,
                                label=label,
                                claim_id=claim_id,
                                producer=producer,
                                producer_digest=producer_digest,
                            )
                        )
        elif kind == "immutable_url":
            extra = set(location) - {"kind", "url", "immutable_id"}
            if extra:
                errors.append(f"{label}: immutable_url location has unknown fields: {sorted(extra)}")
            errors.extend(f"{label}: {error}" for error in _validate_url(location))

    errors.extend(f"{label}: {error}" for error in _validate_input_hashes(entry))
    errors.extend(
        f"{label}: {error}" for error in _validate_model(entry, claim_id=claim_id)
    )
    errors.extend(
        f"{label}: {error}" for error in _validate_provenance(entry, claim_id=claim_id)
    )
    return errors


def validate_manifest(
    manifest: object,
    *,
    repo_root: Path = REPO_ROOT,
    registry: Mapping[str, Any] | None = None,
    tracked_paths: Collection[str] | None = None,
) -> list[str]:
    """Return all manifest errors; perform no network requests."""
    if not isinstance(manifest, Mapping):
        return ["manifest root must be an object"]
    errors: list[str] = []
    extra = sorted(set(manifest) - TOP_LEVEL_FIELDS)
    if extra:
        errors.append(f"manifest has unknown fields: {', '.join(extra)}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )
    registry_path, registry_path_error = _safe_relative_path(manifest.get("registry_path"))
    if registry_path_error:
        errors.append(f"registry_path {registry_path_error}")
    elif registry_path != "docs/results_registry.yaml":
        errors.append("registry_path must be docs/results_registry.yaml")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("artifacts must be a list")
        artifacts = []

    if registry is None:
        try:
            registry = load_registry(repo_root / (registry_path or "docs/results_registry.yaml"))
        except (OSError, ValueError) as exc:
            errors.append(f"could not load result registry: {exc}")
            registry = {}
    results = registry.get("results") or []
    claims = registry.get("claims") or []
    registry_result_ids = [
        result.get("id")
        for result in results
        if isinstance(result, Mapping) and isinstance(result.get("id"), str)
    ]
    registry_claim_ids = [
        claim.get("id")
        for claim in claims
        if isinstance(claim, Mapping) and isinstance(claim.get("id"), str)
    ]
    duplicate_registry_results = sorted(
        result_id for result_id, count in Counter(registry_result_ids).items() if count > 1
    )
    duplicate_registry_claims = sorted(
        claim_id for claim_id, count in Counter(registry_claim_ids).items() if count > 1
    )
    if duplicate_registry_results:
        errors.append(f"result registry has duplicate result IDs: {duplicate_registry_results}")
    if duplicate_registry_claims:
        errors.append(f"result registry has duplicate claim IDs: {duplicate_registry_claims}")
    result_claims = {
        result.get("id"): result.get("claim")
        for result in results
        if isinstance(result, Mapping) and isinstance(result.get("id"), str)
    }
    claim_ids = set(registry_claim_ids)

    if tracked_paths is None:
        tracked_set, git_error = _git_tracked_paths(repo_root)
        if git_error:
            errors.append(git_error)
    else:
        tracked_set = set(tracked_paths)

    artifact_ids: set[str] = set()
    manifested_result_ids: Counter[str] = Counter()
    manifested_claim_ids: Counter[str] = Counter()
    for index, entry in enumerate(artifacts):
        errors.extend(
            _validate_entry(
                entry,
                index=index,
                repo_root=repo_root,
                tracked_paths=tracked_set,
                result_claims=result_claims,
                claim_ids=claim_ids,
            )
        )
        if isinstance(entry, Mapping):
            artifact_id = entry.get("artifact_id")
            if isinstance(artifact_id, str):
                if artifact_id in artifact_ids:
                    errors.append(f"duplicate artifact_id {artifact_id!r}")
                artifact_ids.add(artifact_id)
            result_id = entry.get("result_id")
            claim_id = entry.get("claim_id")
            if isinstance(result_id, str):
                manifested_result_ids[result_id] += 1
            if isinstance(claim_id, str):
                manifested_claim_ids[claim_id] += 1

    for result_id in sorted(result_claims):
        count = manifested_result_ids[result_id]
        if count == 0:
            errors.append(f"manifest is missing registry result_id {result_id!r}")
        elif count > 1:
            errors.append(f"manifest contains duplicate result_id {result_id!r} ({count} entries)")
    for claim_id in sorted(claim_ids):
        count = manifested_claim_ids[claim_id]
        if count == 0:
            errors.append(f"manifest is missing registry claim_id {claim_id!r}")
        elif count > 1:
            errors.append(f"manifest contains duplicate claim_id {claim_id!r} ({count} entries)")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--archive-mode",
        action="store_true",
        help=(
            "verify an extracted archive without requiring a .git directory; this checks "
            "present regular files but does not assert Git provenance"
        ),
    )
    args = parser.parse_args(argv)
    try:
        manifest = _load_json(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"paper artifact manifest INVALID: {exc}", file=sys.stderr)
        return 1
    repo_root = args.repo_root.resolve()
    tracked_paths = _archive_regular_paths(repo_root) if args.archive_mode else None
    errors = validate_manifest(
        manifest,
        repo_root=repo_root,
        tracked_paths=tracked_paths,
    )
    if errors:
        print(f"paper artifact manifest INVALID ({args.manifest}):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    locations = [entry["location"]["kind"] for entry in manifest["artifacts"]]
    print(
        "paper artifact manifest valid: "
        f"{len(locations)} entries "
        f"({locations.count('tracked_path')} tracked, {locations.count('immutable_url')} remote)."
    )
    if args.archive_mode:
        print(
            "archive mode: verified membership in present regular files; "
            "Git provenance was not asserted."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
