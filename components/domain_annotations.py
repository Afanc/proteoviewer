from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


INTERPRO_API_URL = "https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/{uniprot}/"
DEFAULT_TIMEOUT = (5.0, 20.0)
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60

REQUEST_HEADERS = {
    "User-Agent": (
        "ProteoViewer/1.9 "
        "(Biozentrum Proteomics Core Facility; InterPro domain annotations)"
    ),
    "Accept": "application/json",
}


@dataclass(frozen=True)
class DomainAnnotation:
    accession: str
    name: str
    source: str
    start: int
    end: int


@dataclass(frozen=True)
class DomainAnnotationResult:
    uniprot_id: str
    domains: list[DomainAnnotation]
    protein_length: int | None = None
    source: str = "InterPro representative domains"


def clean_uniprot_id(uniprot_id: str | None) -> str:
    token = str(uniprot_id or "").strip()
    if not token or token.lower() in {"nan", "none"}:
        return ""

    token = token.split(";", 1)[0].strip()
    token = token.split(",", 1)[0].strip()
    token = token.split("-", 1)[0].strip()

    if not re.fullmatch(r"[A-Za-z0-9_]+", token):
        return ""
    return token


def _interpro_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)

    base = Path(os.environ.get("PV_CACHE_DIR", "cache"))
    return Path(os.environ.get("PV_INTERPRO_CACHE_DIR", str(base / "interpro")))


def _cache_is_fresh(path: Path, ttl_seconds: int = CACHE_TTL_SECONDS) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        return (time.time() - path.stat().st_mtime) < ttl_seconds
    except OSError:
        return False


def _as_int(value: Any) -> int | None:
    try:
        out = int(float(value))
    except Exception:
        return None
    return out if out > 0 else None


def _extract_name(metadata: dict) -> str:
    for key in ("name", "short_name", "id", "accession"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def _walk_ranges(obj: Any) -> list[tuple[int, int]]:
    """
    InterPro JSON has changed shape over time. This deliberately scans nested
    location/fragment objects and accepts any dict carrying start/end-like keys.
    """
    ranges: list[tuple[int, int]] = []

    if isinstance(obj, dict):
        start = (
            obj.get("start")
            or obj.get("from")
            or obj.get("begin")
            or obj.get("beg")
        )
        end = (
            obj.get("end")
            or obj.get("to")
            or obj.get("stop")
        )
        s = _as_int(start)
        e = _as_int(end)
        if s is not None and e is not None and e >= s:
            ranges.append((s, e))

        for value in obj.values():
            ranges.extend(_walk_ranges(value))

    elif isinstance(obj, list):
        for value in obj:
            ranges.extend(_walk_ranges(value))

    return ranges


def _extract_protein_length(payload: dict) -> int | None:
    candidates = []

    meta = payload.get("metadata")
    if isinstance(meta, dict):
        candidates.extend([
            meta.get("length"),
            meta.get("protein_length"),
            meta.get("sequence_length"),
        ])

    for result in payload.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        candidates.extend([
            result.get("protein_length"),
            result.get("sequence_length"),
        ])
        for protein in result.get("proteins", []) or []:
            if isinstance(protein, dict):
                candidates.extend([
                    protein.get("protein_length"),
                    protein.get("length"),
                    protein.get("sequence_length"),
                ])

    parsed = [_as_int(x) for x in candidates]
    parsed = [x for x in parsed if x is not None]
    return max(parsed) if parsed else None

def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _location_is_representative(obj: Any) -> bool:
    """
    InterPro representative-domain flags are commonly attached to protein
    location objects rather than the top-level result object.
    """
    if not isinstance(obj, dict):
        return False

    if _truthy(obj.get("representative")):
        return True

    for key in ("fragments", "locations"):
        values = obj.get(key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict) and _truthy(value.get("representative")):
                    return True

    return False


def _iter_representative_location_objects(result: dict) -> list[dict]:
    """
    Return the nested location objects that should define representative-domain
    residue ranges.

    Falls back to the full result only if the result itself is explicitly marked
    representative, which preserves compatibility with simpler cached payloads.
    """
    metadata = result.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    result_is_representative = (
        _truthy(result.get("representative"))
        or _truthy(metadata.get("representative"))
    )

    out: list[dict] = []
    for protein in result.get("proteins", []) or []:
        if not isinstance(protein, dict):
            continue
        for key in ("entry_protein_locations", "locations"):
            for loc in protein.get(key, []) or []:
                if isinstance(loc, dict) and (result_is_representative or _location_is_representative(loc)):
                    out.append(loc)

    if out:
        return out
    return [result] if result_is_representative else []


def _iter_all_location_objects(result: dict) -> list[dict]:
    out: list[dict] = []
    for protein in result.get("proteins", []) or []:
        if not isinstance(protein, dict):
            continue
        for key in ("entry_protein_locations", "locations"):
            for loc in protein.get(key, []) or []:
                if isinstance(loc, dict):
                    out.append(loc)
    return out


def _domain_from_result_locations(
    result: dict,
    *,
    representative_only: bool,
) -> list[DomainAnnotation]:
    metadata = result.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    entry_type = str(metadata.get("type", "")).lower()
    if entry_type != "domain":
        return []

    accession = str(metadata.get("accession") or result.get("accession") or "")
    name = _extract_name(metadata)
    source = str(metadata.get("source_database") or metadata.get("source") or "InterPro")

    locs = (
        _iter_representative_location_objects(result)
        if representative_only
        else _iter_all_location_objects(result)
    )

    out: list[DomainAnnotation] = []
    seen_ranges: set[tuple[int, int]] = set()
    for loc in locs:
        for start, end in _walk_ranges(loc):
            if (start, end) in seen_ranges:
                continue
            seen_ranges.add((start, end))
            out.append(DomainAnnotation(
                accession=accession,
                name=name or accession,
                source=source,
                start=int(start),
                end=int(end),
            ))
    return out


def _domain_length(domain: DomainAnnotation) -> int:
    return max(0, int(domain.end) - int(domain.start) + 1)


def _domain_overlap_fraction(a: DomainAnnotation, b: DomainAnnotation) -> float:
    left = max(int(a.start), int(b.start))
    right = min(int(a.end), int(b.end))
    overlap = max(0, right - left + 1)
    if overlap <= 0:
        return 0.0
    denom = max(1, min(_domain_length(a), _domain_length(b)))
    return overlap / denom


def _domain_priority(domain: DomainAnnotation) -> tuple[int, int, int, str]:
    """
    Lower is better.

    Prefer curated InterPro domain entries, then Pfam, then profile/SMART/CDD.
    Within overlapping groups, prefer domains supported by broader InterPro
    integration and longer residue coverage.
    """
    source = str(domain.source or "").lower()
    accession = str(domain.accession or "")

    if source == "interpro" or accession.startswith("IPR"):
        source_rank = 0
    elif source == "pfam" or accession.startswith("PF"):
        source_rank = 1
    elif source in {"profile", "smart"}:
        source_rank = 2
    elif source == "cdd":
        source_rank = 3
    else:
        source_rank = 4

    return (
        source_rank,
        -_domain_length(domain),
        int(domain.start),
        accession,
    )


def _deduplicate_overlapping_domains(
    domains: list[DomainAnnotation],
    *,
    overlap_threshold: float = 0.70,
) -> list[DomainAnnotation]:
    """
    Collapse near-duplicate domains from different InterPro member databases.

    This keeps the strip readable without hiding truly separate domains.
    """
    kept: list[DomainAnnotation] = []
    for candidate in sorted(domains, key=_domain_priority):
        duplicate_idx = None
        for i, existing in enumerate(kept):
            same_region = _domain_overlap_fraction(candidate, existing) >= overlap_threshold
            same_family = (
                str(candidate.accession) == str(existing.accession)
                or str(candidate.name).lower() == str(existing.name).lower()
            )
            if same_region or same_family:
                duplicate_idx = i
                break

        if duplicate_idx is None:
            kept.append(candidate)
        elif _domain_priority(candidate) < _domain_priority(kept[duplicate_idx]):
            kept[duplicate_idx] = candidate

    return sorted(kept, key=lambda d: (int(d.start), int(d.end), str(d.accession)))


def _parse_representative_domains(payload: dict, uniprot: str) -> DomainAnnotationResult:

    representative_domains: list[DomainAnnotation] = []
    interpro_domains: list[DomainAnnotation] = []
    pfam_domains: list[DomainAnnotation] = []
    other_domains: list[DomainAnnotation] = []

    for result in payload.get("results", []) or []:
        if not isinstance(result, dict):
            continue

        metadata = result.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        representative_domains.extend(
            _domain_from_result_locations(result, representative_only=True)
        )

        source_db = str(metadata.get("source_database") or "").lower()
        accession = str(metadata.get("accession") or result.get("accession") or "")
        all_locs = _domain_from_result_locations(result, representative_only=False)

        if source_db == "interpro" or accession.startswith("IPR"):
            interpro_domains.extend(all_locs)
        elif source_db == "pfam" or accession.startswith("PF"):
            pfam_domains.extend(all_locs)
        else:
            other_domains.extend(all_locs)

    # Preferred architecture:
    # 1) curated InterPro domain entries, because they collapse Pfam/SMART/profile
    #    member evidence into biologically interpretable domain terms.
    # 2) Pfam domains if no InterPro domain entries are available.
    # 3) representative location domains if neither of the above exists.
    # 4) last-resort other domain sources.
    #
    # Then collapse near-duplicates so the strip stays readable.
    if interpro_domains:
        domains = interpro_domains
    elif pfam_domains:
        domains = pfam_domains
    elif representative_domains:
        domains = representative_domains
    else:
        domains = other_domains

    # Keep stable coordinate order and remove exact duplicates.
    dedup: dict[tuple[str, int, int], DomainAnnotation] = {}
    for dom in domains:
        key = (dom.accession, dom.start, dom.end)
        dedup[key] = dom

    out = _deduplicate_overlapping_domains(list(dedup.values()))
    return DomainAnnotationResult(
        uniprot_id=uniprot,
        domains=out,
        protein_length=_extract_protein_length(payload),
    )


def _read_cached(cache_path: Path, uniprot: str) -> DomainAnnotationResult | None:
    if not _cache_is_fresh(cache_path):
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return _parse_representative_domains(payload, uniprot)


def fetch_interpro_representative_domains(
    uniprot_id: str | None,
    *,
    cache_dir: str | Path | None = None,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> DomainAnnotationResult:
    """
    Fetch and cache InterPro representative domains for one UniProt accession.

    This function is intentionally fail-soft: on network/API errors it returns
    an empty domain result, so the stability profile simply has no domain strip.
    """
    uniprot = clean_uniprot_id(uniprot_id)
    if not uniprot:
        return DomainAnnotationResult(uniprot_id="", domains=[], protein_length=None)

    root = _interpro_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"{uniprot}.json"

    cached = _read_cached(cache_path, uniprot)
    if cached is not None:
        return cached

    url = INTERPRO_API_URL.format(uniprot=uniprot)
    all_results: list[dict] = []
    first_payload: dict[str, Any] = {}

    # InterPro can paginate. Keep a hard page limit so the UI never loops forever.
    for _page in range(25):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            if response.status_code == 404:
                return DomainAnnotationResult(uniprot_id=uniprot, domains=[], protein_length=None)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return DomainAnnotationResult(uniprot_id=uniprot, domains=[], protein_length=None)

        if not isinstance(payload, dict):
            return DomainAnnotationResult(uniprot_id=uniprot, domains=[], protein_length=None)

        if not first_payload:
            first_payload = {k: v for k, v in payload.items() if k != "results"}

        results = payload.get("results", [])
        if isinstance(results, list):
            all_results.extend([x for x in results if isinstance(x, dict)])

        next_url = payload.get("next")
        if not next_url:
            break
        url = str(next_url)

    combined = dict(first_payload)
    combined["results"] = all_results
    combined["retrieved_at"] = time.time()

    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        tmp_path.replace(cache_path)
    except OSError:
        pass

    return _parse_representative_domains(combined, uniprot)
