import requests
from functools import lru_cache

# Reasonable timeouts: (connect, read) seconds
_DEFAULT_TIMEOUT = (3.0, 5.0)

STRING_API_URL = "https://version-12-0.string-db.org/api"
STRING_CALLER_IDENTITY = "proteoviewer_biozentrum_pcf_unibas"

@lru_cache(maxsize=4096)
def get_string_link(uniprot_id: str) -> str:
    """
    Returns a STRING redirect link for a given UniProt ID.
    Example: get_string_link("P60624") → "https://string-db.org/cgi/link?to=..."
    Cached and time-bounded so it won't block the app.
    """
    try:
        # Step 1: UniProt → STRING ID
        r = requests.get(
            "https://string-db.org/api/tsv/get_string_ids",
            params={"identifiers": uniprot_id, "format": "tsv"},
            timeout=_DEFAULT_TIMEOUT,
        )
        if not r.ok:
            print("[STRING] Failed get_string_ids")
            return ""

        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            print(f"[STRING] No STRING match found for {uniprot_id}")
            return ""

        string_id = lines[1].split("\t")[1]

        # Step 2: resolve redirect link
        link_r = requests.get(
            "https://string-db.org/api/tsv/get_link",
            params={"identifiers": string_id, "format": "tsv"},
            timeout=_DEFAULT_TIMEOUT,
        )
        if not link_r.ok:
            print("[STRING] get_link failed")
            return ""

        link_lines = link_r.text.strip().split("\n")
        if len(link_lines) >= 2:
            return link_lines[1].strip()

        print(f"[STRING] Unexpected format in get_link response:\n{link_r.text}")
        return ""

    except requests.exceptions.RequestException as e:
        # Any network error / timeout: fail fast and keep UI responsive
        print(f"[STRING] Request failed: {e}")
        return ""

@lru_cache(maxsize=256)
def get_string_functional_enrichment(
    identifiers: tuple[str, ...],
    species: int,
) -> list[dict]:
    """
    Retrieve STRING functional enrichment for a protein set.

    Parameters
    ----------
    identifiers
        Hashable, de-duplicated protein identifiers. UniProt accessions work
        for standard ProteoFlux/ProteoViewer exports.
    species
        NCBI/STRING taxon identifier.

    Returns
    -------
    list[dict]
        JSON rows returned by STRING's enrichment endpoint.
    """
    clean_ids = tuple(
        x for x in (str(i).strip() for i in identifiers)
        if x and x.lower() not in {"nan", "none"}
    )
    if not clean_ids:
        raise ValueError("STRING enrichment requires at least one protein identifier.")
    if species is None:
        raise ValueError("STRING enrichment requires a species.")

    request_url = "/".join([STRING_API_URL, "json", "enrichment"])
    params = {
        "identifiers": "\r".join(clean_ids),
        "species": int(species),
        "caller_identity": STRING_CALLER_IDENTITY,
    }

    try:
        response = requests.post(request_url, data=params, timeout=_DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"STRING enrichment request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("STRING enrichment returned invalid JSON.") from exc

    if not isinstance(data, list):
        raise RuntimeError(f"STRING enrichment returned unexpected payload type: {type(data).__name__}.")
    return data
