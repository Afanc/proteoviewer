import requests
from functools import lru_cache

# Reasonable timeouts: (connect, read) seconds
_DEFAULT_TIMEOUT = (3.0, 5.0)

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

#import requests
#
#def get_string_link(uniprot_id: str) -> str:
#    """
#    Returns a STRING redirect link for a given UniProt ID.
#    Example: get_string_link("P60624") → "https://string-db.org/cgi/link?to=4CA51CFE87DA16C2"
#    """
#    # Step 1: Resolve UniProt ID to STRING internal ID
#    r = requests.get("https://string-db.org/api/tsv/get_string_ids", params={
#        "identifiers": uniprot_id,
#        "format": "tsv",
#    })
#
#    if not r.ok:
#        print("[STRING] Failed get_string_ids")
#        return ""
#
#    lines = r.text.strip().split("\n")
#    if len(lines) < 2:
#        print(f"[STRING] No STRING match found for {uniprot_id}")
#        return ""
#
#    string_id = lines[1].split("\t")[1]
#
#    # Step 2: Get the redirect link
#    link_r = requests.get("https://string-db.org/api/tsv/get_link", params={
#        "identifiers": string_id,
#        "format": "tsv"
#    })
#
#    if link_r.ok:
#        link_lines = link_r.text.strip().split("\n")
#        if len(link_lines) >= 2:
#            return link_lines[1].strip()
#
#    print(f"[STRING] Unexpected format in get_link response:\n{link_r.text}")
#    return ""
#
