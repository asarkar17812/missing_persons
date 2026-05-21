"""
NamUs Missing Persons API scraper.

Pulls every public Missing Persons case from the NamUs API (one state at a time
to avoid the per-call result cap) and writes the full set to a single JSON file
under output/. Originally published by Night Owl Reconnaissance, a 501(c)(3)
nonprofit focused on detection and investigation techniques for finding missing
persons (https://github.com/NightOwlRecon); only minor formatting and inline
documentation have been added here.

How it works:
    1. Hit /api/CaseSets/NamUs/States to list all U.S. states/territories.
    2. For each state, POST to the search endpoint with `take=10000` to get
       every case_id whose `stateOfLastContact` matches.
    3. For each case_id, GET /Cases/{id} to pull the full record.
    4. Save the concatenated list to output/namus-YYYYMMDD.json with one
       case per line (still a valid JSON array) for easy diffing.

The case-fetch step retries with exponential backoff on transient failures and
gives up at the 13th consecutive failure (~68 minutes of waiting). 404s are
treated as "case was removed since the search ran" and skipped without retry.

Run as:
    python scripts/us/data/scraper/namus.py
"""

import datetime
import json
import time
import requests


def load_stored_cases():
    """Load a previously-saved snapshot of cases.

    Currently unused in the live pipeline; kept around because it's useful when
    we want to query only the records we don't already have a local copy of
    (e.g., resuming a partial run).
    """
    with open("output/namus-20240811.json", "r") as f:
        cases = json.load(f)
    return cases


def save_cases(cases):
    """Write the full list of cases to output/namus-YYYYMMDD.json.

    Format: one case per line, wrapped in a JSON array, so the file diffs
    cleanly in git but still parses with `json.load`. (Avoids pulling in a
    jsonlines dependency.) Cases are sorted by ID before writing to keep the
    diff between snapshots stable.
    """
    date = datetime.datetime.now().strftime("%Y%m%d")

    # Sort by case ID so consecutive snapshots produce minimal diffs.
    cases.sort(key=lambda x: x["id"])

    with open(f"output/namus-{date}.json", "w") as f:
        # Open the JSON array manually so we can control line breaks.
        f.write("[\n")

        for case in cases:
            # Leading tab keeps the file readable; trailing comma is omitted
            # on the last entry to maintain valid JSON.
            f.write("\t")
            json.dump(case, f)
            if case == cases[-1]:
                f.write("\n")
            else:
                f.write(",\n")
        f.write("]\n")


def get_states():
    """Return the list of state names NamUs accepts as a search filter.

    Could be hard-coded since the list never changes meaningfully, but a live
    call is cheap and means we automatically pick up territory additions.
    Any exception here is fatal — there's no point retrying anything else if
    the states list won't load.
    """
    states = [state["name"] for state in requests.get("https://www.namus.gov/api/CaseSets/NamUs/States").json()]
    return states


def get_cases_by_state(state):
    """Return all NamUs2 case IDs whose state-of-last-contact matches `state`.

    Uses the public search endpoint with `take=10000`; we choose 10k because
    no single U.S. state has anywhere near that many open NamUs records, so
    pagination is unnecessary in practice.
    """
    res = requests.post(
        "https://www.namus.gov/api/CaseSets/NamUs/MissingPersons/Search",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "take": 10000,
                "projections": ["namus2Number"],
                "predicates": [
                    {
                        "field": "stateOfLastContact",
                        "operator": "IsIn",
                        "values": [state],
                    }
                ],
            }
        ),
    ).json()

    case_ids = [case["namus2Number"] for case in res["results"]]
    return case_ids


def get_case_by_id(case_id):
    """Fetch the full case record for a single NamUs2 case ID."""
    case = requests.get(f"https://www.namus.gov/api/CaseSets/NamUs/MissingPersons/Cases/{case_id}")
    return case


def main():
    """Drive the full scrape: enumerate states, list case IDs, fetch each case.

    Uses an exponential-backoff retry loop per case (2^n seconds, capped at
    13 consecutive failures). A 404 on a case is treated as "the case was
    removed between the time the search returned its ID and the time we
    tried to fetch it" and skipped rather than retried.
    """
    failures = 0
    cases = []
    case_ids = []

    states = get_states()

    # Phase 1: collect every case ID we'll need to fetch.
    for state in states:
        ids = get_cases_by_state(state)
        print(f"Found {len(ids)} cases in {state}")
        case_ids.extend(ids)

    print(f"Found {len(case_ids)} total ")

    # Phase 2: fetch the full record for each case ID, with backoff.
    for i in range(len(case_ids)):
        while True:
            case_id = case_ids[i]
            print(f"Getting case ID {case_id} ({i+1}/{len(case_ids)} - {100*(i+1)/len(case_ids):.2f}%)")
            try:
                case = get_case_by_id(case_id)
                cases.append(case.json())
                failures = 0
                break  # exit the backoff loop for this case
            except Exception as e:
                print(f"Failed to get case ID {case_id}: {e}")
                print(case)
                print(case.text)
                print(case.status_code)

                # A 404 typically means the case was removed between the
                # search returning its ID and us trying to fetch the record.
                # No point retrying; move on to the next case.
                if case.status_code == 404:
                    break

                # Otherwise exponential backoff: 2, 4, 8, ..., capped at 13
                # failures (about 68 minutes of waiting in total).
                failures += 1
                if failures == 13:
                    print("Too many failures, exiting")
                    return
                delay_s = pow(2, failures)
                print(f"Failures: {failures}, sleeping for {delay_s} seconds")
                time.sleep(delay_s)

    save_cases(cases)


if __name__ == '__main__':
    main()
