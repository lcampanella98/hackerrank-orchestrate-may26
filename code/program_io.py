import csv
from typing import Any

from config import SUPPORT_TICKETS_DIR

# Columns expected from the input CSV (case-insensitive matching)
INPUT_COLUMNS = {"Issue", "Subject", "Company"}

OUTPUT_COLUMNS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type", "justification"
]


def read_tickets(filepath: str) -> list[dict[str, Any]]:
    """
    Read support tickets from a CSV file.

    Only the columns defined in INPUT_COLUMNS are retained; any additional
    columns present in the file are silently ignored.

    Args:
        filepath: Path to the input CSV file.

    Returns:
        A list of dicts, one per row, with keys: 'Issue', 'Subject', 'Company'.
    """
    tickets = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append({col.lower(): row[col].strip() for col in INPUT_COLUMNS if col in row})
    return tickets


def write_tickets(filepath: str, rows: list[dict[str, Any]]) -> None:
    """
    Write processed ticket rows to a CSV file.

    The output file will always contain exactly the columns defined in
    OUTPUT_COLUMNS, in that order. Missing keys in a row are written as
    empty strings; extra keys are ignored.

    Args:
        filepath: Path to the output CSV file (created or overwritten).
        rows:     List of dicts whose keys are output column names.
    """
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            extrasaction="ignore",   # silently drop keys not in OUTPUT_COLUMNS
        )
        writer.writeheader()
        for row in rows:
            # Fill in any missing columns with an empty string
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


# ---------------------------------------------------------------------------
# Quick smoke-test (only runs when the script is executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    input_path = SUPPORT_TICKETS_DIR / "support_tickets.csv"
    output_path = SUPPORT_TICKETS_DIR /"output.csv"

    if not os.path.exists(input_path):
        print(f"'{input_path}' not found – skipping smoke-test.")
    else:
        tickets = read_tickets(input_path)
        print(f"Read {len(tickets)} ticket(s) from '{input_path}'.")
        if tickets:
            print("First ticket:", tickets[0])

        # Write the tickets back out with empty extra columns as a sanity check
        write_tickets(output_path, tickets)
        print(f"Wrote {len(tickets)} row(s) to '{output_path}'.")