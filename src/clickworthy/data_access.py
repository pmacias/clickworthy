# DuckDB layer: load the Upworthy archive CSV, build clickworthy.duckdb,
# expose the standard filtered view. Stage 0.
#
# Standard filters (CLAUDE.md "Standard filters" section):
#   1. Drop packages/tests with created_at in [2013-06-25, 2014-01-10] --
#      stands in for the `problem` column, which does not exist in this
#      file version (added in a June 2024 archive update we don't have).
#   2. Drop degenerate tests: fewer than 2 packages, or any package with
#      0 impressions.
#
# No confirmatory-sample view: the confirmatory dataset is access-gated,
# not freely downloadable (see CLAUDE.md's Data section).

from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_CSV_PATH = Path(
    "~/ml_datasets/clickworthy/upworthy-archive-exploratory-packages-03.12.2020.csv"
).expanduser()
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "clickworthy.duckdb"

PROBLEM_WINDOW_START = "2013-06-25"
PROBLEM_WINDOW_END = "2014-01-10"

RAW_TABLE = "packages_raw"
CLEAN_VIEW = "v_exploratory_clean"


def build_database(
    csv_path: Path = DEFAULT_CSV_PATH, db_path: Path = DEFAULT_DB_PATH
) -> duckdb.DuckDBPyConnection:
    """Load the archive CSV into a fresh DuckDB file and create the
    standard filtered view. Returns an open connection to db_path."""
    con = duckdb.connect(str(db_path))
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {RAW_TABLE} AS
        SELECT * EXCLUDE (column00)
        FROM read_csv_auto(?, header=True)
        """,
        [str(csv_path)],
    )
    _create_clean_view(con)
    return con


def connect(db_path: Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open a connection to an already-built clickworthy.duckdb."""
    return duckdb.connect(str(db_path), read_only=False)


def _create_clean_view(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {CLEAN_VIEW} AS
        WITH date_filtered AS (
            SELECT *
            FROM {RAW_TABLE}
            WHERE NOT (
                CAST(created_at AS DATE) BETWEEN
                    DATE '{PROBLEM_WINDOW_START}' AND DATE '{PROBLEM_WINDOW_END}'
            )
        ),
        test_stats AS (
            SELECT
                clickability_test_id,
                COUNT(*) AS n_packages,
                MIN(impressions) AS min_impressions
            FROM date_filtered
            GROUP BY clickability_test_id
        )
        SELECT d.*
        FROM date_filtered d
        JOIN test_stats t USING (clickability_test_id)
        WHERE t.n_packages >= 2
          AND t.min_impressions > 0
        """
    )


def filter_accounting(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Row/test counts remaining after each standard filter, applied
    cumulatively, for the ingestion notebook's accounting table."""
    steps = {}

    steps["0_raw"] = con.execute(
        f"SELECT COUNT(*) n_packages, COUNT(DISTINCT clickability_test_id) n_tests, "
        f"SUM(impressions) total_impressions FROM {RAW_TABLE}"
    ).df()

    steps["1_after_date_exclusion"] = con.execute(
        f"""
        SELECT COUNT(*) n_packages, COUNT(DISTINCT clickability_test_id) n_tests,
               SUM(impressions) total_impressions
        FROM {RAW_TABLE}
        WHERE NOT (
            CAST(created_at AS DATE) BETWEEN
                DATE '{PROBLEM_WINDOW_START}' AND DATE '{PROBLEM_WINDOW_END}'
        )
        """
    ).df()

    steps["2_after_degenerate_test_exclusion"] = con.execute(
        f"SELECT COUNT(*) n_packages, COUNT(DISTINCT clickability_test_id) n_tests, "
        f"SUM(impressions) total_impressions FROM {CLEAN_VIEW}"
    ).df()

    out = pd.concat(steps.values(), keys=steps.keys()).reset_index(level=1, drop=True)
    out.index.name = "step"
    return out.reset_index()


def load_test(
    con: duckdb.DuckDBPyConnection, test_id: str, clean: bool = True
) -> pd.DataFrame:
    """Pull one test's packages as a tidy frame."""
    table = CLEAN_VIEW if clean else RAW_TABLE
    return con.execute(
        f"SELECT * FROM {table} WHERE clickability_test_id = ? ORDER BY created_at",
        [test_id],
    ).df()
