"""
Data pipeline CLI.

    python -m nba_careers.etl fetch [--season 2026 ...] [--refresh]   # stage 1: Basketball-Reference -> data/raw_data
    python -m nba_careers.etl warehouse                               # stage 2: raw CSVs -> data/warehouse/nba_data.duckdb
    python -m nba_careers.etl table                                   # stage 3: warehouse -> modeling table
    python -m nba_careers.etl refresh                                 # fetch any new seasons, then stages 2-3,
                                                                      # retrain the model, and refresh the app DB
"""
import argparse
import sys

from . import fetch, modeling_table, warehouse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nba_careers.etl", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download season data from Basketball-Reference")
    f.add_argument("--season", type=int, nargs="*",
                   help="season END years, e.g. 2026 for 2025-26 (default: all completed seasons not on disk)")
    f.add_argument("--refresh", action="store_true", help="ignore the HTML cache and re-download")
    sub.add_parser("warehouse", help="rebuild the DuckDB warehouse from raw CSVs")
    sub.add_parser("table", help="rebuild the modeling table from the warehouse")
    r = sub.add_parser("refresh", help="fetch new seasons + rebuild everything + retrain")
    r.add_argument("--no-train", action="store_true")
    args = parser.parse_args(argv)

    if args.cmd in ("fetch", "refresh"):
        seasons = getattr(args, "season", None) or fetch.missing_seasons()
        if not seasons:
            print(f"[fetch] up to date (latest season on disk: {fetch.latest_season_on_disk()})")
        fetcher = fetch.Fetcher(refresh=getattr(args, "refresh", False))
        for end in seasons:
            fetch.fetch_season(end, fetcher)
        if seasons:
            print(f"[fetch] {fetcher.requests_made} network requests")
        if args.cmd == "fetch":
            return 0
    if args.cmd in ("warehouse", "refresh"):
        warehouse.build()
        warehouse.export_app_db()
    if args.cmd in ("table", "refresh"):
        modeling_table.build()
    if args.cmd == "refresh" and not args.no_train:
        from ..train import run
        run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
