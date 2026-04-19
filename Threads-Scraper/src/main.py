import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml
from dotenv import load_dotenv
# Local imports
from scraper.threads_scraper import ThreadsScraper
from scraper.parser import ThreadsParser
from scraper.exporter import Exporter
from scraper.utils.logger import get_logger
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CONFIG_DIR = ROOT / "config"
logger = get_logger(__name__)
def load_settings(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    return settings
def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
def parse_args(default_usernames: List[str], default_min_total: int, default_related_limit: int, default_max_users: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threads Scraper — scrape Threads posts for given usernames."
    )
    parser.add_argument(
        "-u",
        "--usernames",
        nargs="+",
        help="Threads usernames to scrape (without @). Defaults to settings.yaml",
        default=default_usernames,
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (use local sample dump).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of threads per user to collect (if supported by endpoint).",
    )
    parser.add_argument(
        "--min-total",
        type=int,
        default=default_min_total,
        help="Keep collecting from related profiles until this minimum total item count is reached.",
    )
    parser.add_argument(
        "--related-limit",
        type=int,
        default=default_related_limit,
        help="How many related usernames to enqueue from each visited profile.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=default_max_users,
        help="Hard cap on number of visited usernames during expansion.",
    )
    return parser.parse_args()
def main():
    load_dotenv()  # load .env if present
    ensure_dirs()
    settings_path = CONFIG_DIR / "settings.yaml"
    settings = load_settings(settings_path)
    args = parse_args(
        settings.get("usernames", []),
        default_min_total=int(settings.get("min_total_items", 0) or 0),
        default_related_limit=int(settings.get("related_limit", 20) or 20),
        default_max_users=int(settings.get("max_users", 800) or 800),
    )
    if not args.usernames:
        logger.error("No usernames provided via CLI or settings.yaml")
        sys.exit(1)
    # Merge CLI overrides into settings
    settings["use_offline"] = args.offline or settings.get("use_offline", False)
    settings["limit"] = args.limit
    settings["min_total_items"] = max(0, int(args.min_total))
    settings["related_limit"] = max(0, int(args.related_limit))
    settings["max_users"] = max(1, int(args.max_users))
    scraper = ThreadsScraper(
        settings=settings,
        config_dir=CONFIG_DIR,
        data_dir=DATA_DIR,
    )
    parser = ThreadsParser()
    exporter = Exporter(output_dir=OUTPUT_DIR, data_dir=DATA_DIR)
    all_results: List[Dict[str, Any]] = []
    queue: List[str] = []
    for username in args.usernames:
        normalized = str(username).strip().lower()
        if normalized and normalized not in queue:
            queue.append(normalized)

    seen_users = set()
    seen_ids = set()
    while queue and len(seen_users) < settings["max_users"]:
        if settings["min_total_items"] > 0 and len(all_results) >= settings["min_total_items"]:
            break

        username = queue.pop(0)
        if username in seen_users:
            continue
        seen_users.add(username)

        try:
            logger.info(f"Collecting threads for @{username} (offline={settings['use_offline']})")
            raw_items = scraper.fetch_user_threads(username=username, limit=settings["limit"])
            parsed_items = [parser.parse_item(item, default_username=username) for item in raw_items]
            parsed_items = [p for p in parsed_items if p]  # drop None

            new_items_count = 0
            for parsed in parsed_items:
                item_id = str(parsed.get("id") or "")
                if not item_id or item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                all_results.append(parsed)
                new_items_count += 1

            logger.info(
                "Collected %s new items from @%s (running total=%s)",
                new_items_count,
                username,
                len(all_results),
            )

            if (not settings["use_offline"]) and settings["related_limit"] > 0:
                related_usernames = scraper.fetch_related_usernames(
                    username=username,
                    limit=settings["related_limit"],
                )
                for related_username in related_usernames:
                    if related_username in seen_users or related_username in queue:
                        continue
                    queue.append(related_username)
        except Exception as e:
            logger.exception(f"Failed to collect for @{username}: {e}")

    if settings["min_total_items"] > 0 and len(all_results) < settings["min_total_items"]:
        logger.warning(
            "Minimum target %s not reached. Collected %s items after visiting %s users.",
            settings["min_total_items"],
            len(all_results),
            len(seen_users),
        )

    if settings["min_total_items"] > 0 and len(all_results) > settings["min_total_items"]:
        all_results = all_results[: settings["min_total_items"]]

    if not all_results:
        logger.warning("No results collected. Exiting.")
        sys.exit(0)
    # Export to /output
    json_path = exporter.to_json(all_results, filename="threads_results.json")
    csv_path = exporter.to_csv(all_results, filename="threads_results.csv")
    # Also create a processed/clean_threads.csv for convenience
    processed_path = exporter.to_csv(all_results, filename="clean_threads.csv", subdir="data/processed")
    logger.info(f"Wrote JSON -> {json_path}")
    logger.info(f"Wrote CSV  -> {csv_path}")
    logger.info(f"Wrote processed CSV -> {processed_path}")
    # Print a short completion message with summary stats
    users = sorted({r["username"] for r in all_results})
    logger.info(
        json.dumps(
            {"users": users, "total_items": len(all_results), "output_json": str(json_path), "output_csv": str(csv_path)},
            indent=2,
        )
    )
if __name__ == "__main__":
    main()
