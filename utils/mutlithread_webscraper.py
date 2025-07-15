import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from utils.utils import helper

# Config
DELAY_EVERY = 1000
MAX_WORKERS = 4
DELAY_PER_THREAD = 1

# Shared State
processed_count = 0
count_lock = threading.Lock()
rate_limit_event = threading.Event()
countdown_lock = threading.Lock()


def countdown_timer(seconds):
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        timer_display = f"\r Sleeping ... {mins:02d}:{secs:02d} remaining"
        sys.stdout.write(timer_display)
        sys.stdout.flush()
        time.sleep(1)
    logging.info("\r Done Sleeping! Trying Again")


def scrape_players(browser, player, index):
    global processed_count

    stagger = (index % MAX_WORKERS) * DELAY_PER_THREAD
    logging.info(f"[{player}] Initial stagger: {stagger:.1f}s")
    time.sleep(stagger)

    url = f"https://www.sports-reference.com/{player}"
    while True:

        logging.info(f"Processing {player}...")

        try:
            # Wait between 5 to 30 seconds randomly
            time.sleep(random.uniform(5, 30))
            page = browser.new_page()
            response = page.goto(url)
            print(response)
            html = page.content()
            page.close()
            # response = session.get(url, timeout=10)
            if response.status == 429:

                if count_lock.acquire(blocking=False):
                    logging.warning(f"\n[{player}] Got 429! Triggering global rate limit.")
                    try:
                        rate_limit_event.clear()
                        logging.info("\n[GLOBAL] Sleeping for 30 minutes due to rate limit.")
                        countdown_timer(30 * 60)  # Block for 15 mins
                        logging.info("\n[GLOBAL] Finished sleeping. Unblocking threads.")
                        rate_limit_event.set()
                    finally:
                        count_lock.release()
                else:
                    time.sleep(random.uniform(0.5, 2.5))  # prevent reburst
                    if not rate_limit_event.is_set():
                        rate_limit_event.wait()
                        time.sleep(random.uniform(0.5, 2.5))  # prevent reburst
                continue
            elif response.status != 200:
                logging.error(f"\nFailed with status {response.status}, skipping.")
            else:
                soup = BeautifulSoup(html, "html.parser")
                # Locate table with stats (per-season stats table has id="players_per_game")
                meta_div = soup.find("div", id="meta")

                # Check if any link in the meta div contains 'draft' in the href
                drafted = any("draft" in a.get("href", "") for a in meta_div.find_all("a"))
                weight = None
                height = None
                for p in meta_div.find_all("p"):
                    span = [span.text for span in p.find_all("span")]
                    if span:
                        height = span[0]
                        weight = span[1] if len(span) > 1 else None
                bottom_nav = soup.find("div", id="bottom_nav_container")
                nba_link_text = None
                if bottom_nav:
                    for link in bottom_nav.find_all("a"):
                        if link.get_text(strip=True) == "Basketball-Reference.com":
                            nba_link_text = link.get_text(strip=True)
                            break
                if nba_link_text is None:
                    raise ValueError("Did not play in NBA")

                for table, label in [
                    ("players_per_game", "Per_Game"),
                    ("players_totals", "Total_Stats"),
                    ("players_advanced", "Advanced_Stats")
                ]:
                    table = soup.find("table", id=table)
                    helper(table, drafted, weight, height, data_type=label, player=player)

                print(f"\nSuccessfully parsed: {player}")
        except ValueError as e:
            logging.error(f"\nError: {e}. Skipping: {player}")
            break
        except Exception as e:
            logging.error(f"\nError: {e}. Try again record: {player}")
            continue
        break

    with count_lock:
        processed_count += 1
        if processed_count % DELAY_EVERY == 0:
            extra_sleep = random.uniform(60, 180)
            logging.info(f"[{processed_count}] Taking a long break: {extra_sleep:.1f}s")
            time.sleep(extra_sleep)


def main():

    # Set up logging once
    datetime_str = datetime.now().strftime("%Y%m%d-%H")

    logging.basicConfig(
        filename=str(Path(f"../logs/web_scraper_{datetime_str}.log")),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    with Path("../data/raw_data/College_Stats/player_links.txt").open(mode="r") as f:
        player_weblinks = {line.strip() for line in f}
    if player_weblinks:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)  # or False to see browser
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(scrape_players, browser, player, i) for i, player in
                           enumerate(iter(player_weblinks))]
                for _ in tqdm(as_completed(futures), total=len(futures), file=sys.stdout):
                    time.sleep(0.5)
                    sys.stdout.flush()
            browser.close()
    else:
        print(
            "You need to run the higher cell as there are no more weblinks available or you are simply finished "
            "scraping :)")


if __name__ == "__main__":
    main()
