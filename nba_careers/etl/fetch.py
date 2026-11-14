"""
Stage 1: fetch season data from Basketball-Reference into data/raw_data/.

Writes CSVs in the same layout and format as the team's original hand-exported
"Share & Export -> CSV" files, so old and new seasons flow through the warehouse
build identically. Player rows carry Basketball-Reference's stable player id
(e.g. `gilgesh01`) in the last column.

Etiquette: Sports Reference allows at most 20 requests/minute and temporarily
blocks clients that exceed it. This module waits >=3.5s between requests,
caches every page under data/raw_html/ (so re-runs don't refetch), and stops
on HTTP 429 instead of retrying aggressively. A full season is 9 requests.
"""
import csv
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Comment

from ..config import ROOT

BASE_URL = "https://www.basketball-reference.com"
RAW_DIR = ROOT / "data/raw_data"
HTML_CACHE = ROOT / "data/raw_html"
MIN_INTERVAL_S = 3.5
USER_AGENT = "Mozilla/5.0 (compatible; nba-data-pipeline/1.0)"


@dataclass(frozen=True)
class Source:
    """One table on one Basketball-Reference page, for one season."""
    name: str
    url: str            # format with {end} (season end year) / {draft} (draft year)
    table_id: str
    out: str            # output path under data/raw_data, same placeholders + {start}, {yy}
    id_header: str | None = None    # column name for the appended player id
    keep_over_header: bool = False  # team advanced keeps BBRef's group header row
    one_row_per_player: bool = False  # season stat tables: combined row for traded players


# `end` = year the season ends (2025 for 2024-25). Files are named the way the
# original exports were, which differs a little between folders.
SOURCES = [
    Source("player_totals", "/leagues/NBA_{end}_totals.html", "totals_stats",
           "NBA_Player_Data/Total_Player_Data/NBA_{start}_{end}_Total_Player_Stats.csv", "Player-additional", one_row_per_player=True),
    Source("player_per_game", "/leagues/NBA_{end}_per_game.html", "per_game_stats",
           "NBA_Player_Data/Per_Game_Data/NBA_{start}_{end}_Per_Game_Player_Stats.csv", "Player-additional", one_row_per_player=True),
    Source("player_advanced", "/leagues/NBA_{end}_advanced.html", "advanced",
           "NBA_Player_Data/Advanced_Data/NBA_{start}_{end}_Advanced_Player_Stats.csv", "Player-additional", one_row_per_player=True),
    Source("team_totals", "/leagues/NBA_{end}.html", "totals-team",
           "Team_Stats/Total_Team_Data/Total_{start}-{yy}.csv"),
    Source("team_per_game", "/leagues/NBA_{end}.html", "per_game-team",
           "Team_Stats/Per_Game_Data/Per_Game_{start}-{yy}.csv"),
    Source("team_advanced", "/leagues/NBA_{end}.html", "advanced-team",
           "Team_Stats/Advanced_Data/Advanced_{start}-{yy}.csv", keep_over_header=True),
    Source("mvp_voting", "/awards/awards_{end}.html", "mvp",
           "MVP_Voting/raw_csvs/MVP_Voting_{start}_{end}.csv", "-9999"),
    Source("roy_voting", "/awards/awards_{end}.html", "roy",
           "Rookie_OTY/raw_csvs/Rookie_Year_{start}_{end}.csv", "-9999"),
    Source("all_nba_voting", "/awards/awards_{end}.html", "leading_all_nba",
           "All_Nba_Data/raw_csvs/All_Nba_Data_{start}-{yy}.csv", "-9999"),
    Source("all_rookie_voting", "/awards/awards_{end}.html", "leading_all_rookie",
           "All_Rookie_Team/raw_csvs/All_Rookie_Team_{start}_{end}.csv", "-9999"),
    # The draft that happens at the end of season `end` (e.g. June 2025)
    Source("draft", "/draft/NBA_{end}.html", "stats",
           "Draft_Pick/Nba_Draft_{end}-{next}.csv", "Player-additional"),
]


def _fmt(template: str, end: int) -> str:
    return template.format(end=end, start=end - 1, yy=f"{end % 100:02d}", next=end + 1)


class Fetcher:
    def __init__(self, refresh: bool = False, session: requests.Session | None = None):
        self.refresh = refresh
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last = 0.0
        self.requests_made = 0
        self._memo: dict[str, str] = {}  # several tables share a page; fetch it once per run

    def get(self, path: str) -> str:
        if path in self._memo:
            return self._memo[path]
        self._memo[path] = html = self._get(path)
        return html

    def _get(self, path: str) -> str:
        cache = HTML_CACHE / path.lstrip("/")
        if cache.exists() and not self.refresh:
            return cache.read_text(encoding="utf-8")
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(BASE_URL + path, timeout=30)
        self._last = time.monotonic()
        self.requests_made += 1
        if resp.status_code == 429:
            raise RuntimeError(
                f"Rate limited by Basketball-Reference on {path} "
                f"(Retry-After: {resp.headers.get('Retry-After', 'unknown')}). "
                "Wait before re-running; cached pages will not be refetched."
            )
        if resp.status_code == 404:
            raise FileNotFoundError(f"{path} does not exist yet (season not played?)")
        resp.raise_for_status()
        resp.encoding = "utf-8"  # BBRef serves UTF-8; don't let requests guess latin-1
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text, encoding="utf-8")
        return resp.text


def find_table(html: str, table_id: str):
    """Return the <table>, including tables BBRef ships inside HTML comments."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment) and f'id="{table_id}"' in s):
        table = BeautifulSoup(comment, "lxml").find("table", id=table_id)
        if table is not None:
            return table
    raise LookupError(f"table #{table_id} not found")


def _player_id(row) -> str:
    for cell in row.find_all(["th", "td"]):
        if cell.get("data-append-csv"):
            return cell["data-append-csv"]
        link = cell.find("a", href=lambda h: h and h.startswith("/players/"))
        if link:
            return link["href"].rsplit("/", 1)[-1].removesuffix(".html")
    return ""


def _cell_text(cell) -> str:
    if cell.get("data-stat") == "awards":
        # BBRef's CSV export concatenates award links without separators
        # ("MVP-1DPOY-10AS"); match it so old and new files parse the same.
        return "".join(a.get_text(strip=True) for a in cell.find_all("a")) or cell.get_text(strip=True)
    text = cell.get_text(strip=True)
    if text.replace(",", "").isdigit():  # "754,832" -> "754832"
        text = text.replace(",", "")
    return text


def table_to_rows(table, id_header: str | None, keep_over_header: bool,
                  one_row_per_player: bool = False) -> list[list[str]]:
    header_rows = table.find("thead").find_all("tr")
    header = [c.get_text(strip=True) for c in header_rows[-1].find_all(["th", "td"])]
    rows = []
    if keep_over_header and len(header_rows) > 1:
        over = []
        for c in header_rows[0].find_all(["th", "td"]):
            over += [c.get_text(strip=True)] * int(c.get("colspan", 1))
        rows.append(over)
    rows.append(header + ([id_header] if id_header else []))
    seen = set()
    for tr in table.find("tbody").find_all("tr"):
        classes = tr.get("class") or []
        if "thead" in classes or "over_header" in classes:
            continue
        cells = [_cell_text(c) for c in tr.find_all(["th", "td"])]
        if not any(cells):
            continue
        pid = _player_id(tr) if id_header else None
        if one_row_per_player:
            # Traded players get a combined "2TM"/"3TM" row first, then one row
            # per team. Keep only the combined row, as the original exports did.
            if not pid or pid in seen:
                continue
            seen.add(pid)
        rows.append(cells + ([pid] if id_header else []))
    # Team tables keep the league-average row in the footer
    tfoot = table.find("tfoot")
    if tfoot is not None and not id_header:
        for tr in tfoot.find_all("tr"):
            cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
            if cells and "League Average" in cells:
                rows.append(cells)
    return rows


def last_completed_season(today: date | None = None) -> int:
    """End year of the most recent season whose regular season + awards are done."""
    today = today or date.today()
    return today.year if today >= date(today.year, 7, 1) else today.year - 1


def fetch_season(end: int, fetcher: Fetcher | None = None, sources=SOURCES, out_dir: Path = RAW_DIR) -> list[Path]:
    fetcher = fetcher or Fetcher()
    written = []
    for src in sources:
        html = fetcher.get(_fmt(src.url, end))
        rows = table_to_rows(find_table(html, src.table_id), src.id_header, src.keep_over_header,
                             src.one_row_per_player)
        path = out_dir / _fmt(src.out, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        written.append(path)
        print(f"[fetch] {end - 1}-{end % 100:02d} {src.name:18s} {len(rows) - 1:4d} rows -> {path}")
    return written


def latest_season_on_disk(raw_dir: Path = RAW_DIR) -> int:
    files = (raw_dir / "NBA_Player_Data/Total_Player_Data").glob("NBA_*_*_Total_Player_Stats.csv")
    return max(int(p.name.split("_")[2]) for p in files)


def missing_seasons(raw_dir: Path = RAW_DIR, today: date | None = None) -> list[int]:
    return list(range(latest_season_on_disk(raw_dir) + 1, last_completed_season(today) + 1))
