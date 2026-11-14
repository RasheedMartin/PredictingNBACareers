from datetime import date

import duckdb
import pytest

from nba_careers.etl import fetch
from nba_careers.etl.warehouse import WAREHOUSE_PATH, clean_name

PAGE = """
<html><body>
<table id="totals_stats">
  <thead><tr><th>Rk</th><th>Player</th><th>Team</th><th>G</th><th>Awards</th></tr></thead>
  <tbody>
    <tr><th>1</th><td data-stat="name_display" data-append-csv="jokicni01">Nikola Jokić</td>
        <td>DEN</td><td>70</td>
        <td data-stat="awards"><a>MVP-2</a>,<a>AS</a>,<a>NBA1</a></td></tr>
    <tr><th>2</th><td data-stat="name_display" data-append-csv="tradepl01">Traded Player</td>
        <td>2TM</td><td>60</td><td data-stat="awards"></td></tr>
    <tr class="thead"><th>Rk</th><td>Player</td><td>Team</td><td>G</td><td>Awards</td></tr>
    <tr><th>2</th><td data-stat="name_display" data-append-csv="tradepl01">Traded Player</td>
        <td>BOS</td><td>30</td><td data-stat="awards"></td></tr>
    <tr><th></th><td>League Average</td><td></td><td>1,234</td><td></td></tr>
  </tbody>
</table>
<!--
<table id="advanced-team"><thead>
  <tr><th colspan="2"></th><th colspan="1">Offense Four Factors</th></tr>
  <tr><th>Rk</th><th>Team</th><th>eFG%</th></tr></thead>
  <tbody><tr><th>1</th><td>Boston Celtics*</td><td>.560</td></tr></tbody></table>
-->
</body></html>
"""


def test_finds_tables_hidden_in_html_comments():
    assert fetch.find_table(PAGE, "advanced-team") is not None


def test_player_table_matches_bbref_csv_export_format():
    rows = fetch.table_to_rows(fetch.find_table(PAGE, "totals_stats"), "Player-additional",
                               keep_over_header=False, one_row_per_player=True)
    header, *body = rows
    assert header[-1] == "Player-additional"
    assert [r[-1] for r in body] == ["jokicni01", "tradepl01"]   # league average + per-team rows dropped
    assert body[1][2] == "2TM"                                    # combined row kept for traded player
    assert body[0][4] == "MVP-2ASNBA1"                            # awards concatenated like the export


def test_team_table_keeps_group_header_and_strips_thousands_separators():
    rows = fetch.table_to_rows(fetch.find_table(PAGE, "advanced-team"), None, keep_over_header=True)
    assert rows[0] == ["", "", "Offense Four Factors"]
    assert rows[1] == ["Rk", "Team", "eFG%"]


def test_only_completed_seasons_are_fetched():
    assert fetch.last_completed_season(date(2026, 11, 14)) == 2026   # 2025-26 finished in June
    assert fetch.last_completed_season(date(2027, 3, 1)) == 2026     # 2026-27 still in progress


def test_name_cleaning():
    assert clean_name("Nikola Jokić") == "Nikola Jokic"
    assert clean_name("J.J. Redick*") == "JJ Redick"


@pytest.mark.skipif(not WAREHOUSE_PATH.exists(), reason="run `python -m nba_careers.etl warehouse` first")
def test_warehouse_invariants():
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    q = lambda sql: con.sql(sql).fetchall()  # noqa: E731
    assert q("SELECT count(*) FROM teams WHERE name = 'Unknown'") == [(0,)]
    assert q("SELECT count(*) FROM players WHERE name LIKE 'None %'") == [(0,)]
    # same-name players stay distinct (e.g. Patrick Ewing Sr./Jr.)
    assert q("SELECT count(*) FROM players WHERE name = 'Patrick Ewing' AND bbref_id IS NOT NULL") == [(2,)]
    # one row per player-season-stat: traded players aren't double counted
    assert q("""SELECT count(*) FROM (SELECT player_id, season_id, stat_type_id, stat_id FROM player_stat_values
                GROUP BY ALL HAVING count(*) > 1)""") == [(0,)]
    con.close()
