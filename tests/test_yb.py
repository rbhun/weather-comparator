from __future__ import annotations

from pmc.io.yb import downsample_track, parse_leaderboard_csv


CSV = """THESE RESULTS ARE PREDICTED OR PROVISIONAL
Rank,Team,TCF,Time (UTC),Latitude,Longitude,Average COG,Average SOG (knots),VMG so far (knots),DTF (NM)

Line Honours
1,BLACK JACK 100,1.000,21/08/2025 01:55:00,043 44.244N,007 25.674E,000,0.1,10.9,0.0
2,NO REGRET,1.000,21/08/2025 20:15:00,043 44.232N,007 25.668E,055,0.1,7.5,0.0
RTD,ALAKALUF,1.000,24/08/2025 16:20:00,041 14.292N,009 11.754E,000,0.0,2.1,169.1

ORC 1
1,NO REGRET,446.000,21/08/2025 20:15:00,043 44.232N,007 25.668E,055,0.1,7.5,0.0

Committee
1,RACE COMITEE,1.000,19/08/2025 10:00:00,038 12.000N,013 20.000E,000,0.0,0.0,0.0
"""


def test_parse_leaderboard_keeps_classes_and_skips_committee() -> None:
    boats = parse_leaderboard_csv(CSV, 2025)
    assert "RACE COMITEE" not in {b.name for b in boats.values()}
    jack = boats["BLACK JACK 100"]
    assert jack.absolute_rank == 1
    assert jack.finished is True
    regret = boats["NO REGRET"]
    assert regret.absolute_rank == 2
    assert {c.class_id for c in regret.classes} == {"linehonours", "orc1"}
    assert next(c.rank for c in regret.classes if c.class_id == "orc1") == 1
    assert boats["ALAKALUF"].finished is False


def test_downsample_keeps_ends() -> None:
    lons = [float(i) for i in range(400)]
    lats = [float(i) * 0.1 for i in range(400)]
    out_lon, out_lat, _ = downsample_track(lons, lats, max_points=40)
    assert out_lon[0] == 0.0
    assert out_lon[-1] == 399.0
    assert len(out_lon) <= 41
    assert len(out_lon) == len(out_lat)
