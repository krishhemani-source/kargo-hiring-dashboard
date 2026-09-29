"""Calibration test: score the 8 past hires in seed/ and check the ranking.

Expected: Exceeds (Rohan, Sunita, Aditya, Meghna, Lavanya) rank above Meets
(Vikram, Rahul) and Below (Preetham); for the two PM hires, Lavanya well above Vikram.

Which score ranks a hire: the two PMs (Lavanya, Vikram) are ranked on the PM rubric.
The other six were hired into other functions (engineering, ops consulting, sales, CS,
dev, marketing), so they're ranked on their best-fit score (max of PM and SPM).

    python calibrate.py            # uses cached scores when the CV + model are unchanged
    python calibrate.py --fresh    # re-score everything
"""
import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import config
from app.parsing import extract_details, extract_text, redact
from app.pipeline import score_both

SEED = config.ROOT / "seed"
CACHE = config.ROOT / "data" / "calibration_cache.json"
OUT = config.ROOT / "data" / "calibration_results.json"
LOCK = threading.Lock()

RATINGS = {
    "rohan": "Exceeds", "sunita": "Exceeds", "aditya": "Exceeds", "meghna": "Exceeds", "lavanya": "Exceeds",
    "vikram": "Meets", "rahul": "Meets", "preetham": "Below",
}
PM_HIRES = {"lavanya", "vikram"}
WELL_ABOVE = 20.0  # percentage points between Lavanya and Vikram on PM


def key_for(path: Path) -> str:
    return path.stem.split("_")[2].lower()  # cv_07_lavanya_iyer -> lavanya


def score_file(path: Path, cache: dict, fresh: bool) -> dict:
    data = path.read_bytes()
    ck = hashlib.sha256(data).hexdigest()
    ex = extract_details(path.name, extract_text(path.name, data))
    if not fresh and ck in cache:
        scores = cache[ck]
    else:
        pii = {"name": ex.name, "email": ex.email, "phone": ex.phone, "links": ex.links}
        scores = score_both(redact(ex, path.name), pii)
        cache[ck] = scores
        with LOCK:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache, indent=1))
    k = key_for(path)
    primary_role = "PM" if k in PM_HIRES else max(("PM", "SPM"), key=lambda r: scores[r]["pct"])
    return {"key": k, "name": ex.name, "rating": RATINGS.get(k, "?"), "scores": scores,
            "primary_role": primary_role, "primary": scores[primary_role]["pct"]}


def crit_table(r: dict) -> str:
    c = r["scores"][r["primary_role"]]["criteria"]
    return "  ".join(f"{x['name'][:22]}={x['score']}" for x in c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignore cached scores")
    args = ap.parse_args()

    files = sorted(SEED.glob("*.docx")) + sorted(SEED.glob("*.pdf"))
    if not files:
        sys.exit("No CVs in seed/")
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"Scoring {len(files)} past hires with {' -> '.join(config.GEMINI_MODELS)} (both rubrics each)...\n")
    def safe(p):
        try:
            return score_file(p, cache, args.fresh)
        except Exception as e:
            print(f"  ! {p.name}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=config.LLM_WORKERS) as pool:
        results = [r for r in pool.map(safe, files) if r]
    if len(results) < len(files):
        sys.exit(f"\nOnly {len(results)}/{len(files)} scored; rerun to finish (scored CVs are cached).")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1))
    OUT.write_text(json.dumps(results, indent=1))

    results.sort(key=lambda r: r["primary"], reverse=True)
    print(f"{'#':>2}  {'Name':<22} {'Actual':<8} {'PM %':>6} {'SPM %':>6}  {'Ranked on':<9} {'Score':>6}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i:>2}  {r['name']:<22} {r['rating']:<8} {r['scores']['PM']['pct']:>6.1f} "
              f"{r['scores']['SPM']['pct']:>6.1f}  {r['primary_role']:<9} {r['primary']:>6.1f}")
    print()

    ok = True
    exceeds = [r for r in results if r["rating"] == "Exceeds"]
    others = [r for r in results if r["rating"] != "Exceeds"]
    lowest_ex = min(exceeds, key=lambda r: r["primary"])
    highest_other = max(others, key=lambda r: r["primary"])
    if lowest_ex["primary"] > highest_other["primary"]:
        print(f"PASS  Every Exceeds hire ranks above every Meets/Below hire "
              f"(lowest Exceeds {lowest_ex['primary']}% > highest other {highest_other['primary']}%).")
    else:
        ok = False
        print("FAIL  Ranking is off. Out-of-order pairs and the criteria behind them:")
        for e in exceeds:
            for o in others:
                if e["primary"] <= o["primary"]:
                    print(f"\n  {e['name']} (Exceeds, {e['primary']}% on {e['primary_role']}) <= "
                          f"{o['name']} ({o['rating']}, {o['primary']}% on {o['primary_role']})")
                    print(f"    {e['name']}: {crit_table(e)}")
                    print(f"    {o['name']}: {crit_table(o)}")
                    if e["primary_role"] == o["primary_role"]:
                        ec = {x["id"]: x for x in e["scores"][e["primary_role"]]["criteria"]}
                        for x in o["scores"][o["primary_role"]]["criteria"]:
                            if x["score"] >= ec[x["id"]]["score"]:
                                print(f"    - {x['name']}: {o['name']} {x['score']} vs {e['name']} {ec[x['id']]['score']}"
                                      f" | {o['name']}: {x['reason']} | {e['name']}: {ec[x['id']]['reason']}")

    lav = next((r for r in results if r["key"] == "lavanya"), None)
    vik = next((r for r in results if r["key"] == "vikram"), None)
    if lav and vik:
        lp, vp = lav["scores"]["PM"]["pct"], vik["scores"]["PM"]["pct"]
        gap = round(lp - vp, 1)
        if gap >= WELL_ABOVE:
            print(f"PASS  PM hires: Lavanya {lp}% vs Vikram {vp}% (gap {gap} pts, needs >= {WELL_ABOVE}).")
        else:
            ok = False
            print(f"FAIL  PM hires: Lavanya {lp}% vs Vikram {vp}% (gap {gap} pts, needs >= {WELL_ABOVE}).")
            vc = {x["id"]: x for x in vik["scores"]["PM"]["criteria"]}
            for x in lav["scores"]["PM"]["criteria"]:
                print(f"    - {x['name']}: Lavanya {x['score']} vs Vikram {vc[x['id']]['score']} | {vc[x['id']]['reason']}")

    print(f"\nFull scores and evidence: {OUT.relative_to(config.ROOT)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
