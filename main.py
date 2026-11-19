"""Trophy Dilution Forecaster.

Reads every case file in the input directory and writes one forecast file
per case. For every trophy the program forecasts its Exophase earn rate at
the case target as five quantiles (levels 0.05, 0.25, 0.5, 0.75, 0.95).

The idea: an earn rate is a ratio, earned players over tracked players, and
both sides of the ratio grow after launch. So the program projects both
counts to the target instead of extrapolating the raw rate:

* Tracked players follow a power law in time since release; the program
  fits log(players) against log(1 + hours since release) over the reads,
  and falls back to a unit exponent anchored on the last read when the
  history is flat (Exophase refreshes in jumps, so a flat stretch of reads
  still hides growth that the next refresh reveals).
* Each trophy's earned count follows its own power law in the same clock,
  fitted the same way. When a trophy's earned count never moved, the
  program assumes the newcomers earn it at a share of its current rate
  that grows with how common the trophy already is: everyone picks up the
  opening trophies fast, almost nobody picks up the rare ones.
* A second view fits the logit of the rate itself against the same clock;
  the two medians are blended in logit space so neither view can run away
  with the forecast on its own.

The spread is a logit-scale width that grows with the horizon and is
widened further when the history is flat (a flat history means no refresh
has been seen, so the next one is a bigger unknown) and when the fits
leave large residuals. It is deliberately wider than a plain random walk,
because a refresh can move every rate at once and a band that misses the
refresh pays for every trophy outside it.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
NORMAL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)


def parse_instant(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def logit(pct):
    p = clamp(pct / 100.0, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit(z):
    z = clamp(z, -12.0, 12.0)
    return 100.0 / (1.0 + math.exp(-z))


def fit_line(xs, ys):
    """Least-squares line through (xs, ys); returns (slope, intercept)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def project_players(reads, clock, target_x):
    """Project the tracked-player count at the target clock value."""
    xs = []
    ys = []
    for read in reads:
        players = read.get("players_tracked")
        if players and players > 0:
            xs.append(clock(read["at"]))
            ys.append(math.log(players))
    last_players = reads[-1].get("players_tracked") or 1
    latest_read_x = clock(reads[-1]["at"])
    horizon_gap = max(0.0, target_x - latest_read_x)
    players_span = (max(ys) - min(ys)) if ys else 0.0
    if len(xs) >= 3 and players_span > 1e-9:
        fit = fit_line(xs, ys)
    else:
        fit = None
    if fit is not None:
        slope, intercept = fit
        slope = clamp(slope, -0.5, 4.0)
        guess = math.exp(intercept + slope * target_x)
    else:
        # The reads have not moved at all, which means the tracker has yet
        # to publish a new snapshot: the growth the next snapshot reveals
        # is still hidden. Project the pool as if it kept climbing on a
        # unit power law in time since release, with extra room for the
        # snapshot the horizon is likely to catch.
        guess = last_players * math.exp(horizon_gap) * (1.0 + 0.6 * horizon_gap)
    # A refresh can only add tracked players, never remove them, and the
    # launch-week pool grows fast; keep the projection sane either way.
    return clamp(guess, last_players, last_players * 400.0)


def project_earned(series, target_players, last_players, last_rate,
                   target_x):
    """Project one trophy's earned count at the target clock value.

    `series` is a list of (clock value, earned count) pairs, oldest first.
    """
    xs = []
    ys = []
    for x, earned in series:
        if earned > 0:
            xs.append(x)
            ys.append(math.log(earned))
    earned_span = (max(ys) - min(ys)) if ys else 0.0
    if earned_span > 1e-9 and len(xs) >= 3:
        fit = fit_line(xs, ys)
    else:
        fit = None
    if fit is not None:
        slope, intercept = fit
        slope = clamp(slope, -1.0, 8.0)
        guess = math.exp(intercept + slope * target_x)
        return clamp(guess, 0.05, target_players)
    # Flat earned history: the newcomers do the growing. A newcomer earns
    # this trophy with a probability that is a share of its current rate,
    # small for rare trophies and near the rate itself for common ones.
    last_earned = series[-1][1] if series else 0.0
    newcomers = max(0.0, target_players - last_players)
    share = clamp(last_rate / 100.0, 0.0, 1.0)
    catchup = 0.25 + 0.75 * math.sqrt(share)
    guess = last_earned + newcomers * share * catchup
    return clamp(guess, 0.05, target_players)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    release = parse_instant(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_instant(case["target"])
    checkpoint = parse_instant(case["at"])
    horizon_days = max(0.0, case.get("horizon_minutes", 0) / 1440.0)

    def clock(when):
        moment = parse_instant(when)
        hours = (moment - release).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    target_x = clock(case["target"])
    _ = (target, checkpoint)

    flat = True
    if len(history) >= 2:
        seen_players = {r.get("players_tracked") for r in history}
        seen_rates = set()
        for read in history:
            for rate in read.get("rates") or []:
                if rate is not None:
                    seen_rates.add(rate)
        flat = len(seen_players) < 2 and len(seen_rates) < 2

    players_target = project_players(history, clock, target_x) if history else 1.0
    last_players = history[-1].get("players_tracked") if history else 1
    if not last_players or last_players <= 0:
        last_players = 1

    # Residual spread of the log-player fit: how noisy the growth was.
    resid = 0.0
    if len(history) >= 3:
        xs = [clock(r["at"]) for r in history]
        ys = [math.log(r["players_tracked"]) for r in history
              if r.get("players_tracked")]
        fit = fit_line(xs, ys)
        if fit is not None:
            slope, intercept = fit
            sq = [(y - (intercept + slope * x)) ** 2
                  for x, y in zip(xs, ys)]
            resid = math.sqrt(sum(sq) / len(sq))

    spread = 0.55 + 0.30 * math.sqrt(horizon_days)
    if flat:
        # No refresh seen yet: the next one is the big unknown.
        spread += 0.45
    spread += min(0.35, resid)
    spread = clamp(spread, 0.55, 1.5)

    rows = []
    for pos, trophy in enumerate(trophies):
        last_rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if pos < len(rates) and rates[pos] is not None:
                last_rate = rates[pos]
                break
        if last_rate is None:
            last_rate = 0.5

        series = []
        for read in history:
            rates = read.get("rates") or []
            players = read.get("players_tracked") or 0
            if pos < len(rates) and rates[pos] is not None and players > 0:
                series.append((clock(read["at"]),
                               max(0.05, rates[pos] * players / 100.0)))

        earned_target = project_earned(series, players_target, last_players,
                                       last_rate, target_x)
        count_median = 100.0 * earned_target / players_target

        # Second view: the logit of the rate against the same clock.
        trend_median = None
        if len(series) >= 3:
            xs = []
            ys = []
            for read in history:
                rates = read.get("rates") or []
                if pos < len(rates) and rates[pos] is not None:
                    xs.append(clock(read["at"]))
                    ys.append(logit(rates[pos]))
            fit = fit_line(xs, ys)
            if fit is not None:
                slope, intercept = fit
                slope = clamp(slope, -3.0, 3.0)
                trend_median = inv_logit(intercept + slope * target_x)

        if trend_median is not None:
            centre = 0.65 * logit(count_median) + 0.35 * logit(trend_median)
        else:
            centre = logit(count_median)

        quantiles = []
        previous = None
        for z in NORMAL_Z:
            value = round(inv_logit(centre + z * spread), 4)
            value = clamp(value, 0.01, 100.0)
            if previous is not None and value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"id": trophy["id"], "q": quantiles})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    case_files = sorted(glob.glob(os.path.join(input_dir, "*.in")))
    for path in case_files:
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        forecast = forecast_case(case)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, stem + ".out")
        text = json.dumps(forecast, separators=(",", ":"))
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
