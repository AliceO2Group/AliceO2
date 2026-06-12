#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


LINE_RE = re.compile(
    r"\[[^\]]*gpu-reconstruction[^\]]*\]:\s*"
    r"\[(?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\[INFO\]\s*"
    r"(?P<kind>Processing timeslice:|Done processing timeslice:)"
    r"(?P<timeslice>\d+)"
)


def parse_hms_to_seconds(hms: str) -> float:
    hhmmss, *frac = hms.split(".")
    h, m, s = map(int, hhmmss.split(":"))

    seconds = h * 3600 + m * 60 + s

    if frac:
        seconds += float("0." + frac[0])

    return seconds


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def read_timeslice_durations(logfile: Path):
    starts = {}
    ends = {}
    durations = {}

    day_offset = 0.0
    previous_raw_timestamp = None

    with logfile.open("r", errors="replace") as f:
        for line in f:
            match = LINE_RE.search(line)
            if not match:
                continue

            raw_timestamp = parse_hms_to_seconds(match.group("time"))

            # Handle midnight wraparound in log order
            if (
                previous_raw_timestamp is not None
                and raw_timestamp < previous_raw_timestamp
            ):
                day_offset += 24 * 3600

            previous_raw_timestamp = raw_timestamp
            timestamp = raw_timestamp + day_offset

            timeslice = int(match.group("timeslice"))
            kind = match.group("kind")

            if kind == "Processing timeslice:":
                starts[timeslice] = timestamp

            elif kind == "Done processing timeslice:":
                if timeslice not in starts:
                    continue

                start = starts[timeslice]
                end = timestamp

                durations[timeslice] = end - start
                ends[timeslice] = end

    return durations, starts, ends

def warn_about_processing_downtime(starts, ends, tolerance_s=0.001):
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    common_timeslices = sorted(set(starts) & set(ends))

    if len(common_timeslices) <= 4:
        return set()

    ignored_timeslices = set(common_timeslices[:2] + common_timeslices[-2:])
    excluded_timeslices = set()

    for ts in common_timeslices:
        next_ts = ts + 1

        if next_ts not in starts:
            continue

        # Do not warn if the boundary touches one of the first two or last two
        # complete timeslices.
        if ts in ignored_timeslices or next_ts in ignored_timeslices:
            continue

        gap = starts[next_ts] - ends[ts]

        if gap > tolerance_s:
            affected = {ts - 1, ts, next_ts, next_ts + 1}
            affected = {
                x for x in affected
                if x in common_timeslices and x not in ignored_timeslices
            }

            print(
                f"{YELLOW}{BOLD}WARNING:{RESET} "
                f"{YELLOW}Processing downtime detected between "
                f"timeslice {ts} and {next_ts}: "
                f"end[{ts}] -> start[{next_ts}] gap = {gap * 1000:.3f} ms. "
                f"Excluding timeslices {sorted(affected)} from calculation.{RESET}",
                flush=True,
            )

            excluded_timeslices.update(affected)

        elif gap < -tolerance_s:
            affected = {ts - 1, ts, next_ts, next_ts + 1}
            affected = {
                x for x in affected
                if x in common_timeslices and x not in ignored_timeslices
            }

            print(
                f"{RED}{BOLD}WARNING:{RESET} "
                f"{RED}Processing overlap or timestamp ordering issue between "
                f"timeslice {ts} and {next_ts}: "
                f"start[{next_ts}] is {-gap * 1000:.3f} ms before end[{ts}]. "
                f"Excluding timeslices {sorted(affected)} from calculation.{RESET}",
                flush=True,
            )

            excluded_timeslices.update(affected)

    return excluded_timeslices

def fit_gaussian_to_histogram(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    sample_mean = np.mean(values)
    sample_sigma = np.std(values, ddof=1)

    if not SCIPY_AVAILABLE:
        return None, counts, edges

    nonzero = counts > 0
    if np.count_nonzero(nonzero) < 3:
        return None, counts, edges

    x = centers[nonzero]
    y = counts[nonzero]

    p0 = [np.max(y), sample_mean, sample_sigma]

    try:
        popt, _ = curve_fit(
            gaussian,
            x,
            y,
            p0=p0,
            maxfev=10000,
            bounds=(
                [0.0, -np.inf, 1e-12],
                [np.inf, np.inf, np.inf],
            ),
        )
        return popt, counts, edges
    except Exception:
        return None, counts, edges


def main():
    parser = argparse.ArgumentParser(
        description="Analyze gpu-reconstruction timeslice processing durations."
    )
    parser.add_argument("-l", "--logfile", type=Path, help="Path to the log file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("gpu_reconstruction_times.png"),
        help="Output plot filename",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins",
    )
    parser.add_argument(
        "--unit",
        choices=["s", "ms"],
        default="s",
        help="Plot durations in seconds or milliseconds",
    )
    parser.add_argument(
        "--gap-tolerance-ms",
        type=float,
        default=1.0,
        help="Allowed gap between end of timeslice n and start of timeslice n+1 in ms",
    )

    args = parser.parse_args()

    durations_by_timeslice, starts_by_timeslice, ends_by_timeslice = read_timeslice_durations(
        args.logfile
    )

    excluded_timeslices = warn_about_processing_downtime(
        starts_by_timeslice,
        ends_by_timeslice,
        tolerance_s=args.gap_tolerance_ms / 1000.0,
    )
    if excluded_timeslices:
        print(
            f"Excluded timeslices due to downtime/overlap: "
            f"{sorted(excluded_timeslices)}"
        )

    if len(durations_by_timeslice) < 5:
        raise RuntimeError(
            f"Found only {len(durations_by_timeslice)} complete timeslices. "
            "Need at least 5 to drop first two and last two."
        )

    timeslices = sorted(durations_by_timeslice)
    trimmed_timeslices = [
        ts for ts in timeslices[2:-2]
        if ts not in excluded_timeslices
    ]

    values = np.array(
        [durations_by_timeslice[ts] for ts in trimmed_timeslices],
        dtype=float,
    )

    if args.unit == "ms":
        values *= 1000.0
        unit_label = "ms"
    else:
        unit_label = "s"

    n_total = len(timeslices)
    n_used = len(values)

    sample_mean = np.mean(values)
    sample_sigma = np.std(values, ddof=1)

    avg_from_sum = np.mean(values)

    fit_result, counts, edges = fit_gaussian_to_histogram(values, args.bins)

    plt.figure(figsize=(9, 6))

    plt.hist(
        values,
        bins=args.bins,
        histtype="stepfilled",
        alpha=0.45,
        label="Timeslice duration distribution",
    )

    if fit_result is not None:
        amp, fit_mean, fit_sigma = fit_result
        xfit = np.linspace(edges[0], edges[-1], 1000)
        yfit = gaussian(xfit, amp, fit_mean, fit_sigma)

        plt.plot(
            xfit,
            yfit,
            linewidth=2,
            label=f"Gaussian fit: mean={fit_mean:.4g} {unit_label}, sigma={fit_sigma:.4g} {unit_label}",
        )
    else:
        plt.plot([], [], label="Gaussian fit: unavailable")

    plt.plot(
        [],
        [],
        label=f"Sample: mean={sample_mean:.4g} {unit_label}, sigma={sample_sigma:.4g} {unit_label}",
    )

    plt.xlabel(f"Processing time per timeslice [{unit_label}]")
    plt.ylabel("Entries")
    plt.title("gpu-reconstruction timeslice processing duration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)

    print(f"Input file: {args.logfile}")
    print(f"Complete timeslices found: {n_total}")
    print(f"Timeslices used after dropping first/last two: {n_used}")
    print(f"First used timeslice: {trimmed_timeslices[0]}")
    print(f"Last used timeslice: {trimmed_timeslices[-1]}")
    print(f"Average duration = sum(durations)/(processed timeslices - 4): {avg_from_sum:.6g} {unit_label}")
    print(f"Sample mean: {sample_mean:.6g} {unit_label}")
    print(f"Sample sigma: {sample_sigma:.6g} {unit_label}")

    if fit_result is not None:
        _, fit_mean, fit_sigma = fit_result
        print(f"Gaussian fit mean: {fit_mean:.6g} {unit_label}")
        print(f"Gaussian fit sigma: {fit_sigma:.6g} {unit_label}")
    else:
        print("Gaussian fit failed or scipy is unavailable.")

    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()