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

CYAN = "\033[96m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

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

def format_ranges(values):
    values = sorted(values)
    if not values:
        return "[]"

    ranges = []
    start = prev = values[0]

    for value in values[1:]:
        if value == prev + 1:
            prev = value
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = value

    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")

    return "[" + ", ".join(ranges) + "]"


def analyze_processing_sequences(starts, ends, tolerance_s=0.001, n_drop_edges=2):

    complete_timeslices = sorted(set(starts) & set(ends))

    if len(complete_timeslices) <= 2 * n_drop_edges:
        return set(), [], np.nan

    used_timeslices = complete_timeslices[n_drop_edges:-n_drop_edges]
    used_set = set(used_timeslices)

    excluded_timeslices = set()
    large_gap_boundaries = set()

    for ts, next_ts in zip(complete_timeslices[:-1], complete_timeslices[1:]):
        if ts not in used_set or next_ts not in used_set:
            continue

        if next_ts != ts + 1:
            print(
                f"{RED}{BOLD}WARNING:{RESET} "
                f"{RED}Missing timeslice(s) between {ts} and {next_ts}. "
                f"Splitting processing sequence. "
                f"Excluding timeslices {ts} and {next_ts} from calculation.{RESET}",
                flush=True,
            )
            excluded_timeslices.update({ts, next_ts})
            large_gap_boundaries.add((ts, next_ts))
            continue

        gap = starts[next_ts] - ends[ts]

        if gap > tolerance_s:
            print(
                f"{YELLOW}{BOLD}WARNING:{RESET} "
                f"{YELLOW}Processing downtime detected between "
                f"timeslice {ts} and {next_ts}: "
                f"end[{ts}] -> start[{next_ts}] gap = {gap * 1000:.3f} ms. "
                f"Splitting processing sequence. "
                f"Excluding timeslices {ts} and {next_ts} from calculation.{RESET}",
                flush=True,
            )
            excluded_timeslices.update({ts, next_ts})
            large_gap_boundaries.add((ts, next_ts))

        elif gap < -tolerance_s:
            print(
                f"{RED}{BOLD}WARNING:{RESET} "
                f"{RED}Processing overlap or timestamp ordering issue between "
                f"timeslice {ts} and {next_ts}: "
                f"start[{next_ts}] is {-gap * 1000:.3f} ms before end[{ts}]. "
                f"Splitting processing sequence. "
                f"Excluding timeslices {ts} and {next_ts} from calculation.{RESET}",
                flush=True,
            )
            excluded_timeslices.update({ts, next_ts})
            large_gap_boundaries.add((ts, next_ts))

    clean_timeslices = [
        ts for ts in used_timeslices
        if ts not in excluded_timeslices
    ]

    sequences = []
    current_sequence = []

    for ts in clean_timeslices:
        if not current_sequence:
            current_sequence = [ts]
            continue

        previous_ts = current_sequence[-1]

        if (
            ts == previous_ts + 1
            and (previous_ts, ts) not in large_gap_boundaries
        ):
            current_sequence.append(ts)
        else:
            sequences.append(current_sequence)
            current_sequence = [ts]

    if current_sequence:
        sequences.append(current_sequence)

    total_wall_time = 0.0
    total_timeslices = 0

    for sequence in sequences:
        first_ts = sequence[0]
        last_ts = sequence[-1]

        # This includes small allowed gaps inside the sequence.
        idx = complete_timeslices.index(first_ts)
        if idx == 0:
            continue
        previous_ts = complete_timeslices[idx - 1]
        sequence_wall_time = ends[last_ts] - ends[previous_ts]

        total_wall_time += sequence_wall_time
        total_timeslices += len(sequence)

    if total_timeslices > 0:
        wall_time_mean = total_wall_time / total_timeslices
    else:
        wall_time_mean = np.nan

    return excluded_timeslices, sequences, wall_time_mean

def fit_gaussian_to_histogram(values, bins):
    mask = np.abs(values - np.mean(values)) < 5 * np.std(values)
    values = values[mask]
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
        default=Path("gpu_reconstruction_times.pdf"),
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
        default=50.0,
        help="Allowed gap between end of timeslice n and start of timeslice n+1 in ms",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("gpu_reconstruction_summary.txt"),
        help="Output text file for printed summary",
    )

    args = parser.parse_args()

    durations_by_timeslice, starts_by_timeslice, ends_by_timeslice = read_timeslice_durations(
        args.logfile
    )

    excluded_timeslices, processing_sequences, wall_time_mean = analyze_processing_sequences(
        starts_by_timeslice,
        ends_by_timeslice,
        tolerance_s=args.gap_tolerance_ms / 1000.0,
        n_drop_edges=2,
    )

    if excluded_timeslices:
        print(
            f"{RED}{BOLD}Excluded timeslices due to large gaps/overlaps:{RESET} "
            f"{RED}{sorted(excluded_timeslices)}{RESET}"
        )

    print(
        f"{RED}{BOLD}Continuous processing sequences used for wall-time average:{RESET} "
        f"{RED}"
        + ", ".join(format_ranges(seq) for seq in processing_sequences)
        + f"{RESET}"
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

    if args.unit == "ms":
        wall_time_mean_print = wall_time_mean * 1000.0
    else:
        wall_time_mean_print = wall_time_mean

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
    plt.savefig(args.output)

    print(f"{BOLD}Input file:{RESET} {args.logfile}")
    print(f"{CYAN}{BOLD}Complete timeslices found:{RESET} {n_total}")
    print(f"{CYAN}{BOLD}Timeslices used after dropping first/last two:{RESET} {n_used}")
    print(f"{CYAN}{BOLD}First used timeslice:{RESET} {trimmed_timeslices[0]}")
    print(f"{CYAN}{BOLD}Last used timeslice:{RESET} {trimmed_timeslices[-1]}")

    print(
        f"{GREEN}{BOLD}Wall-time mean including allowed gaps:{RESET} "
        f"{GREEN}{wall_time_mean_print:.6g} {unit_label}{RESET}"
    )

    print(
        f"{MAGENTA}{BOLD}Individual duration sample mean:{RESET} "
        f"{MAGENTA}{sample_mean:.6g} {unit_label}{RESET}"
    )
    print(
        f"{MAGENTA}{BOLD}Individual duration sample sigma:{RESET} "
        f"{MAGENTA}{sample_sigma:.6g} {unit_label}{RESET}"
    )

    if fit_result is not None:
        _, fit_mean, fit_sigma = fit_result
        print(
            f"{YELLOW}{BOLD}Gaussian fit mean:{RESET} "
            f"{YELLOW}{fit_mean:.6g} {unit_label}{RESET}"
        )
        print(
            f"{YELLOW}{BOLD}Gaussian fit sigma:{RESET} "
            f"{YELLOW}{fit_sigma:.6g} {unit_label}{RESET}"
        )
    else:
        print(f"{RED}{BOLD}Gaussian fit failed or scipy is unavailable.{RESET}")

    if args.summary_output:

        def save_summary_output(output_file: Path, lines):
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with output_file.open("w") as f:
                for line in lines:
                    f.write(line + "\n")

            print(f"Saved summary output to: {output_file}")

        summary_lines = [
            f"Input file: {args.logfile}",
            f"Complete timeslices found: {n_total}",
            f"Timeslices used after dropping first/last two: {n_used}",
            f"First used timeslice: {trimmed_timeslices[0]}",
            f"Last used timeslice: {trimmed_timeslices[-1]}",
            f"Wall-time mean including allowed gaps: {wall_time_mean_print:.6g} {unit_label}",
            f"Individual duration sample mean: {sample_mean:.6g} {unit_label}",
            f"Individual duration sample sigma: {sample_sigma:.6g} {unit_label}",
        ]

        if fit_result is not None:
            _, fit_mean, fit_sigma = fit_result
            summary_lines.extend(
                [
                    f"Gaussian fit mean: {fit_mean:.6g} {unit_label}",
                    f"Gaussian fit sigma: {fit_sigma:.6g} {unit_label}",
                ]
            )
        else:
            summary_lines.append("Gaussian fit failed or scipy is unavailable.")

        save_summary_output(args.summary_output, summary_lines)


if __name__ == "__main__":
    main()