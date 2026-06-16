#!/usr/bin/env python3

import argparse
import csv
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    
# Example usage
# python3 $O2_ROOT/prodtests/full-system-test/analyze_gpu_all_tasks.py -l log.log --unit ms --duration-source wall


CYAN = "\033[96m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


LINE_RE = re.compile(
    r"^\[(?P<pid>\d+):(?P<task>[^\]]+)\]:\s*"
    r"\[(?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]"
    r"\[(?P<level>[A-Z]+)\]\s*"
    r"(?P<kind>Processing timeslice:|Done processing timeslice:)"
    r"(?P<timeslice>\d+)"
    r"(?P<rest>.*)$"
)

WALL_RE = re.compile(r"(?:^|,\s*)wall:(?P<wall>\d+)")


@dataclass
class TaskTiming:
    durations: dict = field(default_factory=dict)
    starts: dict = field(default_factory=dict)
    ends: dict = field(default_factory=dict)
    wall_ns: dict = field(default_factory=dict)
    n_starts: int = 0
    n_ends: int = 0
    n_unmatched_ends: int = 0
    n_duplicate_pairs: int = 0


@dataclass
class TaskAnalysis:
    task: str
    n_complete: int
    n_used: int
    first_used: int | None
    last_used: int | None
    wall_time_mean: float
    sample_mean: float
    sample_sigma: float
    fit_mean: float | None
    fit_sigma: float | None
    excluded_timeslices: set
    processing_sequences: list


def parse_hms_to_seconds(hms: str) -> float:
    hhmmss, *frac = hms.split(".")
    h, m, s = map(int, hhmmss.split(":"))

    seconds = h * 3600 + m * 60 + s

    if frac:
        seconds += float("0." + frac[0])

    return seconds


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name)
    return name.strip("_") or "unnamed_task"


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


def read_all_task_timeslice_durations(logfile: Path):
    """
    Reads all task names from [pid:task] log prefixes.

    For each task, pairs Processing timeslice:N with Done processing timeslice:N.

    Pairing is FIFO per task/timeslice, so repeated entries such as
    internal-dpl-injected-dummy-sink processing the same timeslice multiple
    times are handled without overwriting earlier occurrences.

    For the final per-timeslice dictionaries, repeated task/timeslice pairs are
    collapsed by keeping the last observed complete pair. This preserves
    compatibility with the original sequence analysis.
    """

    pending_starts = defaultdict(lambda: defaultdict(deque))
    task_timings = defaultdict(TaskTiming)

    day_offset = 0.0
    previous_raw_timestamp = None

    with logfile.open("r", errors="replace") as f:
        for line in f:
            match = LINE_RE.search(line)
            if not match:
                continue

            raw_timestamp = parse_hms_to_seconds(match.group("time"))

            # Handle midnight wraparound in log order.
            # Only treat it as midnight if the timestamp jumps backwards by many hours.
            # Small backwards jumps are normal in multi-process logs.
            if previous_raw_timestamp is not None:
                backward_jump = previous_raw_timestamp - raw_timestamp

                if backward_jump > 12 * 3600:
                    day_offset += 24 * 3600

            previous_raw_timestamp = raw_timestamp
            timestamp = raw_timestamp + day_offset

            previous_raw_timestamp = raw_timestamp
            timestamp = raw_timestamp + day_offset

            task = match.group("task")
            timeslice = int(match.group("timeslice"))
            kind = match.group("kind")
            rest = match.group("rest")

            timing = task_timings[task]

            if kind == "Processing timeslice:":
                pending_starts[task][timeslice].append(timestamp)
                timing.n_starts += 1

            elif kind == "Done processing timeslice:":
                timing.n_ends += 1

                wall_match = WALL_RE.search(rest)
                wall_ns = int(wall_match.group("wall")) if wall_match else None

                if not pending_starts[task][timeslice]:
                    timing.n_unmatched_ends += 1
                    continue

                start = pending_starts[task][timeslice].popleft()
                end = timestamp

                if timeslice in timing.durations:
                    timing.n_duplicate_pairs += 1

                timing.starts[timeslice] = start
                timing.ends[timeslice] = end
                timing.durations[timeslice] = end - start

                if wall_ns is not None:
                    timing.wall_ns[timeslice] = wall_ns

    return dict(task_timings)


def analyze_processing_sequences(
    starts,
    ends,
    task_name,
    tolerance_s=0.001,
    n_drop_edges=2,
    verbose=False,
):
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
            if verbose:
                print(
                    f"{RED}{BOLD}WARNING [{task_name}]:{RESET} "
                    f"{RED}Missing timeslice(s) between {ts} and {next_ts}. "
                    f"Excluding boundary timeslices {ts} and {next_ts}.{RESET}",
                    flush=True,
                )

            excluded_timeslices.update({ts, next_ts})
            large_gap_boundaries.add((ts, next_ts))
            continue

        gap = starts[next_ts] - ends[ts]

        if gap > tolerance_s:
            if verbose:
                print(
                    f"{YELLOW}{BOLD}WARNING [{task_name}]:{RESET} "
                    f"{YELLOW}Downtime between timeslice {ts} and {next_ts}: "
                    f"{gap * 1000:.3f} ms. "
                    f"Excluding boundary timeslices {ts} and {next_ts}.{RESET}",
                    flush=True,
                )

            excluded_timeslices.update({ts, next_ts})
            large_gap_boundaries.add((ts, next_ts))

        elif gap < -tolerance_s:
            if verbose:
                print(
                    f"{RED}{BOLD}WARNING [{task_name}]:{RESET} "
                    f"{RED}Overlap/timestamp ordering issue between "
                    f"timeslice {ts} and {next_ts}: "
                    f"{-gap * 1000:.3f} ms. "
                    f"Excluding boundary timeslices {ts} and {next_ts}.{RESET}",
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

        sequence_wall_time = ends[last_ts] - starts[first_ts]

        total_wall_time += sequence_wall_time
        total_timeslices += len(sequence)

    if total_timeslices > 0:
        wall_time_mean = total_wall_time / total_timeslices
    else:
        wall_time_mean = np.nan

    return excluded_timeslices, sequences, wall_time_mean


def fit_gaussian_to_histogram(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    sample_mean = np.mean(values)
    sample_sigma = np.std(values, ddof=1) if len(values) > 1 else 0.0

    if not SCIPY_AVAILABLE:
        return None, counts, edges

    nonzero = counts > 0
    if np.count_nonzero(nonzero) < 3:
        return None, counts, edges

    if sample_sigma <= 0:
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


def get_duration_values(
    timing: TaskTiming,
    trimmed_timeslices,
    duration_source: str,
):
    """
    duration_source:
      timestamp: use end_time - start_time from log timestamps
      wall: use wall:<ns> field from Done lines, converted to seconds
      auto: use wall if available for all used timeslices, otherwise timestamp
    """

    if duration_source == "wall":
        missing = [ts for ts in trimmed_timeslices if ts not in timing.wall_ns]
        if missing:
            raise RuntimeError(
                f"Requested --duration-source wall, but wall field is missing "
                f"for {len(missing)} used timeslice(s), e.g. {missing[:5]}"
            )

        return np.array(
            [timing.wall_ns[ts] * 1e-9 for ts in trimmed_timeslices],
            dtype=float,
        ), "wall"

    if duration_source == "timestamp":
        return np.array(
            [timing.durations[ts] for ts in trimmed_timeslices],
            dtype=float,
        ), "timestamp"

    if duration_source == "auto":
        if trimmed_timeslices and all(ts in timing.wall_ns for ts in trimmed_timeslices):
            return np.array(
                [timing.wall_ns[ts] * 1e-9 for ts in trimmed_timeslices],
                dtype=float,
            ), "wall"

        return np.array(
            [timing.durations[ts] for ts in trimmed_timeslices],
            dtype=float,
        ), "timestamp"

    raise ValueError(f"Invalid duration_source: {duration_source}")


def plot_task_histogram(
    task,
    values,
    unit_label,
    bins,
    output_file,
    fit_result,
    counts,
    edges,
    sample_mean,
    sample_sigma,
    duration_source_used,
):
    plt.figure(figsize=(9, 6))

    plt.hist(
        values,
        bins=bins,
        histtype="stepfilled",
        alpha=0.45,
        label=f"Timeslice duration distribution ({duration_source_used})",
    )

    if fit_result is not None:
        amp, fit_mean, fit_sigma = fit_result
        xfit = np.linspace(edges[0], edges[-1], 1000)
        yfit = gaussian(xfit, amp, fit_mean, fit_sigma)

        plt.plot(
            xfit,
            yfit,
            linewidth=2,
            label=(
                f"Gaussian fit: mean={fit_mean:.4g} {unit_label}, "
                f"sigma={fit_sigma:.4g} {unit_label}"
            ),
        )
    else:
        plt.plot([], [], label="Gaussian fit: unavailable")

    plt.plot(
        [],
        [],
        label=(
            f"Sample: mean={sample_mean:.4g} {unit_label}, "
            f"sigma={sample_sigma:.4g} {unit_label}"
        ),
    )

    plt.xlabel(f"Processing time per timeslice [{unit_label}]")
    plt.ylabel("Entries")
    plt.title(f"{task} timeslice processing duration")
    plt.legend()
    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150)
    plt.close()


def analyze_task(
    task,
    timing,
    args,
    plot_dir,
):
    durations_by_timeslice = timing.durations
    starts_by_timeslice = timing.starts
    ends_by_timeslice = timing.ends

    if len(durations_by_timeslice) < args.min_complete:
        return None

    excluded_timeslices, processing_sequences, wall_time_mean = analyze_processing_sequences(
        starts_by_timeslice,
        ends_by_timeslice,
        task_name=task,
        tolerance_s=args.gap_tolerance_ms / 1000.0,
        n_drop_edges=args.drop_edges,
        verbose=args.verbose,
    )

    timeslices = sorted(durations_by_timeslice)

    if len(timeslices) <= 2 * args.drop_edges:
        return None

    trimmed_timeslices = [
        ts for ts in timeslices[args.drop_edges:-args.drop_edges]
        if ts not in excluded_timeslices
    ]

    if len(trimmed_timeslices) < args.min_used:
        return None

    values, duration_source_used = get_duration_values(
        timing,
        trimmed_timeslices,
        args.duration_source,
    )

    if args.unit == "ms":
        values = values * 1000.0
        unit_label = "ms"
        wall_time_mean_print = wall_time_mean * 1000.0
    else:
        unit_label = "s"
        wall_time_mean_print = wall_time_mean

    sample_mean = float(np.mean(values))
    sample_sigma = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

    fit_result, counts, edges = fit_gaussian_to_histogram(values, args.bins)

    fit_mean = None
    fit_sigma = None

    if fit_result is not None:
        _, fit_mean, fit_sigma = fit_result
        fit_mean = float(fit_mean)
        fit_sigma = float(fit_sigma)

    if not args.no_plots:
        output_file = plot_dir / f"{sanitize_filename(task)}.png"
        plot_task_histogram(
            task=task,
            values=values,
            unit_label=unit_label,
            bins=args.bins,
            output_file=output_file,
            fit_result=fit_result,
            counts=counts,
            edges=edges,
            sample_mean=sample_mean,
            sample_sigma=sample_sigma,
            duration_source_used=duration_source_used,
        )

    return TaskAnalysis(
        task=task,
        n_complete=len(timeslices),
        n_used=len(values),
        first_used=trimmed_timeslices[0] if trimmed_timeslices else None,
        last_used=trimmed_timeslices[-1] if trimmed_timeslices else None,
        wall_time_mean=float(wall_time_mean_print),
        sample_mean=sample_mean,
        sample_sigma=sample_sigma,
        fit_mean=fit_mean,
        fit_sigma=fit_sigma,
        excluded_timeslices=excluded_timeslices,
        processing_sequences=processing_sequences,
    )


def write_summary_csv(output_file: Path, analyses, task_timings, unit_label):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "task",
                "complete_timeslices",
                "used_timeslices",
                "first_used_timeslice",
                "last_used_timeslice",
                f"wall_time_mean_{unit_label}",
                f"sample_mean_{unit_label}",
                f"sample_sigma_{unit_label}",
                f"gaussian_fit_mean_{unit_label}",
                f"gaussian_fit_sigma_{unit_label}",
                "n_starts",
                "n_ends",
                "n_unmatched_ends",
                "n_duplicate_pairs",
                "excluded_timeslices",
                "processing_sequences",
            ]
        )

        for analysis in analyses:
            timing = task_timings[analysis.task]

            writer.writerow(
                [
                    analysis.task,
                    analysis.n_complete,
                    analysis.n_used,
                    analysis.first_used,
                    analysis.last_used,
                    f"{analysis.wall_time_mean:.10g}",
                    f"{analysis.sample_mean:.10g}",
                    f"{analysis.sample_sigma:.10g}",
                    (
                        f"{analysis.fit_mean:.10g}"
                        if analysis.fit_mean is not None
                        else ""
                    ),
                    (
                        f"{analysis.fit_sigma:.10g}"
                        if analysis.fit_sigma is not None
                        else ""
                    ),
                    timing.n_starts,
                    timing.n_ends,
                    timing.n_unmatched_ends,
                    timing.n_duplicate_pairs,
                    format_ranges(analysis.excluded_timeslices),
                    "; ".join(format_ranges(seq) for seq in analysis.processing_sequences),
                ]
            )


def write_summary_txt(output_file: Path, analyses, task_timings, args, unit_label):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        f.write(f"Input file: {args.logfile}\n")
        f.write(f"Duration unit: {unit_label}\n")
        f.write(f"Duration source: {args.duration_source}\n")
        f.write(f"Drop edges: {args.drop_edges}\n")
        f.write(f"Gap tolerance: {args.gap_tolerance_ms} ms\n")
        f.write(f"Analyzed tasks: {len(analyses)}\n")
        f.write("\n")

        for analysis in analyses:
            timing = task_timings[analysis.task]

            f.write("=" * 100 + "\n")
            f.write(f"Task: {analysis.task}\n")
            f.write(f"Complete timeslices found: {analysis.n_complete}\n")
            f.write(f"Timeslices used: {analysis.n_used}\n")
            f.write(f"First used timeslice: {analysis.first_used}\n")
            f.write(f"Last used timeslice: {analysis.last_used}\n")
            f.write(
                f"Wall-time mean including allowed gaps: "
                f"{analysis.wall_time_mean:.6g} {unit_label}\n"
            )
            f.write(
                f"Individual duration sample mean: "
                f"{analysis.sample_mean:.6g} {unit_label}\n"
            )
            f.write(
                f"Individual duration sample sigma: "
                f"{analysis.sample_sigma:.6g} {unit_label}\n"
            )

            if analysis.fit_mean is not None:
                f.write(
                    f"Gaussian fit mean: "
                    f"{analysis.fit_mean:.6g} {unit_label}\n"
                )
                f.write(
                    f"Gaussian fit sigma: "
                    f"{analysis.fit_sigma:.6g} {unit_label}\n"
                )
            else:
                f.write("Gaussian fit failed or scipy is unavailable.\n")

            f.write(f"Starts seen: {timing.n_starts}\n")
            f.write(f"Ends seen: {timing.n_ends}\n")
            f.write(f"Unmatched ends: {timing.n_unmatched_ends}\n")
            f.write(f"Duplicate completed task/timeslice pairs: {timing.n_duplicate_pairs}\n")
            f.write(
                f"Excluded timeslices: "
                f"{format_ranges(analysis.excluded_timeslices)}\n"
            )
            f.write(
                "Continuous processing sequences used for wall-time average: "
                + ", ".join(format_ranges(seq) for seq in analysis.processing_sequences)
                + "\n"
            )
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Auto-detect all DPL tasks in a log file and analyze timeslice "
            "processing durations for each task."
        )
    )

    parser.add_argument(
        "-l",
        "--logfile",
        type=Path,
        required=True,
        help="Path to the log file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("task_timing_analysis"),
        help="Output directory",
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
        help="Plot and summary durations in seconds or milliseconds",
    )
    parser.add_argument(
        "--duration-source",
        choices=["auto", "timestamp", "wall"],
        default="auto",
        help=(
            "Duration source. "
            "'timestamp' uses Done timestamp minus Processing timestamp. "
            "'wall' uses wall:<ns> from Done lines. "
            "'auto' uses wall if available for all used timeslices, otherwise timestamp."
        ),
    )
    parser.add_argument(
        "--gap-tolerance-ms",
        type=float,
        default=50.0,
        help="Allowed gap between end of timeslice n and start of timeslice n+1 in ms",
    )
    parser.add_argument(
        "--drop-edges",
        type=int,
        default=2,
        help="Drop this many first and last timeslices per task",
    )
    parser.add_argument(
        "--min-complete",
        type=int,
        default=5,
        help="Minimum complete timeslices required before analyzing a task",
    )
    parser.add_argument(
        "--min-used",
        type=int,
        default=1,
        help="Minimum used timeslices required after trimming/exclusions",
    )
    parser.add_argument(
        "--include-task-regex",
        type=str,
        default=None,
        help="Only analyze tasks whose name matches this regex",
    )
    parser.add_argument(
        "--exclude-task-regex",
        type=str,
        default=None,
        help="Skip tasks whose name matches this regex",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not create per-task histogram plots",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print gap/overlap warnings per task",
    )

    args = parser.parse_args()

    if not args.logfile.exists():
        raise FileNotFoundError(args.logfile)

    output_dir = args.output_dir
    plot_dir = output_dir / "plots"

    include_re = re.compile(args.include_task_regex) if args.include_task_regex else None
    exclude_re = re.compile(args.exclude_task_regex) if args.exclude_task_regex else None

    task_timings = read_all_task_timeslice_durations(args.logfile)

    if not task_timings:
        raise RuntimeError(
            "No task Processing/Done timeslice lines were found in the log file."
        )

    analyses = []

    for task in sorted(task_timings):
        if include_re and not include_re.search(task):
            continue

        if exclude_re and exclude_re.search(task):
            continue

        timing = task_timings[task]

        try:
            analysis = analyze_task(
                task=task,
                timing=timing,
                args=args,
                plot_dir=plot_dir,
            )
        except RuntimeError as exc:
            print(
                f"{YELLOW}{BOLD}Skipping task {task}:{RESET} "
                f"{YELLOW}{exc}{RESET}"
            )
            continue

        if analysis is not None:
            analyses.append(analysis)

    if not analyses:
        raise RuntimeError(
            "Found tasks, but none had enough complete timeslices after filtering."
        )

    analyses.sort(key=lambda x: x.sample_mean, reverse=True)

    unit_label = args.unit

    summary_csv = output_dir / "summary.csv"
    summary_txt = output_dir / "summary.txt"

    write_summary_csv(summary_csv, analyses, task_timings, unit_label)
    write_summary_txt(summary_txt, analyses, task_timings, args, unit_label)

    print(f"{BOLD}Input file:{RESET} {args.logfile}")
    print(f"{CYAN}{BOLD}Tasks found:{RESET} {len(task_timings)}")
    print(f"{CYAN}{BOLD}Tasks analyzed:{RESET} {len(analyses)}")
    print(f"{CYAN}{BOLD}Summary CSV:{RESET} {summary_csv}")
    print(f"{CYAN}{BOLD}Summary TXT:{RESET} {summary_txt}")

    if not args.no_plots:
        print(f"{CYAN}{BOLD}Plots directory:{RESET} {plot_dir}")

    print()
    print(f"{BOLD}Top tasks by individual duration sample mean:{RESET}")

    for analysis in analyses[:20]:
        print(
            f"{GREEN}{analysis.task:45s}{RESET} "
            f"n={analysis.n_used:6d}  "
            f"mean={analysis.sample_mean:.6g} {unit_label}  "
            f"sigma={analysis.sample_sigma:.6g} {unit_label}  "
            f"wall-mean={analysis.wall_time_mean:.6g} {unit_label}"
        )


if __name__ == "__main__":
    main()