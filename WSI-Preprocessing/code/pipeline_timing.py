# pipeline_sbatch
#!/usr/bin/env python3
"""
Parse a container run log for tagged timestamps and print a timing summary.

Expected tags in the log (format: "TAG <unix_epoch_seconds.nanoseconds>"):
    CONTAINER_ENTRY, BEFORE_KERNPROF, AFTER_KERNPROF, AFTER_REPORT,
    PYTHON_ENTRY, IMPORTS_DONE, PIPELINE_START, PIPELINE_DONE

Usage:
    python timing_summary.py <raw_log_path> <t_launch> <t_container_done>
"""
import sys


def get_ts(lines, tag, first=False):
    """Return the (first or last) timestamp for a given tag, or None."""
    matches = [
        line.split()[1]
        for line in lines
        if line.split() and line.split()[0] == tag
    ]
    if not matches:
        return None
    return float(matches[0] if first else matches[-1])


def duration(a, b):
    if a is None or b is None:
        return None
    return b - a


def fmt_row(label, value):
    if value is None:
        print(f"{label:<45} {'n/a':>10}")
    else:
        print(f"{label:<45} {value:10.3f} s")


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <raw_log_path> <t_launch> <t_container_done>")
        sys.exit(1)

    log_path = sys.argv[1]
    t_launch = float(sys.argv[2])
    t_container_done = float(sys.argv[3])

    with open(log_path) as f:
        lines = f.readlines()

    t_container_entry = get_ts(lines, "CONTAINER_ENTRY")
    t_before_kernprof = get_ts(lines, "BEFORE_KERNPROF")
    t_after_kernprof = get_ts(lines, "AFTER_KERNPROF")
    t_after_report = get_ts(lines, "AFTER_REPORT")
    t_python_entry = get_ts(lines, "PYTHON_ENTRY", first=True)
    t_imports_done = get_ts(lines, "IMPORTS_DONE", first=True)
    t_pipeline_start = get_ts(lines, "PIPELINE_START")
    t_pipeline_done = get_ts(lines, "PIPELINE_DONE")

    print()
    print("=================== TIMING SUMMARY ===================")
    fmt_row("Apptainer startup", duration(t_launch, t_container_entry))
    fmt_row("Shell setup (cd, echo)", duration(t_container_entry, t_before_kernprof))
    fmt_row("Python imports", duration(t_python_entry, t_imports_done))
    fmt_row("Pipeline execution", duration(t_pipeline_start, t_pipeline_done))
    fmt_row("kernprof + pipeline execution", duration(t_before_kernprof, t_after_kernprof))
    fmt_row("line_profiler report generation", duration(t_after_kernprof, t_after_report))
    fmt_row("CONTAINER TOTAL", duration(t_launch, t_container_done))
    print("========================================================")
    print()


if __name__ == "__main__":
    main()