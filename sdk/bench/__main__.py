import sys

from bench.bench import LIBRARIES, duration, orchestrate, run_process

argv = sys.argv[1:]
if not argv:
    print(
        "usage: python -m bench <case> [<library>...|<process>|duration]\n"
        f"  <library> one or more of {LIBRARIES} (run only those; omit to run all)",
        file=sys.stderr,
    )
    raise SystemExit(2)

case = argv[0]
rest = argv[1:]

if not rest:
    orchestrate(case)
elif rest == ["duration"]:
    duration(case)
elif rest[0].startswith("run_") or rest[0] in ("train", "profile_sdk"):
    run_process(case, rest[0])
else:
    unknown = [lib for lib in rest if lib not in LIBRARIES]
    if unknown:
        print(f"unknown library {unknown}; choose from {LIBRARIES}", file=sys.stderr)
        raise SystemExit(2)
    orchestrate(case, [f"run_{lib}" for lib in rest])
