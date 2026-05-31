import sys

from bench.bench import orchestrate, duration, run_process

argv = sys.argv[1:]
if not argv:
    print("usage: python -m bench <case> [<process>|duration]", file=sys.stderr)
    raise SystemExit(2)

if len(argv) == 1:
    orchestrate(argv[0])
elif argv[1] == "duration":
    duration(argv[0])
else:
    run_process(argv[0], argv[1])
