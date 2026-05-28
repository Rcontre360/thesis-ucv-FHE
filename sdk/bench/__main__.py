import sys

from bench.bench import orchestrate, run_process

argv = sys.argv[1:]
if not argv:
    print("usage: python -m bench <case> [<process>]", file=sys.stderr)
    raise SystemExit(2)

if len(argv) == 1:
    orchestrate(argv[0])
else:
    run_process(argv[0], argv[1])
