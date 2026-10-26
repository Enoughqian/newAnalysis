#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
if [[ ! -x .venv/bin/python ]]; then
  "${PYTHON_BIN:-python3}" -m venv .venv
fi
dependency_hash="$(.venv/bin/python -c 'import hashlib; print(hashlib.sha256(open("requirements.lock.txt", "rb").read()).hexdigest())')"
if [[ ! -f .venv/.requirements.sha256 ]] || [[ "$(cat .venv/.requirements.sha256)" != "$dependency_hash" ]]; then
  .venv/bin/python -m pip install -r requirements.lock.txt
  printf '%s\n' "$dependency_hash" > .venv/.requirements.sha256
fi
mode="${1:-serve}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  setup) exit 0 ;;
  serve) exec .venv/bin/python -m gunicorn -c gunicorn.conf.py main_news_api:app "$@" ;;
  dev) exec .venv/bin/python main_news_api.py "$@" ;;
  test)
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python scripts/e2e.py --output test-results/e2e-python
    exec .venv/bin/python scripts/e2e.py --scheduled --server gunicorn --output test-results/e2e-scheduled "$@"
    ;;
  models) exec .venv/bin/python scripts/check_models.py "$@" ;;
  samples) exec .venv/bin/python scripts/audit_samples.py "$@" ;;
  e2e) exec .venv/bin/python scripts/e2e.py "$@" ;;
  doctor) exec .venv/bin/python scripts/check_environment.py "$@" ;;
  *) echo '用法: ./run.sh {setup|serve|dev|test|models|samples|e2e|doctor}' >&2; exit 2 ;;
esac
