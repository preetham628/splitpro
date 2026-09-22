#!/usr/bin/env bash
# Provisions the SplitPro SQLite database from schema.sql.
#
# Reads DB_PATH the same way the app does: from .env if present, else the
# environment — never hardcoded, never prompted for. Safe to re-run — it's a
# no-op if the file already exists. Prefers the sqlite3 CLI; falls back to
# the Python sqlite3 module (e.g. inside the Docker image, which doesn't
# ship the sqlite3 CLI binary) if that's not on PATH.
#
# Usage:
#   ./scripts/init_db.sh                # uses DB_PATH from .env / environment
#   DB_PATH=other.db ./scripts/init_db.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# An explicitly-passed/exported DB_PATH takes priority over .env — matches
# python-dotenv's load_dotenv(), which never overrides an already-set var.
_explicit_db_path="${DB_PATH:-}"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -n "$_explicit_db_path" ]; then
  DB_PATH="$_explicit_db_path"
fi

if [ -z "${DB_PATH:-}" ]; then
  echo "DB_PATH is not set (checked .env and the environment)." >&2
  echo "Set it in .env or export it, e.g.: DB_PATH=splitpro.db ./scripts/init_db.sh" >&2
  exit 1
fi

if [ -f "$DB_PATH" ]; then
  echo "$DB_PATH already exists — nothing to do."
  exit 0
fi

if command -v sqlite3 >/dev/null 2>&1; then
  dir="$(dirname "$DB_PATH")"
  [ "$dir" != "." ] && mkdir -p "$dir"
  echo "Creating $DB_PATH via sqlite3 CLI..."
  sqlite3 "$DB_PATH" < schema.sql
else
  echo "sqlite3 CLI not found — falling back to the Python path (e.g. inside Docker)..."
  python3 -m core.database "$DB_PATH"
fi

echo "Database ready at $DB_PATH"
