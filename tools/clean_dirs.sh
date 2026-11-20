#!/usr/bin/env bash
#
# clean_dirs.sh - recursively remove "empty-ish" folders that match a rule.
#
# Usage:
#   clean_dirs.sh [options] <folder>
#
# Options:
#   -n, --dry-run     only show what would be removed
#   -r, --rule NAME   use only this rule (repeatable; default: all rules)
#   -l, --list-rules  list available rules and exit
#   -v, --verbose     also print folders that were checked but kept
#   -h, --help        show this help
#
# HOW TO ADD A NEW PATTERN
#   1. Write a function  rule_<name>()  below. It receives the folder path
#      as $1 and must return 0 if the folder should be removed, 1 otherwise.
#   2. Register it by adding <name> to the RULES array.
#   Helpers: name_is, only_contains, is_empty.
#   Folders are visited bottom-up, so after a child was removed the parent
#   is evaluated against its new content.

set -eo pipefail

# ----------------------------------------------------------------------------
# Helpers for writing rules
# ----------------------------------------------------------------------------

# name_is <dir> <basename>  -> true if the folder has exactly this name
name_is() { [[ "$(basename -- "$1")" == "$2" ]]; }

# list_entries <dir> -> fills global ENTRIES with the names (incl. hidden) in dir
list_entries() {
    ENTRIES=()
    local e
    while IFS= read -r -d '' e; do
        ENTRIES+=("${e##*/}")
    done < <(find "$1" -mindepth 1 -maxdepth 1 -print0)
}

# is_empty <dir> -> true if the folder has no entries
is_empty() { list_entries "$1"; (( ${#ENTRIES[@]} == 0 )); }

# only_contains <dir> <name>... -> true if the folder contains exactly these
# entries (as regular files, no more, no less).
only_contains() {
    local dir=$1; shift
    list_entries "$dir"
    (( ${#ENTRIES[@]} == $# )) || return 1
    local n
    for n in "$@"; do
        [[ -f "$dir/$n" && ! -L "$dir/$n" ]] || return 1
    done
}

# ----------------------------------------------------------------------------
# Rules  (add new ones here)
# ----------------------------------------------------------------------------

# Remove Python bytecode and common tool cache directories by name.
rule_pycache() { name_is "$1" "__pycache__"; }
rule_pytest_cache() { name_is "$1" ".pytest_cache"; }
rule_ruff_cache() { name_is "$1" ".ruff_cache"; }

# Remove folders whose only entry is the package marker file.
rule_only_init() { only_contains "$1" "__init__.py"; }

RULES=(
    pycache
    only_init
    pytest_cache
    ruff_cache
)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

usage() { sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

DRY_RUN=0
VERBOSE=0
SELECTED=()
TARGET=""

while (( $# )); do
    case "$1" in
        -n|--dry-run)    DRY_RUN=1 ;;
        -v|--verbose)    VERBOSE=1 ;;
        -r|--rule)       [[ $# -ge 2 ]] || { echo "error: $1 needs a name" >&2; exit 2; }
                         SELECTED+=("$2"); shift ;;
        -l|--list-rules) printf '%s\n' "${RULES[@]}"; exit 0 ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; TARGET="${1:-}"; break ;;
        -*)              echo "error: unknown option $1" >&2; exit 2 ;;
        *)               [[ -z "$TARGET" ]] || { echo "error: only one folder allowed" >&2; exit 2; }
                         TARGET="$1" ;;
    esac
    shift
done

[[ -n "$TARGET" ]] || { usage >&2; exit 2; }
[[ -d "$TARGET" ]] || { echo "error: '$TARGET' is not a directory" >&2; exit 2; }

# Validate / select rules
ACTIVE=("${RULES[@]}")
if (( ${#SELECTED[@]} )); then
    ACTIVE=()
    for r in "${SELECTED[@]}"; do
        declare -F "rule_$r" >/dev/null || { echo "error: unknown rule '$r' (see --list-rules)" >&2; exit 2; }
        ACTIVE+=("$r")
    done
fi

removed=0
# Bottom-up (-depth), snapshot first so deletions don't disturb traversal.
# The target folder itself is never removed (-mindepth 1).
DIRS=()
while IFS= read -r -d '' d; do
    DIRS+=("$d")
done < <(find "$TARGET" -mindepth 1 -depth -type d -print0)

for dir in "${DIRS[@]}"; do
    [[ -d "$dir" ]] || continue            # already gone
    matched=""
    for r in "${ACTIVE[@]}"; do
        if "rule_$r" "$dir"; then matched=$r; break; fi
    done
    if [[ -n "$matched" ]]; then
        if (( DRY_RUN )); then
            echo "[dry-run] would remove: $dir  (rule: $matched)"
        else
            rm -rf -- "$dir"
            echo "removed: $dir  (rule: $matched)"
        fi
        removed=$((removed + 1))
    elif (( VERBOSE )); then
        echo "kept: $dir"
    fi
done

if (( DRY_RUN )); then
    echo "Done: $removed folder(s) would be removed."
else
    echo "Done: $removed folder(s) removed."
fi
