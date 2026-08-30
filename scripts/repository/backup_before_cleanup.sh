#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 3 ]] || die "usage: $0 OLD_REF CLEANUP_REF EXTERNAL_BACKUP_DIR"
old_ref=$1 cleanup_ref=$2 destination=$3
[[ -n $destination ]] || die "backup destination is empty"
repo=$(git rev-parse --show-toplevel) || die "not in a Git repository"
repo=$(realpath -e "$repo")
home=$(realpath -e "$HOME")
destination=$(realpath -m "$destination")
[[ $destination != / && $destination != "$home" && $destination != "$repo" ]] || die "unsafe backup destination"
[[ $destination != "$repo"/* ]] || die "backup must be outside the repository"
[[ $destination != "$home"/.. && $destination != "$(dirname "$repo")" ]] || die "destination is too broad"
git rev-parse --verify "$old_ref^{commit}" >/dev/null
git rev-parse --verify "$cleanup_ref^{commit}" >/dev/null
[[ ! -e $destination ]] || die "destination already exists"
mkdir -p "$destination/files"
paths_file=$(mktemp)
trap 'rm -f "$paths_file"' EXIT
git diff --name-only --diff-filter=D -z "$old_ref" "$cleanup_ref" |
  python -c 'import sys; p=sys.stdin.buffer.read().split(b"\0"); sys.stdout.buffer.write(b"\0".join(sorted(x for x in p if x))+b"\0")' >"$paths_file"
while IFS= read -r -d '' path; do
  [[ $path != /* && $path != *../* && $path != ../* ]] || die "unsafe repository path: $path"
  if [[ -f "$repo/$path" ]]; then
    mkdir -p "$destination/files/$(dirname "$path")"
    cp -p -- "$repo/$path" "$destination/files/$path"
  fi
done <"$paths_file"
python - "$destination" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1]); files = root / "files"
rows = []
for item in sorted(p for p in files.rglob("*") if p.is_file()):
    rel = item.relative_to(files).as_posix()
    digest = hashlib.sha256(item.read_bytes()).hexdigest()
    rows.append(f"{digest}\t{item.stat().st_size}\t{rel}\n")
(root / "manifest.tsv").write_text("sha256\tsize\tpath\n" + "".join(rows), encoding="utf-8")
PY
sha256sum "$destination/manifest.tsv" >"$destination/manifest.tsv.sha256"
(cd "$destination" && sha256sum -c manifest.tsv.sha256 >/dev/null)
printf 'Verified backup: %s\n' "$destination"
