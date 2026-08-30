#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
[[ $# -ge 1 ]] || die "usage: $0 VERIFIED_BACKUP_DIR [REPOSITORY_PATH ...]"
backup=$(realpath -e "$1") || die "backup does not exist"; shift
[[ -f $backup/manifest.tsv && -f $backup/manifest.tsv.sha256 ]] || die "backup is incomplete"
(cd "$backup" && sha256sum -c manifest.tsv.sha256 >/dev/null) || die "manifest verification failed"
repo=$(git rev-parse --show-toplevel) || die "not in a Git repository"
python - "$backup/manifest.tsv" <<'PY'
import pathlib, sys
for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()[1:]:
    _, _, name = line.split("\t", 2)
    p = pathlib.PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not name:
        raise SystemExit(f"unsafe manifest path: {name}")
PY
declare -A wanted=()
for path in "$@"; do wanted["$path"]=1; done
while IFS=$'\t' read -r digest size path; do
  [[ $digest == sha256 ]] && continue
  (( ${#wanted[@]} == 0 )) || [[ -n ${wanted[$path]+x} ]] || continue
  [[ -f $backup/files/$path ]] || die "missing backup member: $path"
  [[ $(sha256sum "$backup/files/$path" | cut -d' ' -f1) == "$digest" ]] || die "hash mismatch: $path"
  [[ $(stat -c %s "$backup/files/$path") == "$size" ]] || die "size mismatch: $path"
  if git -C "$repo" ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then die "refusing to overwrite tracked path: $path"; fi
  [[ ! -e $repo/$path ]] || die "refusing to overwrite existing path: $path"
  git -C "$repo" check-ignore -q --no-index -- "$path" || die "restore target is not ignored: $path"
  mkdir -p "$repo/$(dirname "$path")"
  cp -p -- "$backup/files/$path" "$repo/$path"
  [[ $(sha256sum "$repo/$path" | cut -d' ' -f1) == "$digest" ]] || die "restored hash mismatch: $path"
  git -C "$repo" ls-files --error-unmatch -- "$path" >/dev/null 2>&1 && die "restored path became tracked: $path"
  printf 'Restored and verified: %s\n' "$path"
done <"$backup/manifest.tsv"
