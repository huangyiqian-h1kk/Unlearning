# Safely synchronizing the repository cleanup to SQUID

`git rm --cached` leaves a file in the cleanup author's working tree, but records its deletion in the commit. A later pull on SQUID applies that recorded deletion to SQUID's tracked working tree. Ignore rules do not protect a file while it is still tracked, so back up required experiment artifacts first.

## Before the cleanup is merged

Obtain the preparation script without switching branches if necessary:

```bash
git show <phase-3a-ref>:scripts/repository/backup_before_cleanup.sh > /tmp/backup_before_cleanup.sh
chmod +x /tmp/backup_before_cleanup.sh
```

After a proposed cleanup ref is available locally, create a new backup directory on external storage (never inside the checkout):

```bash
/tmp/backup_before_cleanup.sh <pre-cleanup-ref> <cleanup-ref> /external/squid-backups/unlearning-YYYYMMDD
cd /external/squid-backups/unlearning-YYYYMMDD
sha256sum -c manifest.tsv.sha256
```

The script deterministically discovers paths deleted between the refs, copies only locally existing regular files with repository-relative paths, writes `manifest.tsv` with sizes and SHA-256 hashes, and verifies the manifest. It never pulls, merges, deletes, stages, or edits repository files.

## Apply and restore

Only after the backup verifies, apply the cleanup using the site's normal reviewed pull procedure. Then restore all backed-up files, or list selected repository-relative paths:

```bash
scripts/repository/restore_after_cleanup.sh /external/squid-backups/unlearning-YYYYMMDD
scripts/repository/restore_after_cleanup.sh /external/squid-backups/unlearning-YYYYMMDD llm2vec/grid_search/configs/example.json
```

Restore rejects absolute/traversal manifest paths, damaged manifests or members, tracked conflicts, existing destination files, and targets not covered by ignore rules. It verifies size and hash, and never stages or commits. Confirm the result explicitly:

```bash
git check-ignore -v --no-index llm2vec/grid_search/configs/example.json
git ls-files --error-unmatch llm2vec/grid_search/configs/example.json && echo UNEXPECTED_TRACKED || echo untracked
git status --short --ignored
```
