#!/usr/bin/env bash
# GoTrain post-chain publication (after scripts/chain.sh has worked the whole queue).
#
#   scripts/publish.sh --stage chain|main|all [--dry-run] [--chain-dir DIR] [--main-dir DIR] [--cards-from-chain]
#
# STAGE chain (in --chain-dir, default ~/GoTrain, branch run/chain). Refused unless `scripts/chain.sh status` shows the
# queue complete, there is no PAUSE marker and no chain process works in that checkout. Complete = no `next:` item
# (every queue line is in done.txt). Not done == total: queue.txt lists `verify m-SA-crop-type main` twice and
# chain.sh skips a repeated line, so done.txt ends one line short of queue.txt.
#   a. python -m gotrain.verify --session <id> for every main-arm session in workspace/sessions.jsonl, and for the OFF
#      arm in experiments/playbook-ab/m-SA-crop-type/off/workspace
#   b. python -m gotrain.registry sync; python -m gotrain.model_card <task> for each task in workspace/vault.jsonl,
#      by default run with --main-dir's gotrain/model_card.py (see --cards-from-chain)
#   c. python experiments/playbook-ab/analyze.py --on ...@workspace --off ...@<off workspace> (metrics.json + curves.png
#      in experiments/playbook-ab/m-SA-crop-type/); a missing A/B stops the publication
#   d. python -m gotrain.report --out report/
#   e. python -m gotrain.verify --full
#   f. copy $GOTRAIN_STATE_DIR/tested.jsonl to workspace/test_registry.jsonl; git add workspace logs experiments report
#      (files > 5 MB unstaged); commit "publish: registry, model cards, A/B analysis, report"; git push origin run/chain
#   Skipped when run/chain's publish commit is already merged into origin/main.
# STAGE main (in --main-dir, default ~/GoTrain-main, branch main). Refused with uncommitted changes to tracked files
# other than README.md, report/ and docs/DELIVERABLES.md (operator status edits ride along in the publish commit).
#   g. git fetch origin; git merge --no-ff origin/run/chain (skipped when already merged; refused when origin/run/chain
#      has no publish commit, a local run/chain has unpushed commits, or the merge would conflict)
#   h. python -m gotrain.report --out report/ --readme; only the block between the README's results markers may change
#   i. python -m gotrain.verify --full. Checkpoints are gitignored, so they exist only in the chain checkout: verify
#      reads GOTRAIN_WORKSPACE=<chain-dir>/workspace after checking that main's workspace/ tree is exactly the chain
#      checkout's committed workspace/ (= origin/run/chain's); the git checks read this repository (main + origin/main)
#   j. reminder for docs/DELIVERABLES.md; commit "Publish: README results filled from the report"; git tag -a submission;
#      git push origin main submission
#
# Every command is printed (+ ...) before it runs. Python runs under nice -n 19. Re-runnable: registry sync and the
# merge skip what is done, empty commits are skipped, an existing `submission` tag at HEAD is kept (elsewhere: refused).
#
# --dry-run          everything except git commit / push / tag / merge (and `git add`: the files it would stage are
#                    listed instead). Failures are collected instead of stopping, and the run exits 1 if any. Stage
#                    main writes its report and filled README to a scratch folder, so the worktree stays clean.
#                    Stage chain still writes registry/cards/report/test_registry into the chain checkout (uncommitted).
# --tag              stage main: also set and push the `submission` tag. Off by default, so the docs session's README
#                    and the MCP work can be merged into main after the publish; run `--stage main --tag` once last.
# --cards-from-chain write the model cards with run/chain's frozen gotrain/model_card.py instead of the default,
#                    --main-dir's committed one. Main has the later card fixes (commit 7d3cf54: lesson z in the
#                    lesson's own direction, seed sigma to 4 decimals); the frozen copy prints a negative-sign lesson
#                    that claims a gain as "z -1.64" and seed sigma unrounded. Cards live in workspace/, which main may
#                    never change, so the cards committed here are the published ones. Stage chain refuses when
#                    --main-dir's gotrain/model_card.py has uncommitted changes. (--cards-from-main = the default.)
# Environment: GOTRAIN_STATE_DIR (default ~/.gotrain-state) for the queue and tested.jsonl; GOTRAIN_WORKSPACE is
# ignored (each command gets its workspace explicitly).
set -euo pipefail

AB_TASK=m-SA-crop-type
CHAIN_MSG="publish: registry, model cards, A/B analysis, report"
MAIN_MSG="Publish: README results filled from the report"
MERGE_MSG="Merge run/chain: results of the GPU chain"
TAG=submission
TAG_MSG="GoTrain submission (AUB ACM x RASID hackathon, 1 Nov 2026): results of the GPU chain. verify --full passed on the build machine against the chain checkout's checkpoints (gitignored; a fresh clone cannot re-check artifact hashes)"
MAX_BYTES=5242880

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }

DRY=0 STAGE="" CARDS_FROM_MAIN=1 DO_TAG=0
CHAIN="${HOME}/GoTrain" MAIN="${HOME}/GoTrain-main"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --stage) STAGE="${2:-}"; shift ;;
    --stage=*) STAGE="${1#*=}" ;;
    --chain-dir) CHAIN="${2:?--chain-dir DIR}"; shift ;;
    --main-dir) MAIN="${2:?--main-dir DIR}"; shift ;;
    --cards-from-main) CARDS_FROM_MAIN=1 ;;
    --cards-from-chain) CARDS_FROM_MAIN=0 ;;
    --tag) DO_TAG=1 ;;
    -h|--help) usage 0 ;;
    *) echo "publish: unknown argument $1" >&2; usage ;;
  esac
  shift
done
case "$STAGE" in chain|main|all) ;; *) echo "publish: --stage chain|main|all is required" >&2; usage ;; esac
CHAIN="$(cd "$CHAIN" && pwd -P)"
MAIN="$(cd "$MAIN" && pwd -P)"
STATE="${GOTRAIN_STATE_DIR:-$HOME/.gotrain-state}"
export GOTRAIN_STATE_DIR="$STATE"
unset GOTRAIN_WORKSPACE
TMP="$(mktemp -d "${TMPDIR:-/tmp}/gotrain-publish.XXXXXX")"
FAILS=()

say()  { printf '\n== %s\n' "$*"; }
note() { printf 'publish: %s\n' "$*"; }
die()  { printf 'publish: ERROR: %s\n' "$*" >&2; exit 1; }
# a failed check: stops a real run; a dry run records it, says where the real run would stop, and goes on
fail() {
  if [ "$DRY" = 1 ]; then FAILS+=("$1"); printf 'publish: DRY-RUN FAILURE: %s (a real run STOPS here)\n' "$1" >&2
  else die "$1"; fi
}
show() { local q; q="$(printf '%q ' "$@")"; printf '+ [%s] %s\n' "$PWD" "${q% }"; }
run()  { show "$@"; "$@"; }
# python from the current checkout, at low priority; leading VAR=value words go to env
pyrun() { run nice -n 19 env PYTHONPATH="$PWD" "$@"; }
# git commands that write history or a remote: printed but skipped in --dry-run
gitw() { if [ "$DRY" = 1 ]; then show git "$@"; note "[dry-run] skipped: git $1"; else run git "$@"; fi; }

# session ids with a session_start record of the given arm in <workspace>/sessions.jsonl (read-only)
session_ids() {
  printf '+ [%s] python (read %s: session_start records, arm %s)\n' "$PWD" "$1/sessions.jsonl" "$2" >&2
  nice -n 19 python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
p, arm, seen = Path(sys.argv[1]) / "sessions.jsonl", sys.argv[2], []
for line in (p.read_text().splitlines() if p.exists() else []):
    r = json.loads(line) if line.strip() else {}
    if r.get("type") == "session_start" and r.get("arm") == arm and r["session_id"] not in seen:
        seen.append(r["session_id"])
print("\n".join(seen)) if seen else None
PY
}

# finalized tasks, in vault order (read-only)
vault_tasks() {
  printf '+ [%s] python (read %s: finalized tasks)\n' "$PWD" "$1/vault.jsonl" >&2
  nice -n 19 python - "$1" <<'PY'
import json, sys
from pathlib import Path
p, seen = Path(sys.argv[1]) / "vault.jsonl", []
for line in (p.read_text().splitlines() if p.exists() else []):
    if line.strip() and json.loads(line)["task"] not in seen:
        seen.append(json.loads(line)["task"])
print("\n".join(seen)) if seen else None
PY
}

# pids of chain processes (runner, session driver, training) whose working directory is $1 (not this script's ancestors)
chain_procs() {
  local pid cwd mine=" " p=$$
  while [ -n "$p" ] && [ "$p" -gt 1 ]; do mine+="$p "; p="$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null || true)"; done
  for pid in $(pgrep -f 'chain\.sh run|gotrain\.run|gotrain\.engine\.(train|data)' || true); do
    case "$mine" in *" $pid "*) continue ;; esac
    cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
    [ "$cwd" = "$1" ] && echo "$pid"
  done
  return 0
}

contains() { local x="$1"; shift; printf '%s\n' "$@" | grep -qxF -- "$x"; }

# ---------------------------------------------------------------------------------------------------- stage chain
stage_chain() {
  say "STAGE chain: $CHAIN"
  cd "$CHAIN"
  local branch; branch="$(git rev-parse --abbrev-ref HEAD)"
  [ "$branch" = run/chain ] || die "$CHAIN is on branch $branch, not run/chain: refused"

  run git fetch -q origin || note "WARNING: git fetch failed; using the local origin/* refs"
  if git rev-parse -q --verify origin/main >/dev/null && [ "$(git log -1 --format=%s HEAD)" = "$CHAIN_MSG" ] \
      && git merge-base --is-ancestor HEAD origin/main; then
    note "run/chain's publish commit $(git rev-parse --short HEAD) is already merged into origin/main: stage chain is done"
    git status --short -- workspace logs experiments report | head -5
    return 0
  fi

  run bash scripts/chain.sh status | tee "$TMP/status.txt"
  local counts done_n total
  counts="$(sed -n 's/^done: *\([0-9][0-9]*\) *\/ *\([0-9][0-9]*\)$/\1 \2/p' "$TMP/status.txt")"
  read -r done_n total <<<"${counts:-0 0}"
  if [ "$total" -eq 0 ] || [ "$done_n" -eq 0 ] || grep -q '^next: ' "$TMP/status.txt"; then
    die "chain queue not complete ($done_n / $total in $STATE/queue, $(grep '^next: ' "$TMP/status.txt" || echo 'no item done')): refused"
  fi
  [ "$done_n" -eq "$total" ] || note "every queue item is done ($done_n lines in done.txt, $total in queue.txt: repeated queue lines run once)"
  if grep -q '^PAUSED' "$TMP/status.txt" || [ -e "$STATE/queue/PAUSE" ]; then
    die "$STATE/queue/PAUSE is set (a failed or paused item): refused"
  fi
  local busy; busy="$(chain_procs "$CHAIN" | paste -sd ' ')"
  [ -z "$busy" ] || die "chain processes still work in $CHAIN (pids $busy): refused"
  local other; other="$(git status --porcelain --untracked-files=no | grep -vE '^.. (workspace|logs|experiments|report)/' || true)"
  [ -z "$other" ] || note "WARNING: tracked changes outside workspace/ logs/ experiments/ report/ stay uncommitted:"$'\n'"$other"

  # card source, checked before anything is written
  if [ "$CARDS_FROM_MAIN" = 1 ]; then
    [ -f "$MAIN/gotrain/model_card.py" ] || die "no $MAIN/gotrain/model_card.py for the model cards (use --main-dir, or --cards-from-chain)"
    git -C "$MAIN" diff --quiet HEAD -- gotrain/model_card.py \
      || die "$MAIN/gotrain/model_card.py has uncommitted changes: commit them first (cards are published from committed code)"
    note "model cards: $MAIN/gotrain/model_card.py at $(git -C "$MAIN" log -1 --format='%h %s' -- gotrain/model_card.py) ($(git -C "$MAIN" rev-parse --short HEAD) checked out)"
    local dirty; dirty="$(git -C "$MAIN" status --porcelain --untracked-files=no -- gotrain/ | paste -sd ' ')"
    [ -z "$dirty" ] || note "WARNING: uncommitted changes in $MAIN/gotrain/ (imported by the card code): $dirty"
  else
    note "model cards: run/chain's frozen gotrain/model_card.py (--cards-from-chain: lesson z not sign-corrected, seed sigma unrounded)"
  fi

  local main_ws="$CHAIN/workspace" off_ws="$CHAIN/experiments/playbook-ab/$AB_TASK/off/workspace"
  local on_id="${AB_TASK}__main__chain" off_id="${AB_TASK}__off__chain"

  say "a. verify every session"
  local main_ids off_ids sid
  mapfile -t main_ids < <(session_ids "$main_ws" main)
  [ "${#main_ids[@]}" -gt 0 ] && [ -n "${main_ids[0]}" ] || die "no main-arm session in $main_ws/sessions.jsonl"
  for sid in "${main_ids[@]}"; do
    pyrun GOTRAIN_WORKSPACE="$main_ws" python -m gotrain.verify --session "$sid" || fail "verify --session $sid failed"
  done
  mapfile -t off_ids < <(session_ids "$off_ws" off)
  for sid in "${off_ids[@]}"; do
    [ -n "$sid" ] || continue
    pyrun GOTRAIN_WORKSPACE="$off_ws" python -m gotrain.verify --session "$sid" || fail "verify --session $sid (OFF arm) failed"
  done

  say "b. registry sync + model cards"
  pyrun GOTRAIN_WORKSPACE="$main_ws" python -m gotrain.registry sync > "$TMP/sync.json" || fail "registry sync failed"
  cat "$TMP/sync.json"
  if grep -q '"skipped"' "$TMP/sync.json"; then fail "registry sync skipped a vault entry (see above)"; fi
  local tasks task
  mapfile -t tasks < <(vault_tasks "$main_ws")
  for task in "${tasks[@]}"; do
    [ -n "$task" ] || continue
    if [ "$CARDS_FROM_MAIN" = 1 ]; then
      (cd "$MAIN" && pyrun GOTRAIN_WORKSPACE="$main_ws" python -m gotrain.model_card "$task") || fail "model card $task failed"
    else
      pyrun GOTRAIN_WORKSPACE="$main_ws" python -m gotrain.model_card "$task" || fail "model card $task failed"
    fi
  done

  say "c. with/without-playbook analysis ($AB_TASK)"
  if ! contains "$on_id" "${main_ids[@]}" || ! contains "$off_id" "${off_ids[@]}"; then
    fail "A/B not available: ON session $on_id $(contains "$on_id" "${main_ids[@]}" && echo found || echo MISSING) in workspace, OFF session $off_id $(contains "$off_id" "${off_ids[@]}" && echo found || echo MISSING) in ${off_ws#"$CHAIN"/}; analyze.py not run"
  else
    pyrun python experiments/playbook-ab/analyze.py --on "$on_id@workspace" \
      --off "$off_id@experiments/playbook-ab/$AB_TASK/off/workspace" --out "experiments/playbook-ab/$AB_TASK" \
      || fail "analyze.py failed (exit $?): no A/B result"
    [ -s "experiments/playbook-ab/$AB_TASK/metrics.json" ] && [ -s "experiments/playbook-ab/$AB_TASK/curves.png" ] \
      || fail "analyze.py wrote no metrics.json / curves.png in experiments/playbook-ab/$AB_TASK"
  fi

  say "d. report"
  pyrun GOTRAIN_WORKSPACE="$main_ws" python -m gotrain.report --out report/ || fail "report failed"

  say "e. verify --full"
  pyrun GOTRAIN_WORKSPACE="$main_ws" python -m gotrain.verify --full || fail "verify --full failed on run/chain"

  say "f. commit + push run/chain"
  if [ ! -f "$STATE/tested.jsonl" ]; then
    fail "$STATE/tested.jsonl not found"
  elif ! cmp -s "$STATE/tested.jsonl" workspace/test_registry.jsonl; then
    run cp "$STATE/tested.jsonl" workspace/test_registry.jsonl
    note "the test-registry copy changed: verify --full again"
    pyrun GOTRAIN_WORKSPACE="$main_ws" python -m gotrain.verify --full || fail "verify --full failed with the new test-registry copy"
  else
    note "workspace/test_registry.jsonl already equals $STATE/tested.jsonl"
  fi
  local paths=() p
  for p in workspace logs experiments report; do [ -e "$p" ] && paths+=("$p"); done
  if [ "$DRY" = 1 ]; then
    show git add -A -- "${paths[@]}"; note "[dry-run] not staged; it would stage:"
    git add -n -A -- "${paths[@]}" > "$TMP/add.txt"
    sed 's/^/    /' "$TMP/add.txt" | head -40
    [ "$(wc -l < "$TMP/add.txt")" -le 40 ] || note "    ... $(wc -l < "$TMP/add.txt") paths in all"
    sed -n "s/^add '\(.*\)'$/\1/p" "$TMP/add.txt" | while IFS= read -r f; do
      if [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt "$MAX_BYTES" ]; then note "    would unstage large file $f"; fi
    done
  else
    run git add -A -- "${paths[@]}"
    git diff --cached --name-only -z | while IFS= read -r -d '' f; do
      if [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt "$MAX_BYTES" ]; then run git reset -q -- "$f"; note "unstaged large file $f"; fi
    done
  fi
  if [ "$DRY" = 0 ] && git diff --cached --quiet; then
    note "nothing new to commit on run/chain"
  else
    gitw commit -q -m "$CHAIN_MSG"
  fi
  gitw push origin run/chain
  if [ "$DRY" = 0 ] && [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/run/chain)" ]; then
    die "origin/run/chain is not HEAD after the push"
  fi
}

# ----------------------------------------------------------------------------------------------------- stage main
# README.md outside its results block must equal HEAD's
readme_only_block_changed() {
  printf '+ [%s] python (compare %s outside the results markers with HEAD:README.md)\n' "$PWD" "$1"
  git show HEAD:README.md > "$TMP/README.head.md"
  nice -n 19 python - "$TMP/README.head.md" "$1" <<'PY'
import sys
B = "<!-- results:begin (written by python -m gotrain.report --readme; do not edit by hand) -->"
E = "<!-- results:end -->"
def outside(path):
    t = open(path).read()
    if B not in t or E not in t:
        sys.exit(f"{path}: no results markers")
    head, rest = t.split(B, 1)
    return head, rest.split(E, 1)[1]
sys.exit(0 if outside(sys.argv[1]) == outside(sys.argv[2]) else f"{sys.argv[2]} changed outside the results block")
PY
}

stage_main() {
  say "STAGE main: $MAIN"
  cd "$MAIN"
  local branch; branch="$(git rev-parse --abbrev-ref HEAD)"
  [ "$branch" = main ] || die "$MAIN is on branch $branch, not main: refused"
  local dirty; dirty="$(git status --porcelain --untracked-files=no | grep -vE '^.. (README\.md|report/|docs/DELIVERABLES\.md)' || true)"
  if [ -n "$dirty" ]; then fail "uncommitted changes to tracked files in $MAIN (commit them first):"$'\n'"$dirty"; fi

  say "g. fetch + merge origin/run/chain"
  run git fetch origin
  local pub; pub="$(git log -1 --format=%h --fixed-strings --grep="$CHAIN_MSG" origin/run/chain)"
  [ -n "$pub" ] || fail "origin/run/chain has no '$CHAIN_MSG' commit: run --stage chain first"
  if git rev-parse -q --verify refs/heads/run/chain >/dev/null \
      && ! git merge-base --is-ancestor run/chain origin/run/chain; then
    fail "local run/chain $(git rev-parse --short run/chain) has commits that origin/run/chain lacks (push it first)"
  fi
  if git merge-base --is-ancestor origin/run/chain HEAD; then
    note "origin/run/chain ($(git rev-parse --short origin/run/chain)) is already merged into main: merge skipped"
  else
    show git merge-tree --write-tree --name-only HEAD origin/run/chain
    git merge-tree --write-tree --name-only HEAD origin/run/chain > "$TMP/merge-tree.txt" \
      || fail "merging origin/run/chain would conflict:"$'\n'"$(sed -n '2,/^$/p' "$TMP/merge-tree.txt")"
    # untracked files that the merge would have to overwrite
    comm -12 <(git ls-files --others --exclude-standard | sort) \
             <(git diff --name-only HEAD...origin/run/chain | sort) > "$TMP/collide.txt"
    [ ! -s "$TMP/collide.txt" ] || fail "untracked files in $MAIN would be overwritten by the merge (move them away):"$'\n'"$(head "$TMP/collide.txt")"
    gitw merge --no-ff --no-stat origin/run/chain -m "$MERGE_MSG"
    [ "$DRY" = 1 ] || run git diff --shortstat HEAD^1 HEAD
  fi

  say "h. report + README results block"
  local out=report/ readme=README.md
  if [ "$DRY" = 1 ]; then  # keep the worktree clean for the real run
    out="$TMP/report/" readme="$TMP/README.md"
    run cp README.md "$readme"
    note "[dry-run] report and filled README go to $TMP"
  fi
  pyrun python -m gotrain.report --out "$out" --readme "$readme" || fail "report --readme failed"
  readme_only_block_changed "$readme" || fail "README.md changed outside the results block"

  say "i. verify --full (after the merge)"
  local chain_head; chain_head="$(git -C "$CHAIN" rev-parse HEAD)"
  [ "$chain_head" = "$(git rev-parse origin/run/chain)" ] \
    || fail "$CHAIN HEAD ${chain_head:0:7} is not origin/run/chain $(git rev-parse --short origin/run/chain)"
  [ -z "$(git -C "$CHAIN" status --porcelain -- workspace)" ] \
    || fail "$CHAIN/workspace has uncommitted changes: it is not the published workspace"
  run git diff --quiet origin/run/chain HEAD -- workspace/ \
    || fail "main's workspace/ differs from origin/run/chain's$([ "$DRY" = 1 ] && echo ' (expected in a dry run: the merge was skipped)')"
  pyrun GOTRAIN_WORKSPACE="$CHAIN/workspace" python -m gotrain.verify --full || fail "verify --full failed after the merge"

  say "j. commit, tag, push"
  note "REMINDER: docs/DELIVERABLES.md statuses are updated by hand (rows 1, 3, 8-12). Uncommitted edits to it are"
  note "          committed with the README now; later edits need their own commit (the tag will not include them)."
  if [ "$DRY" = 1 ]; then
    show git add -- README.md report docs/DELIVERABLES.md; note "[dry-run] not staged"
  else
    run git add -A -- README.md report docs/DELIVERABLES.md
  fi
  local tag_at=""
  if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then tag_at="$(git rev-parse "$TAG^{commit}")"; fi
  if [ "$DRY" = 0 ] && git diff --cached --quiet; then
    note "nothing new to commit on main"
  elif [ -n "$tag_at" ] && [ "$DO_TAG" = 1 ]; then  # checked before committing: the tag would miss the new commit
    fail "tag $TAG already exists (at ${tag_at:0:7}); not committing new changes it would miss: commit and move the tag by hand if intended"
  else
    gitw commit -q -m "$MAIN_MSG"
  fi
  if [ "$DO_TAG" = 0 ]; then
    gitw push origin main
    note "no tag set (default). After the docs/MCP merges, run: scripts/publish.sh --stage main --tag"
    return 0
  fi
  if [ -z "$tag_at" ]; then
    gitw tag -a "$TAG" -m "$TAG_MSG"
  elif [ "$tag_at" = "$(git rev-parse HEAD)" ]; then
    note "tag $TAG already points at HEAD: kept"
  else
    fail "tag $TAG exists at ${tag_at:0:7}, not HEAD: move it by hand if intended"
  fi
  gitw push origin main "$TAG"
}

[ "$DRY" = 1 ] && note "DRY RUN: no git commit / push / tag / merge; scratch folder $TMP"
case "$STAGE" in
  chain) stage_chain ;;
  main) stage_main ;;
  all) stage_chain; stage_main ;;
esac

if [ "${#FAILS[@]}" -gt 0 ]; then
  say "DRY RUN: ${#FAILS[@]} failure(s); the real run would stop at the first"
  printf '  - %s\n' "${FAILS[@]}"
  exit 1
fi
say "publish ($STAGE) finished$([ "$DRY" = 1 ] && echo ' (dry run)')"
