#!/usr/bin/env bash
# GoTrain GPU chain (CREATIVE_PLAN.md Step 11): runs the queue unattended, resumable, one item at a time.
#
#   scripts/chain.sh init <B>     # write the default queue for budget B into $GOTRAIN_STATE_DIR/queue/
#   nohup scripts/chain.sh run > /dev/null 2>&1 &     # work the queue (logs to logs/chain.log)
#   scripts/chain.sh status       # done / next / paused
#   touch $GOTRAIN_STATE_DIR/queue/PAUSE              # stop after the current item (rm to resume)
#
# Queue items (one per line in queue.txt; finished ones are appended to done.txt and skipped on resume):
#   build_cache <task>                         train/val cache v2, outside every pot (under the GPU lock)
#   session <task> <arm> <phases> [args...]    python -m gotrain.run with session id <task>__<arm>__chain
#   verify <task> <arm>                        python -m gotrain.verify --session <task>__<arm>__chain (if built)
#   snapshot                                   freeze workspace/playbook.jsonl -> experiments/playbook-ab/playbook-snapshot.jsonl
#   commit <message>                           commit + push workspace/ logs/ experiments/ (files > 5 MB unstaged)
#   wait_deploy                                set PAUSE: the operator merges the Level 2 deploy, then resumes
# A failing item sets PAUSE and stops the chain (harness failure). A failed push never blocks the queue.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
STATE="${GOTRAIN_STATE_DIR:-$HOME/.gotrain-state}"
Q="$STATE/queue"
LOG="$REPO/logs/chain.log"
mkdir -p "$Q" "$REPO/logs"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

off_ws() { echo "experiments/playbook-ab/$1/off/workspace"; }

commit() {
  local msg="$1"
  git add -A workspace logs experiments 2>/dev/null
  # never commit large files: unstage anything over 5 MB
  git diff --cached --name-only -z | while IFS= read -r -d '' f; do
    if [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 5242880 ]; then git reset -q -- "$f"; log "unstaged large file $f"; fi
  done
  if git diff --cached --quiet; then log "commit: nothing to commit ($msg)"; return 0; fi
  git commit -q -m "$msg" && log "committed: $msg"
  git push -q origin HEAD 2>>"$LOG" || log "push failed (queue continues): $msg"
  return 0
}

run_item() {
  local item="$1"; set -- $item
  local kind="$1"; shift
  case "$kind" in
    build_cache) python -m gotrain.engine.data build "$1" ;;
    session)
      local task="$1" arm="$2" phases="$3"; shift 3
      if [ "$arm" = off ]; then
        mkdir -p "$(off_ws "$task")"
        GOTRAIN_WORKSPACE="$(off_ws "$task")" python -m gotrain.run --task "$task" --arm off --phases "$phases" \
          --session-id "${task}__off__chain" "$@"
      else
        python -m gotrain.run --task "$task" --arm "$arm" --phases "$phases" --session-id "${task}__${arm}__chain" "$@"
      fi ;;
    verify)
      local task="$1" arm="$2" ws=""
      [ "$arm" = off ] && ws="$(off_ws "$task")"
      if python -c "import gotrain.verify" 2>/dev/null; then
        GOTRAIN_WORKSPACE="${ws:-}" python -m gotrain.verify --session "${task}__${arm}__chain" \
          --checks transcript,ledger,noise,pot
      else
        log "verify: gotrain.verify not built; skipped"
      fi ;;
    snapshot)
      mkdir -p experiments/playbook-ab
      cp workspace/playbook.jsonl experiments/playbook-ab/playbook-snapshot.jsonl 2>/dev/null \
        || : > experiments/playbook-ab/playbook-snapshot.jsonl
      sha256sum experiments/playbook-ab/playbook-snapshot.jsonl | tee -a "$LOG" ;;
    commit) commit "chain: $*" ;;
    wait_deploy) touch "$Q/PAUSE"; log "PAUSE set: merge the Level 2 deploy into run/chain, then rm $Q/PAUSE and rerun"; return 0 ;;
    *) log "unknown queue item: $item"; return 2 ;;
  esac
}

cmd="${1:-status}"
case "$cmd" in
  init)
    B="${2:?usage: chain.sh init <B>}"
    [ -s "$Q/queue.txt" ] && { echo "queue exists: $Q/queue.txt (remove it first)"; exit 1; }
    {
      for t in m-eurosat m-brick-kiln m-pv4ger-seg m-cashew-plant; do
        echo "build_cache $t"
        echo "session $t main level1,reflect --budget $B"
        echo "verify $t main"
        echo "commit $t level1 + reflect"
      done
      echo "snapshot"
      echo "commit playbook snapshot after #4"
      echo "build_cache m-SA-crop-type"
      echo "session m-SA-crop-type main level1,reflect --budget $B --playbook-snapshot experiments/playbook-ab/playbook-snapshot.jsonl"
      echo "verify m-SA-crop-type main"
      echo "commit m-SA-crop-type ON level1"
      echo "session m-SA-crop-type off level1 --budget $B"
      echo "verify m-SA-crop-type off"
      echo "commit m-SA-crop-type OFF level1"
      echo "wait_deploy"
      for t in m-eurosat m-brick-kiln m-pv4ger-seg m-cashew-plant m-SA-crop-type; do
        echo "session $t main level2,finalize"
        echo "verify $t main"
        echo "commit $t level2 + finalize"
      done
    } > "$Q/queue.txt"
    : > "$Q/done.txt"
    echo "queue written: $Q/queue.txt ($(wc -l < "$Q/queue.txt") items, B=$B)" ;;
  status)
    touch "$Q/done.txt"
    echo "done: $(wc -l < "$Q/done.txt") / $(wc -l < "$Q/queue.txt" 2>/dev/null || echo 0)"
    grep -vxFf "$Q/done.txt" "$Q/queue.txt" 2>/dev/null | head -1 | sed 's/^/next: /'
    if [ -e "$Q/PAUSE" ]; then echo "PAUSED ($Q/PAUSE)"; fi ;;
  run)
    touch "$Q/done.txt"
    log "chain start (pid $$)"
    while IFS= read -r item; do
      [ -z "$item" ] && continue
      grep -qxF "$item" "$Q/done.txt" && continue
      if [ -e "$Q/PAUSE" ]; then log "PAUSE marker present: stopping before '$item'"; exit 0; fi
      log "start: $item"
      if run_item "$item" >> "$LOG" 2>&1; then
        echo "$item" >> "$Q/done.txt"; log "done: $item"
      else
        touch "$Q/PAUSE"; log "FAILED: $item -> PAUSE set (harness failure; fix, rm PAUSE, rerun)"; exit 1
      fi
    done < "$Q/queue.txt"
    log "chain: queue complete" ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
