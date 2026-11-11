# SEQREF-MRI-DIAGJSHGBLK v0.6 -- mock-tree tests of the stage G pre-check, launch and move-aside blocks (sandbox paths; evidence only)
# LIFETIME: KEEP
# Paths and the pin are rewritten to a throwaway mock tree. DECLARED TEST
# DOUBLES: systemctl (timer times), the V05LT python (find_spec only), the G
# command (records its arguments), and a pass-path copy of the pre-check
# with the free-space minimum set to 1 B (the sandbox has < 64 GiB free).
# Changelog (v0.6, 2026-11-11): pre-check v6 / launch v6 (G v0.6 hash);
#   move-aside v2 (code/ = G v0.5 set -> code_Gv05_run1): M1-M5.
# Changelog (v0.5, 2026-11-11): pre-check v5 / launch v5 (G v0.5 hash).
# Changelog (v0.4, 2026-11-11): pre-check v4 / launch v4 (G v0.4 hash).
# Changelog (v0.3, 2026-11-11): pre-check v3 / launch v3 (G v0.3 hash).
# Changelog (v0.2, 2026-11-11; review): apt services already running refused
#   (C12, C13); pre-check v2 (code hashes for G v0.2 / resources v0.2).
# Changelog (v0.1, 2026-11-11): first version.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=/tmp/claude-0/-home-claude/aae3b5ac-3a04-5a76-befe-d0f05278fca8/scratchpad/gblk; rm -rf $T; mkdir -p $T; cd $T
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t
R=$T/CSMFII; git init -q -b seqref-v05lt $R; cd $R
mkdir -p seqref_mri/scripts seqref_mri/results/v05lt/launch; echo x > seqref_mri/scripts/a.py
printf '.venv_mri/\n' > .gitignore; git add .; git commit -qm pin; P=$(git rev-parse HEAD)
echo "v06lt_evidence/" >> .git/info/exclude
DJ=$R/v06lt_evidence/DIAG_JSH; mkdir -p $DJ/code $DJ/out/F1_20261110T190054Z
for f in diag_jsh_{core,adapter,f1_prepare,i3,g_run,f2_resources}.py diag_jsh_g_cmd.sh; do cp /home/claude/v06lt_dev/diag_jsh/next/$f $DJ/code/; done
cp /mnt/user-data/attachments/aae3b5ac-3a04-5a76-befe-d0f05278fca8/d15aca7f-diag_jsh_manifest.json $DJ/out/F1_20261110T190054Z/diag_jsh_manifest.json
cp /mnt/user-data/attachments/aae3b5ac-3a04-5a76-befe-d0f05278fca8/7a09d30b-diag_jsh_manifest.json.sha256 $DJ/out/F1_20261110T190054Z/diag_jsh_manifest.json.sha256
mkdir -p seqref_mri/.venv_mri/bin
printf '#!/bin/bash\n[[ -n "${NOPLOT:-}" ]] && exit 1\nexec python3 "$@"\n' > seqref_mri/.venv_mri/bin/python; chmod +x seqref_mri/.venv_mri/bin/python
mkdir -p $T/stub
cat > $T/stub/systemctl <<'S'
#!/bin/bash
if [[ "$1" == is-active ]]; then
  if [[ "$2" == apt-daily-upgrade.service && -n "${SVC:-}" ]]; then echo "$SVC"; exit 0; fi
  echo inactive; exit 3
fi
case "${TIMER:-far}" in
  far) date -d '+2 days' '+%a %Y-%m-%d %H:%M:%S %Z' ;;
  soon) date -d '+1 hour' '+%a %Y-%m-%d %H:%M:%S %Z' ;;
  empty) echo "" ;;
esac
S
chmod +x $T/stub/systemctl
mk() { sed -e "s#/home/benjamin/CSMFII#$R#g" -e "s#7455da8027f33a6cca4dbf72f04e72420e938f45#$P#" "$HERE/$1"; }
mk g_precheck__v6.txt > $T/pre_real.sh
mk g_precheck__v6.txt | sed 's/68754079744/1/g' > $T/pre_pass.sh
N=0; F=0
chk() { local want=$1 lab=$2 script=$3; shift 3
  out=$(env PATH=$T/stub:$PATH "$@" bash $script 2>&1); N=$((N+1))
  if [[ "$out" == *"$want"* ]]; then echo "ok   $lab"; else F=$((F+1)); echo "FAIL $lab"; echo "$out" | tail -4; fi; }
chk "precheck_exit=0" "C1 pass (all checks; timers far)" $T/pre_pass.sh
o=$(env PATH=$T/stub:$PATH bash $T/pre_pass.sh 2>&1); N=$((N+1))
[[ $(grep -c "TIMER OK" <<< "$o") == 2 && $(grep -c "SERVICE OK" <<< "$o") == 2 && "$o" == *"PLOTTING LIBRARIES present"* && "$o" == *"diag_jsh_manifest.json: OK"* && $(grep -c ": OK" <<< "$o") == 8 ]] && echo "ok   C1b seven code files + manifest OK, plotting found, two timers" || { F=$((F+1)); echo "FAIL C1b"; echo "$o"; }
chk "STOP: free" "C2 real threshold 68,754,079,744 B refuses the small sandbox disk" $T/pre_real.sh
chk "STOP: cap + 15 min ends" "C3 timer within cap + 15 min refused" $T/pre_pass.sh TIMER=soon
chk "STOP: apt-daily.timer has no next elapse time" "C4 timer without next elapse refused" $T/pre_pass.sh TIMER=empty
chk "STOP: plotting libraries missing" "C5 plotting libraries missing refused" $T/pre_pass.sh NOPLOT=1
git -C $R switch -qc other; chk "STOP: branch other" "C6 wrong branch refused" $T/pre_pass.sh; git -C $R switch -q seqref-v05lt
echo y >> $R/seqref_mri/scripts/a.py; chk "STOP: not clean" "C7 dirty tree refused" $T/pre_pass.sh; git -C $R checkout -q seqref_mri/scripts/a.py
chmod u+w $DJ/code/diag_jsh_g_run.py; echo '#' >> $DJ/code/diag_jsh_g_run.py; chk "FAILED" "C8 changed code file refused" $T/pre_pass.sh
cp /home/claude/v06lt_dev/diag_jsh/next/diag_jsh_g_run.py $DJ/code/
echo ' ' >> $DJ/out/F1_20261110T190054Z/diag_jsh_manifest.json; chk "FAILED" "C9 changed manifest refused" $T/pre_pass.sh
cp /mnt/user-data/attachments/aae3b5ac-3a04-5a76-befe-d0f05278fca8/d15aca7f-diag_jsh_manifest.json $DJ/out/F1_20261110T190054Z/diag_jsh_manifest.json
bash -c 'exec -a diag_jsh_g_run.py sleep 30' & J=$!; sleep 0.3
chk "STOP: pgrep rc 0" "C10 running DIAG-JSH job refused" $T/pre_pass.sh; kill $J; wait $J 2>/dev/null
chk "STOP: apt-daily-upgrade.service is active" "C12 apt service already running refused" $T/pre_pass.sh SVC=active
chk "STOP: apt-daily-upgrade.service is activating" "C13 apt service starting refused" $T/pre_pass.sh SVC=activating
chk "precheck_exit=0" "C11 pass again after restoring everything" $T/pre_pass.sh
# launch: stub command records its arguments
cp $DJ/code/diag_jsh_g_cmd.sh $T/g_cmd_real.sh
printf '#!/bin/bash\nprintf "%%s\\n" "$@" > %s/args.txt\nexit 0\n' "$T" > $DJ/code/diag_jsh_g_cmd.sh
o=$(mk g_launch__v6.txt | bash 2>&1); N=$((N+1))
mapfile -t A < $T/args.txt
W=( "$R/seqref_mri/results/v05lt/launch/v05lt_launch_chain.json" 6f751f0387bb15dd3c70c82d4f247f6f2b51f62df8f66f1884f0f4381a6b478a 68719476736 16246 "$DJ/out/F1_20261110T190054Z/diag_jsh_manifest.json" d3e8176ef91dc1eb319b42f8510170eb6fe658a1ad1d99c561523fecd121d765 85410c77472cf876021a49bd72254ac98461227887178e24cdb6d0ce5295c0fb c566858189be4a3dee822ce4eef46b156986881a92cb6d4360b386ca785707c1 dc74b0693b1eb0c0ecaa88fd813e22b925c6d99025d834c20c01cbcd71995a1f 70f6b57511f62b3f178fa53ffcda938740b584e491da139895f17396e9aa1dca ee6580fe354310ce43e593d193f3460f219cd2a9520b059f3363a6272e499cd6 9af68faebe49d02d23aa9843023e0aaa4b4423611ebf1d9b7a27063775009b67 )
[[ "${A[*]}" == "${W[*]}" && ${#A[@]} == 12 && "$o" == *"g_exit=0"* ]] && echo "ok   L1 launch passes exactly the 12 registered arguments in order" || { F=$((F+1)); echo "FAIL L1"; printf '%s\n' "${A[@]}"; }
N=$((N+1)); grep -q 'Usage: bash diag_jsh_g_cmd.sh CHAIN CHAIN_SHA RESERVE_BYTES MAX_WALL_S \\' $T/g_cmd_real.sh && grep -q 'MANIFEST MANIFEST_SHA CORE_SHA ADAPTER_SHA F1_SHA I3_SHA G_SHA F2RES_SHA' $T/g_cmd_real.sh && echo "ok   L2 order matches the G command's usage line" || { F=$((F+1)); echo "FAIL L2"; }
N=$((N+1)); python3 -c "import sys; sys.path.insert(0,'/home/claude/v06lt_dev/diag_jsh/next'); import diag_jsh_f2_resources as R; sys.exit(0 if R.MAX_WALL_S == 16246 and R.G_BUDGET_BYTES == 32*2**20 else 1)" && grep -q 'G_BUDGET=$((32 \* 1024 \* 1024))' $T/g_cmd_real.sh && echo "ok   L3 cap 16,246 s and G budget 32 MiB agree with the resource record" || { F=$((F+1)); echo "FAIL L3"; }
# move-aside v2: separate mock tree whose code/ holds the 8 exported G v0.5 files
R2=$T/R2; M=$R2/v06lt_evidence/DIAG_JSH
mkm() { mkdir -p $M/code; for f in diag_jsh_{core,adapter,f1_prepare,i3,f2_resources}.py diag_jsh_{f1,g}_cmd.sh; do cp /home/claude/v06lt_dev/diag_jsh/next/$f $M/code/; done
  cp /home/claude/v06lt_dev/diag_jsh/prev_f2v05/diag_jsh_g_run_v0.5.py $M/code/diag_jsh_g_run.py; }
ma() { sed -e "s#/home/benjamin/CSMFII#$R2#g" "$HERE/g_moveaside__v2.txt" | bash 2>&1; }
mkm; mkdir -p $M/code_Gv05_run1; o=$(ma); N=$((N+1)); [[ "$o" == *"STOP: code_Gv05_run1 exists"* && "$o" == *"move_exit=1"* && -d $M/code ]] && echo "ok   M1 target exists refused" || { F=$((F+1)); echo "FAIL M1"; echo "$o"; }; rmdir $M/code_Gv05_run1
echo '#' >> $M/code/diag_jsh_g_run.py; o=$(ma); N=$((N+1)); [[ "$o" == *"FAILED"* && "$o" == *"move_exit=1"* && -d $M/code ]] && echo "ok   M2 changed file refused (e.g. G v0.6 deployed instead of v0.5)" || { F=$((F+1)); echo "FAIL M2"; echo "$o"; }; rm -rf $M/code; mkm
echo x > $M/code/extra.txt; o=$(ma); N=$((N+1)); [[ "$o" == *"expected 8"* && "$o" == *"move_exit=1"* ]] && echo "ok   M3 extra file refused" || { F=$((F+1)); echo "FAIL M3"; echo "$o"; }; rm $M/code/extra.txt
rm $M/code/diag_jsh_i3.py; o=$(ma); N=$((N+1)); [[ "$o" == *"expected 8"* && "$o" == *"move_exit=1"* ]] && echo "ok   M4 missing file refused" || { F=$((F+1)); echo "FAIL M4"; echo "$o"; }; rm -rf $M/code; mkm
o=$(ma); N=$((N+1)); [[ "$o" == *"MOVED code -> code_Gv05_run1"* && "$o" == *"move_exit=0"* && ! -e $M/code && $(ls $M/code_Gv05_run1 | wc -l) == 8 ]] && echo "ok   M5 pass: the 8 G v0.5 files moved, nothing else changed" || { F=$((F+1)); echo "FAIL M5"; echo "$o"; }
echo "RESULT $((N-F))/$N checks passed"
cd /; rm -rf $T
