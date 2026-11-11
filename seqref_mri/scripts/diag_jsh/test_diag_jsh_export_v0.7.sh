# SEQREF-MRI-DIAGJSHEXPTEST v0.7 -- throwaway-repository refusal tests for DIAG-JSH export helper v0.7 (hardcoded sandbox paths; evidence only, not for host execution)
# LIFETIME: KEEP
set -u
# Changelog (v0.7, 2026-11-11): for helper v0.7 (G v0.3).
# Changelog (v0.6, 2026-11-11): for helper v0.6 (G v0.2, resources v0.2).
# Changelog (v0.5, 2026-11-11): fixtures include the stage G files
#   (g_run, f2_resources, g_cmd) for helper v0.5.
# Changelog (v0.4, 2026-11-10): fixtures copied from the v0.6 staging folder
#   (F1 prepare v0.6) for helper v0.4; D1 gives the old v0.2 helper the
#   v0.6 F1 hash (declared test double) so it still reproduces its finding.
# Changelog (v0.3, 2026-11-10): R12-R14 failed sed / sort in the source-scan
#   comparison refused (stub commands on PATH); D1 reproduces the v0.2
#   finding (failed producer accepted). v0.2.1: R1 / R1b split.
HERE=$(cd "$(dirname "$0")" && pwd)
T=/tmp/claude-0/-home-claude/aae3b5ac-3a04-5a76-befe-d0f05278fca8/scratchpad/exptest; rm -rf $T; mkdir -p $T; cd $T
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t
git init -q --bare o.git; git init -q -b seqref-v05lt R; cd R
mkdir -p seqref_mri/configs seqref_mri/scripts; echo x > seqref_mri/scripts/a.py; echo c > seqref_mri/configs/c.json
printf '*.sh\n.venv_mri/\nresults/\n' > .gitignore; git add .; git commit -qm pin; P=$(git rev-parse HEAD)
git remote add origin $T/o.git; git push -q origin seqref-v05lt 2>/dev/null
echo "v06lt_evidence/" >> .git/info/exclude; mkdir v06lt_evidence
mkdir -p seqref_mri/.venv_mri/lib; echo v > seqref_mri/.venv_mri/lib/site.py; mkdir -p seqref_mri/results; echo r > seqref_mri/results/r.py
git switch -qc ch2-diag; mkdir -p seqref_mri/scripts/diag_jsh
cp /home/claude/v06lt_dev/diag_jsh/next/diag_jsh_{core,adapter,f1_prepare,i3,g_run,f2_resources}.py /home/claude/v06lt_dev/diag_jsh/next/diag_jsh_{f1,g}_cmd.sh seqref_mri/scripts/diag_jsh/
sed -e "s#^V05=.*#V05=$T/R#" -e "s#^PIN=.*#PIN=$P#" "$HERE/diag_jsh_export.sh" > seqref_mri/scripts/diag_jsh/diag_jsh_export.sh
git add seqref_mri; git add -f seqref_mri/scripts/diag_jsh/*.sh; git commit -qm dev; C=$(git rev-parse HEAD)
mkdir -p $T/stub
cat > $T/stub/sed <<'S'
#!/bin/bash
[[ "${FAILCMD:-}" == sed && "$*" == *'s/^F //p'* ]] && exit 4
exec /usr/bin/sed "$@"
S
cat > $T/stub/sort <<'S'
#!/bin/bash
[[ "${FAILCMD:-}" == sort_scan && "$*" == *scan.raw* ]] && exit 4
[[ "${FAILCMD:-}" == sort_trk && "$*" == *trk.raw* ]] && exit 4
exec /usr/bin/sort "$@"
S
chmod +x $T/stub/sed $T/stub/sort
N=0; F=0
run() { local want=$1 lab=$2; shift 2
  out=$(bash <(git -C $T/R show ch2-diag:seqref_mri/scripts/diag_jsh/diag_jsh_export.sh) "$@" 2>&1); rc=$?
  N=$((N+1)); if [[ $rc == "${want%%:*}" && ( "$want" != *:* || "$out" == *"${want#*:}"* ) ]]; then echo "ok   $lab (rc=$rc)"; else F=$((F+1)); echo "FAIL $lab (rc=$rc)"; echo "$out" | tail -3; fi; }
br() { git -C $T/R branch --show-current; }
run "2:fetch origin ch2-diag failed" "R1 branch not on origin refused" $C
git push -q origin ch2-diag 2>/dev/null
echo n > n.txt; git add n.txt; git commit -qm local; C2=$(git rev-parse HEAD)
run "2:is not pushed" "R1b commit not pushed refused" $C2
git reset -q --hard $C
bash -c 'exec -a seqref_mri.scripts.v05lt_fake sleep 30' & J=$!; sleep 0.3
run "2:job is running" "R2 running V05LT job refused before switching" $C; kill $J; wait $J 2>/dev/null
[[ $(br) == ch2-diag ]] && echo "ok   R2b branch unchanged after job refusal" || { F=$((F+1)); echo "FAIL R2b"; }; N=$((N+1))
echo y >> seqref_mri/scripts/a.py; run "2:not clean" "R3 uncommitted change refused" $C; git checkout -q seqref_mri/scripts/a.py
run "2:not found" "R4 unknown commit refused" 0000000000000000000000000000000000000000
mkdir -p seqref_mri/data; echo d > seqref_mri/data/ignored_by_scan.py; echo "seqref_mri/data/" >> .git/info/exclude
echo s > seqref_mri/scripts/stray.py; echo "seqref_mri/scripts/stray.py" >> .git/info/exclude
run "2:not tracked at the pin" "R5 ignored extra .py refused (venv/results/data skipped)" $C
[[ -z $(git -C $T/R status --porcelain) ]]; rm seqref_mri/scripts/stray.py
git switch -q ch2-diag
echo k > seqref_mri/configs/new.json; echo "seqref_mri/configs/new.json" >> .git/info/exclude
run "2:not tracked at the pin" "R6 extra untracked config file refused" $C; rm seqref_mri/configs/new.json; git switch -q ch2-diag
ln -s /etc/hostname seqref_mri/scripts/lnk.txt; echo "seqref_mri/scripts/lnk.txt" >> .git/info/exclude
run "2:symlink" "R7 symlink in source scan refused" $C; rm seqref_mri/scripts/lnk.txt; git switch -q ch2-diag
mkdir $T/elsewhere; ln -s $T/elsewhere v06lt_evidence/DIAG_JSH
run "2:not a plain directory" "R8 DIAG_JSH symlink refused before creation" $C
[[ -z $(ls -A $T/elsewhere) ]] && echo "ok   R8b nothing created through the symlink" || { F=$((F+1)); echo "FAIL R8b"; }; N=$((N+1))
rm v06lt_evidence/DIAG_JSH; git switch -q ch2-diag
FAILCMD=sed PATH=$T/stub:$PATH run "2:sed over the source scan failed" "R12 failed sed in the scan comparison refused" $C; git switch -q ch2-diag
FAILCMD=sort_scan PATH=$T/stub:$PATH run "2:sort of the source scan failed" "R13 failed sort of the scan list refused" $C; git switch -q ch2-diag
FAILCMD=sort_trk PATH=$T/stub:$PATH run "2:sort of the tracked list failed" "R14 failed sort of the tracked list refused" $C; git switch -q ch2-diag
if [[ -f "$HERE/diag_jsh_export_v0.2.sh" ]]; then
  sed -e "s#^V05=.*#V05=$T/R#" -e "s#^PIN=.*#PIN=$P#" \
      -e "s#3472714752948d5fbf63d7a96ffa64db2b787d6f9a40b6720e944537242fd739#dc74b0693b1eb0c0ecaa88fd813e22b925c6d99025d834c20c01cbcd71995a1f#" \
      "$HERE/diag_jsh_export_v0.2.sh" > $T/v02.sh   # test double: v0.2 given the v0.6 F1 hash
  echo s > seqref_mri/scripts/stray2.py; echo "seqref_mri/scripts/stray2.py" >> .git/info/exclude
  out=$(FAILCMD=sed PATH=$T/stub:$PATH bash $T/v02.sh $C 2>&1); rc=$?; N=$((N+1))
  if [[ $rc == 0 ]]; then echo "ok   D1 v0.2 accepts a failed sed as an empty comparison despite an extra .py (reproduces the finding)"; else F=$((F+1)); echo "FAIL D1 (rc=$rc)"; fi
  rm seqref_mri/scripts/stray2.py; rm -rf v06lt_evidence/DIAG_JSH; git switch -q ch2-diag
fi
run "0:RESULT DIAGJSH_EXPORT PASS" "P1 pass (venv/results/data .py ignored, as the pinned scan)" $C
[[ $(br) == seqref-v05lt ]] && echo "ok   P1b on seqref-v05lt after pass" || { F=$((F+1)); echo "FAIL P1b"; }; N=$((N+1))
run "0:already identical" "P2 identical repeat leaves the deployed files unchanged" $C
K=v06lt_evidence/DIAG_JSH/code
chmod u+w $K/diag_jsh_i3.py; mv $K/diag_jsh_i3.py $T/i3.py; ln -s $T/i3.py $K/diag_jsh_i3.py
run "2:not a regular file" "R9 identical-content symlink in code/ refused" $C
rm $K/diag_jsh_i3.py; mv $T/i3.py $K/diag_jsh_i3.py
echo x > $K/extra.txt; run "2:holds other files" "R10 extra file in code/ refused" $C; rm $K/extra.txt
echo '#' >> $K/diag_jsh_i3.py; run "2:differs" "R11 differing file in code/ refused" $C
ls -A v06lt_evidence | grep -q diagjsh_export && { F=$((F+1)); echo "FAIL temp folder left"; } || echo "ok   T temp folders removed"; N=$((N+1))
echo "RESULT $((N-F))/$N checks passed"
cd /; rm -rf $T
