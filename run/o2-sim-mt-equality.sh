#!/usr/bin/env bash
# Checks that multithreaded Geant4 in o2-sim gives the same output for 1 and N transport threads.
#
# Usage: o2-sim-mt-equality.sh [-n events] [-j threads] [-g generator] [--field kG] [--seed seed]
#                              [--seeding track|subevent] [--tolerance fraction] [--workdir dir] [module ...]
#
# Runs the same seeded workload with --isMT -j 1 and --isMT -j N and compares the kinematics and hit files
# event by event and branch by branch. A sequential run with the same seed is the reference for the number of
# hits of each detector, which must not be zero and must agree within the tolerance (default 0.2). Exits with
# 1 at the first difference. Without modules, the passive geometry with ITS is used.

nev=2
nthreads=2
gen=pythia8pp
field=-5
seed=4711
seeding=track
tolerance=0.2
workdir=
modules=()
while [ $# -gt 0 ]; do
  case $1 in
    -n) nev=$2; shift 2 ;;
    -j) nthreads=$2; shift 2 ;;
    -g) gen=$2; shift 2 ;;
    --field) field=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --seeding) seeding=$2; shift 2 ;;
    --tolerance) tolerance=$2; shift 2 ;;
    --workdir) workdir=$2; shift 2 ;;
    -h|--help) sed -n '2,10p' $0; exit 0 ;;
    *) modules+=("$1"); shift ;;
  esac
done
[ ${#modules[@]} -eq 0 ] && modules=(HALL MAG DIPO COMP PIPE ABSO SHIL ITS)
case $seeding in
  track) seedconf="SimCutParams.trackSeed=true" ;;
  subevent) seedconf="SimCutParams.trackSeed=false" ;;
  *) echo "unknown seeding '$seeding', use track or subevent"; exit 1 ;;
esac
o2sim=o2-sim
[ -x "$O2_ROOT/bin/o2-sim" ] && o2sim=$O2_ROOT/bin/o2-sim
[ -z "$workdir" ] && workdir=$(mktemp -d ${TMPDIR:-/tmp}/o2sim-mt-equality-XXXXXX)
mkdir -p $workdir && cd $workdir || exit 1

# content hash and number of entries per event and branch of the o2sim tree, independent of the basket layout
cat > hashtree.C <<'EOF'
void hashtree(const char* fname)
{
  TFile f(fname);
  auto t = (TTree*)f.Get("o2sim");
  if (!t) {
    return;
  }
  std::vector<std::tuple<TClass*, void*, std::string>> objs;
  objs.reserve(t->GetListOfBranches()->GetEntries()); // the branch addresses point into objs
  for (auto b : *t->GetListOfBranches()) {
    auto br = (TBranch*)b;
    TClass* cl = nullptr;
    EDataType dt;
    br->GetExpectedType(cl, dt);
    if (!cl) {
      continue;
    }
    objs.emplace_back(cl, cl->New(), br->GetName());
    br->SetAddress(&std::get<1>(objs.back()));
  }
  for (Long64_t i = 0; i < t->GetEntries(); ++i) {
    t->GetEntry(i);
    for (auto& [cl, obj, name] : objs) {
      TBufferFile buf(TBuffer::kWrite);
      buf.WriteObjectAny(obj, cl);
      TMD5 md5;
      md5.Update((UChar_t*)buf.Buffer(), buf.Length());
      md5.Final();
      printf("HASH event %lld branch %s %s\n", i, name.c_str(), md5.AsString());
      if (auto proxy = cl->GetCollectionProxy()) {
        TVirtualCollectionProxy::TPushPop helper(proxy, obj);
        printf("COUNT event %lld branch %s %u\n", i, name.c_str(), proxy->Size());
      }
    }
  }
}
EOF

run() { # <directory> <o2-sim mode options>
  local dir=$1
  shift
  mkdir -p $dir && cd $dir
  $o2sim "$@" -n $nev -g $gen -e TGeant4 --seed $seed --field $field -m ${modules[@]} --configKeyValues "$seedconf" > sim.log 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || ! grep -q "SIMULATION RETURNED SUCCESFULLY" sim.log; then
    echo "o2-sim in $dir failed (exit code $rc), see $workdir/$dir"
    exit 1
  fi
  for f in o2sim_Kine.root o2sim_Hits*.root; do
    [ -f $f ] || continue
    root -l -b -q "../hashtree.C(\"$f\")" 2>/dev/null > $f.out
    grep "^HASH" $f.out | sed "s/^HASH/$f/" > $f.hashes
    grep "^COUNT" $f.out | awk '{n += $NF} END {print n + 0}' > $f.count
  done
  cd ..
}

echo "o2-sim MT equality: ${modules[*]}, $nev events of $gen, field $field kG, $seeding seeding, 1 against $nthreads threads, in $workdir"
run seq -j 1
run mt1 --isMT 1 -j 1
run mt$nthreads --isMT 1 -j $nthreads
status=0
for h in mt1/*.hashes; do
  f=$(basename $h)
  file=${f%.hashes}
  if [ ! -s $h ]; then
    echo "$file: no entries"
    status=1
  elif cmp -s $h mt$nthreads/$f; then
    echo "$file: identical ($(wc -l < $h) event-branch hashes)"
  else
    echo "$file: DIFFERS, first at $(diff $h mt$nthreads/$f | grep -m1 '^<' | cut -d' ' -f3-6)"
    status=1
  fi
done
[ $(ls mt$nthreads/*.hashes | wc -l) -eq $(ls mt1/*.hashes | wc -l) ] || { echo "the runs wrote different output files"; status=1; }
# the sequential run as reference for the number of hits, which catches a detector clone that makes no hits
for c in seq/o2sim_Hits*.root.count; do
  [ -f "$c" ] || continue
  file=$(basename ${c%.count})
  nseq=$(cat $c)
  nmt=$(cat mt1/$file.count 2>/dev/null || echo 0)
  if [ "$nmt" -eq 0 ]; then
    echo "$file: no hits in MT (sequential: $nseq)"
    status=1
  elif ! awk -v a=$nseq -v b=$nmt -v t=$tolerance 'BEGIN {exit !(a > 0 && (a - b < 0 ? b - a : a - b) <= t * a)}'; then
    echo "$file: $nmt hits in MT against $nseq in sequential mode, beyond the tolerance $tolerance"
    status=1
  else
    echo "$file: $nmt hits in MT, $nseq in sequential mode"
  fi
done
[ $status -eq 0 ] && echo "EQUALITY TEST PASSED" || echo "EQUALITY TEST FAILED"
exit $status
