#!/bin/bash
if [ "0$1" == "0" ]; then
  echo Specify how many timeframe files to generate!
  exit 1
fi

echo Generating $1 timeframe files

# Defaults for 128 orbit TF, feel free to override
NEvents=${NEvents:-650}
NEventsQED=${NEventsQED:-35000}
SHMSIZE=${SHMSIZE:-128000000000}
TPCTRACKERSCRATCHMEMORY=${SHMSIZE:-25000000000}

if [ `which StfBuilder 2> /dev/null | wc -l` == "0" ]; then
  echo ERROR: StfBuilder is not in the path
  exit 1
fi
if [ `which o2-readout-exe 2> /dev/null | wc -l` == "0" ]; then
  echo ERROR: o2-readout-exe is not in the path
  exit 1
fi
if [ "0$O2_ROOT" == "0" ]; then
  echo \$O2_ROOT missing
  exit 1
fi
if [ ! -f matbud.root ]; then
  echo ERROR: matbud.root missing
  exit 1
fi

mkdir -p raw/timeframe
rm -Rf sim
mkdir -p simqed

for i in `seq 0 $(($1 - 1))`; do
  mkdir sim
  pushd sim > /dev/null
  ln -s ../simqed qed
  if [[ "0$COUNT_UP_FIRST_ORBIT_SAMPLED" == "01" ]]; then
    export FIRSTSAMPLEDORBIT=$(($i * 128))
  fi
  SPLITTRDDIGI=0 DISABLE_PROCESSING=1 $O2_ROOT/prodtests/full_system_test.sh
  if [[ $? != 0 ]]; then
    echo "FST failed"
    exit 1
  fi
  $O2_ROOT/prodtests/full-system-test/convert-raw-to-tf-file.sh
  popd > /dev/null
  mv `ls -S sim/raw/timeframe/*.tf | head -n 1` raw/timeframe/`ls raw/timeframe/*.tf 2> /dev/null | wc -l | awk '{printf("%08d.tf", $1+1)}'`
  rm -Rf sim
done

rm -Rf simqed
