#!/bin/bash

#
# This is running a simple anchored MC production to test whether it is working
# simulation -> AOD -> QC -> AnalysisQC
#

if [[ -z ${O2DPG_ROOT+x} ]] ; then
    echo "ERROR: O2DPG is not loaded, probably other packages are missing as well in this environment."
    exit 1
fi

if [[ -z ${O2_ROOT+x} ]] ; then
    echo "ERROR: O2 is not loaded, probably other packages are missing as well in this environment."
    exit 1
fi

if [[ -z ${O2PHYSICS_ROOT+x} ]] ; then
    echo "ERROR: O2Physics is not loaded, probably other packages are missing as well in this environment."
    exit 1
fi

if [[ -z ${QUALITYCONTROL_ROOT+x} ]] ; then
    echo "ERROR: QC is not loaded, probably other packages are missing as well in this environment."
    exit 1
fi

if [[ "$(which jq)" == "" ]] ; then
    echo "ERROR: JQ is not loaded, probably other packages are missing as well in this environment."
    exit 1
fi

export NWORKERS=2

export ALIEN_JDL_LPMANCHORPASSNAME=apass4
export ALIEN_JDL_MCANCHOR=apass4
export ALIEN_JDL_COLLISIONSYSTEM=p-p
export ALIEN_JDL_CPULIMIT=8
export ALIEN_JDL_LPMPASSNAME=apass4
export ALIEN_JDL_LPMRUNNUMBER=526467
export ALIEN_JDL_LPMPRODUCTIONTYPE=MC
export ALIEN_JDL_LPMINTERACTIONTYPE=pp
export ALIEN_JDL_LPMPRODUCTIONTAG=LHC23k2
export ALIEN_JDL_LPMANCHORRUN=526467
export ALIEN_JDL_LPMANCHORPRODUCTION=LHC22o
export ALIEN_JDL_LPMANCHORYEAR=2022

export ALIEN_PROC_ID=2963436952

run_sim()
{
  args=""
  nevents=0
  shellscript=""

  while [ ! -z "$1" ]; do
      option="$1"
      shift

      if [ "$option" = "--nevents" ]; then
          nevents=$1
          shift
      else
          if [ -z "$args" ]; then
              args="$option"
          else
              if [[ "$option" == *".sh" ]]; then
                if [[ -f $option ]]; then
                  shellscript="${PWD}/$option"
                else
                  shellscript=${O2DPG_ROOT}/$option
                fi
                args="$args $shellscript"
              else
                args="$args $option"
              fi

          fi
      fi
  done

  echo "  nevents (for ML counters) $nevents"
  echo "  shell script $shellscript"
  if [[ ! -z "$shellscript" ]]; then
      echo "************* Steering shell script *************"
      cat $shellscript
      echo "************* DONE *************"
  else
      echo "Steering shell script not defined!"
  fi

  echo "Running ${args}"

  # is passed via input files from JDL
  chmod +x *.sh
  chmod +x *.py

  exec ${args}
  return ${?}
}

cp $O2_ROOT/prodtests/full-system-test/anchorMC811.sh .
cp $O2_ROOT/prodtests/full-system-test/async_pass.sh .

run_sim env NTIMEFRAMES=3 NSIGEVENTS=5 SPLITID=100 PRODSPLIT=153 CYCLE=0 ./anchorMC811.sh --nevents 15
exit ${?}
