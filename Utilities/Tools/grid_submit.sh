#!/bin/bash

# A helper script, making it easy to submit existing
# scripts as an ALIEN GRID job (with the following notation):
#
# grid-submit my_script.sh jobname
#
# The script then handles all interaction with the GRID automatically. The user
# does not need to create JDLs files nor upload them to the GRID manually/herself.
#
# The script can also simulate execution of the job locally. To this end, it suffices
# to say
#
# ALIEN_PROC_ID=1 grid-submit my_script.sh
#
# Currently handles only a very basic JDL configuration. Further improvements would be:
#
# -) allow JDL customization via command line arguments or JDL tags inside the script
#

# set -o pipefail

function per() { printf "\033[31m$1\033[m\n" >&2; }
function pok() { printf "\033[32m$1\033[m\n" >&2; }
function banner() { echo ; echo ==================== $1 ==================== ; }

if [[ ! $ALIEN_PROC_ID && ! $1 ]]; then
   per "Please give a job script"
   exit 1
fi

# find out if this script is really executed on GRID
# in this case, we should be in a workdir labeled alien-job-${ALIEN_PROC_ID}
ONGRID=0
$(cd ../alien-job-${ALIEN_PROC_ID} 2> /dev/null)
if [[ "$?" == "0"  ]]; then
  ONGRID=1
fi

# General job configuration
MY_USER=${ALIEN_USER:-`whoami`}
if [[ ! $MY_USER ]]; then
  per "Problems retrieving current AliEn user. Did you run alien-token-init?"
  exit 1
fi
MY_HOMEDIR="/alice/cern.ch/user/${MY_USER:0:1}/${MY_USER}"
MY_BINDIR="$MY_HOMEDIR/bintest"
MY_JOBPREFIX="$MY_HOMEDIR/${ALIEN_TOPWORKDIR:-selfjobs}"
MY_JOBSCRIPT="$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")" # the job script with full path
MY_JOBNAME=${2:-$(basename ${MY_JOBSCRIPT})}
MY_JOBNAMEDATE="${MY_JOBNAME}-$(date -u +%Y%m%d-%H%M%S)"
MY_JOBWORKDIR="$MY_JOBPREFIX/${MY_JOBNAMEDATE}"  # ISO-8601 UTC

pok "Your job's working directory will be $MY_JOBWORKDIR"
pok "Set the job name by running $0 <scriptname> <jobname>"

#
# Generate local workdir
#
if [[ "${ONGRID}" == "0" ]]; then
  WORKDIR=${WORKDIR:-/tmp/alien_work/$(basename "$MY_JOBWORKDIR")}
  mkdir -p ${WORKDIR}
  cp "${MY_JOBSCRIPT}" "${WORKDIR}/alien_jobscript.sh"
fi

# 
# Submitter code
#
if [[ ! $ALIEN_PROC_ID ]]; then
  # We are not on a worker node: assuming client --> test if alien is there?
  which alien.py 2> /dev/null
  # check exit code
  if [[ ! "$?" == "0"  ]]; then
    XJALIEN_LATEST=`find /cvmfs/alice.cern.ch/el7-x86_64/Modules/modulefiles/xjalienfs -type f -printf "%f\n" | tail -n1`
    banner "Loading xjalienfs package $XJALIEN_LATEST since not yet loaded"
    eval "$(/cvmfs/alice.cern.ch/bin/alienv printenv xjalienfs::"$XJALIEN_LATEST")"
  fi

  # Create temporary workdir to assemble files, and submit from there (or execute locally)
  cd "$(dirname "$0")"
  THIS_SCRIPT="$PWD/$(basename "$0")"

  cd "${WORKDIR}"

  # ---- Generate JDL ----------------
  # TODO: Make this configurable or read from a preamble section in the jobfile
  cat > "${MY_JOBNAMEDATE}.jdl" <<EOF
Executable = "${MY_BINDIR}/${MY_JOBNAMEDATE}.sh";
InputFile = "LF:${MY_JOBWORKDIR}/alien_jobscript.sh";
OutputDir = "${MY_JOBWORKDIR}";
Output = {
  "*.log*,log.txt@disk=2"
};
Requirements = member(other.GridPartitions,"multicore_8");
MemorySize = "60GB";
TTL=80000;
EOF
#

  pok "Local working directory is $PWD"

  pok "Preparing job \"$MY_JOBNAMEDATE\""
  (
    set -x
    alien.py rmdir "$MY_JOBWORKDIR" || true                                   # remove existing job dir
    alien.py mkdir "$MY_BINDIR" || true                                       # create bindir
    alien.py mkdir "$MY_JOBPREFIX" || true                                    # create job output prefix
    alien.py mkdir jdl || true
    alien.py mkdir "$MY_JOBWORKDIR" || true
    alien.py rm "$MY_BINDIR/${MY_JOBNAMEDATE}.sh" || true                     # remove current job script
    alien.py cp "${PWD}/${MY_JOBNAMEDATE}.jdl" alien://${MY_HOMEDIR}/jdl/${MY_JOBNAMEDATE}.jdl@ALICE::CERN::EOS || true  # copy the jdl
    alien.py cp "$THIS_SCRIPT" alien://${MY_BINDIR}/${MY_JOBNAMEDATE}.sh@ALICE::CERN::EOS || true  # copy current job script to AliEn
    alien.py cp "${MY_JOBSCRIPT}" alien://${MY_JOBWORKDIR}/alien_jobscript.sh@ALICE::CERN::EOS || true
  ) &> alienlog.txt

  pok "Submitting job \"${MY_JOBNAMEDATE}\" from $PWD"
  (
    alien.py submit jdl/${MY_JOBNAMEDATE}.jdl || true
  ) &>> alienlog.txt

  MY_JOBID=$( (grep 'Your new job ID is' alienlog.txt | grep -oE '[0-9]+' || true) | sort -n | tail -n1)
  if [[ $MY_JOBID ]]; then
    pok "OK, display progress on https://alimonitor.cern.ch/agent/jobs/details.jsp?pid=$MY_JOBID"
  else
    per "Job submission failed: error log follows"
    cat alienlog.txt
  fi

  exit 0
fi

####################################################################################################
# The following part is executed on the worker node or locally
####################################################################################################
if [[ "${ONGRID}" == 0 ]]; then
  banner "Executing job in directory ${WORKDIR}"
  cd "${WORKDIR}" 2> /dev/null
fi

# All is redirected to log.txt but kept on stdout as well
if [[ $ALIEN_PROC_ID ]]; then
  exec &> >(tee -a log.txt)
fi

# ----------- START JOB PREAMBLE  ----------------------------- 
banner "Environment"
env

banner "OS detection"
lsb_release -a || true
cat /etc/os-release || true
cat /etc/redhat-release || true

if [ ! "$O2_ROOT" ]; then
  O2_PACKAGE_LATEST=`find /cvmfs/alice.cern.ch/el7-x86_64/Modules/modulefiles/O2 -type f -printf "%f\n" | tail -n1`
  banner "Loading O2 package $O2_PACKAGE_LATEST"
  eval "$(/cvmfs/alice.cern.ch/bin/alienv printenv O2::"$O2_PACKAGE_LATEST")"
fi
if [ ! "$XJALIEN_ROOT" ]; then
  XJALIEN_LATEST=`find /cvmfs/alice.cern.ch/el7-x86_64/Modules/modulefiles/xjalienfs -type f -printf "%f\n" | tail -n1`
  banner "Loading XJALIEN package $XJALIEN_LATEST"
  eval "$(/cvmfs/alice.cern.ch/bin/alienv printenv xjalienfs::"$XJALIEN_LATEST")"
fi
if [ ! "$O2DPG_ROOT" ]; then
  O2DPG_LATEST=`find /cvmfs/alice.cern.ch/el7-x86_64/Modules/modulefiles/O2DPG -type f -printf "%f\n" | tail -n1`
  banner "Loading O2DPG package $O2DPG_LATEST"
  eval "$(/cvmfs/alice.cern.ch/bin/alienv printenv O2DPG::"$O2DPG_LATEST")"
fi

banner "Running workflow"

ldd `which o2-sim` > ldd.log
ls > ls.log

# collect some common information

cat /proc/cpuinfo > cpuinfo.log 
cat /proc/meminfo > meminfo.log

# ----------- EXECUTE ACTUAL JOB  ----------------------------- 
# source the actual job script from the work dir
chmod +x ./alien_jobscript.sh
./alien_jobscript.sh

# We need to exit for the ALIEN JOB HANDLER!
exit 0
