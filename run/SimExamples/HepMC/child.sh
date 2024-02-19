#!/usr/bin/env bash

cmd="./crmc.sh"
more="GeneratorFileOrCmd.bMaxSwitch=none"
seed=$RANDOM
nev=1
out=

usage()
{
    cat <<EOF
Usage: $0 [OPTIONS]

Options:

  -c,--cmdline COMMAND     Command line
  -m,--more    CONFIG      More configurations ($more)
  -s,--seed    SEED        Random number seed ($seed)
  -n,--nevents EVENTS      Number of events ($nev)
  -o,--output  OUTPUT      Output prefix ($out)
  --                       Rest of command line sent to o2-sim

COMMAND must be quoted if it contains spaces or other special
characters

Below follows the help output of o2-sim

EOF
}

while test $# -gt 0 ; do
    case $1 in
        -c|--cmdline) cmd="$2" ; shift ;;
        -s|--seed)    seed=$2 ; shift ;;
        -n|--nevents) nev=$2 ; shift ;;
        -o|--output)  out=$2 ; shift ;;
        -h|--help) usage; o2-sim --help full ; exit 0 ;;
        --)           shift ; break ;;
        *) echo "Unknown option '$1', did you forget '--'?" >/dev/stderr
           exit 1
           ;;
    esac
    shift
done

if test "x$out" = "x" ; then
    out=`echo $cmd | sed 's,^\./,,' | tr '[$/. ]' '_'`
fi
out=`echo "$out" | tr ' ' '_'`

export VMCWORKDIR=${O2_ROOT}/share
o2-sim -g hepmc --configKeyValues "GeneratorFileOrCmd.cmd=$cmd;${more}" \
       --outPrefix "$out" --seed $seed --nEvents $nev $@
