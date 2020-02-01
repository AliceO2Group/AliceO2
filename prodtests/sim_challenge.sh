#!/bin/bash

# chain of algorithms from MC and reco

# default number of events
nevPP=10
nevPbPb=10

# default collision system
collSyst="pp"

generPP="pythia8"
generPbPb="pythia8hi"

# options to pass to every workflow
gloOpt=" -b --run "


Usage() 
{
  echo "Usage: ${0##*/} [-s system /pp[Def] or pbpb/] [-n Number of events /Def = $nevPP(pp) or $nevPbPb(pbpb)/]"
  exit
}


while [ $# -gt 0 ] ; do
    case $1 in
	-n) nev=$2;  shift 2 ;;
	-s) collSyst=$2; shift 2 ;;
	*) echo "Wrong input"; Usage;
    esac
done

collSyst="${collSyst,,}" # convert to lower case
if [ "$collSyst" == "pp" ]; then
    gener="$generPP"
    [[ "nev" -lt "1"  ]] && nev="$nevPP"
elif [ "$collSyst" == "pbpb" ]; then
    gener="$generPbPb"
    [[ "nev" -lt "1"  ]] && nev="$nevPbPb"
else
    echo "Wrong collision system $collSyst provided, should be pp or pbpb"
    Usage
fi

#---------------------------------------------------
echo "Running simulation for $nev $collSyst events with $gener generator"
o2-sim -n"$nev" --configKeyValue "Diamond.width[2]=6." -g "$gener" &> sim.log

echo "Running digitization"
o2-sim-digitizer-workflow $gloOpt  &>  digi.log
# existing checks
#root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckDigits.C+

echo "Running ITS reco flow"
#needs ITS digitized data
o2-its-reco-workflow  $gloOpt &> itsreco.log
# existing checks
# root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckClusters.C+
# root -b -q O2/Detectors/ITSMFT/ITS/macros/test/CheckTracks.C+

echo "Running MFT reco flow"
#needs MFT digitized data
o2-mft-reco-workflow  $gloOpt &> mftreco.log

echo "Running FIT(FT0) reco flow"
#needs FIT digitized data
o2-fit-reco-workflow $gloOpt &> fitreco.log

echo "Running TPC reco flow"
#needs TPC digitized data
o2-tpc-reco-workflow $gloOpt --tpc-digit-reader '--infile tpcdigits.root' --input-type digits --output-type clusters,tracks  --tpc-track-writer "--treename events --track-branch-name Tracks --trackmc-branch-name TracksMCTruth" &> tpcreco.log

echo "Running ITS-TPC macthing flow"
#needs results of o2-tpc-reco-workflow, o2-its-reco-workflow and o2-fit-reco-workflow
o2-tpcits-match-workflow $gloOpt --tpc-track-reader "tpctracks.root" --tpc-native-cluster-reader "--infile tpc-native-clusters.root" &> itstpcMatch.log

echo "Running ITSTPC-TOF macthing flow"
#needs results of TOF digitized data and results of o2-tpcits-match-workflow
o2-tof-reco-workflow $gloOpt >& tofMatch.log
