#Number of events to simulate
nEvents=${nEvents:-10}

# Simulating
echo "=== Simulating $nEvents events ==="
o2-sim-serial-run5 -n "$nEvents" -g pythia8hi -m TRK --configKeyValues "TRKBase.layoutMLOT=kSegmented" >&sim_TRK.log

# Digitizing
echo "=== Digitizing ==="
o2-sim-digitizer-workflow -b >&digiTRK.log

root.exe -b -q CheckDigits.C+ >&CheckDigits.log

PERFORM_RECO=${PERFORM_RECO:-0}

if [ "$PERFORM_RECO" -eq 0 ]; then
    exit 0
fi

echo "=== Performing reconstruction ==="

# Reconstruction
if [[ -f "/tmp/trkft3_reco.json" ]]; then
    echo '
{
  "inputfiles": {
    "hits": "o2sim_HitsTRK.root",
    "geometry": "o2sim_geometry.root",
    "mcHeader": "o2sim_MCHeader.root",
    "kinematics": "o2sim_Kine.root"
  },
  "geometry": {
    "bz": 20.0,
    "pitch": [0.001, 0.001, 0.001, 0.001, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004]
  },
  "trackingparams": [{
    "NLayers": 11,
    "DeltaROF": 0,
    "LayerZ": [25.1, 25.1, 25.1, 64.2, 64.2, 64.2, 64.2, 64.2, 128.5, 128.5, 128.5],
    "LayerRadii": [0.5, 1.2, 2.5, 7.05, 9.05, 12.05, 20.05, 30.05, 45.05, 60.5, 80.05],
    "LayerxX0": [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "LayerResolution": [0.0003, 0.0003, 0.0003, 0.0003, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012],
    "SystErrorY2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "SystErrorZ2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "AddTimeError": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ZBins": 256,
    "PhiBins": 128,
    "nROFsPerIterations": -1,
    "UseDiamond": false,
    "Diamond": [0.0, 0.0, 0.0],
    "AllowSharingFirstCluster": false,
    "ClusterSharing": 0,
    "MinTrackLength": 7,
    "NSigmaCut": 10,
    "PVres": 0.01,
    "TrackletMinPt": 0.1,
    "TrackletsPerClusterLimit": 2.0,
    "CellDeltaTanLambdaSigma": 0.007,
    "CellsPerClusterLimit": 2.0,
    "MaxChi2ClusterAttachment": 60.0,
    "MaxChi2NDF": 30.0,
    "ReseedIfShorter": 6,
    "MinPt": [0.0, 0.0, 0.0, 0.0, 0.0],
    "StartLayerMask": 4095,
    "RepeatRefitOut": false,
    "ShiftRefToCluster": true,
    "FindShortTracks": false,
    "PerPrimaryVertexProcessing": false,
    "SaveTimeBenchmarks": false,
    "DoUPCIteration": false,
    "FataliseUponFailure": true,
    "TrackFollower": "mix",
    "TrackFollowerNSigmaCutZ": 1.0,
    "TrackFollowerNSigmaCutPhi": 1.0,
    "TrackFollowerMaxHypotheses": 1,
    "createArtefactLabels": false,
    "PrintMemory": false,
    "DropTFUponFailure": false
  }]
}
' >/tmp/trkft3_reco.json
fi

OPT=""
OPT="$OPT -b "
OPT="$OPT --shm-segment-size 16000000000 "
# OPT="$OPT --severity debug"

if [ ! -f o2trac_trk_acts.root ]; then
    time o2-alice3-global-reconstruction-reco-workflow $OPT --useACTS --tracking-from-clusters-config /tmp/trkft3_reco.json
    mv o2trac_trk.root o2trac_trk_acts.root
else
    echo "o2trac_trk_acts.root already exists, skipping reconstruction"
fi
root.exe -b -q 'CheckTracksCA.C+("o2trac_trk_acts.root")' >&CheckTracksCA_acts.log

if [ ! -f o2trac_trk_ca.root ]; then
    time o2-alice3-global-reconstruction-reco-workflow $OPT --tracking-from-clusters-config /tmp/trkft3_reco.json
    mv o2trac_trk.root o2trac_trk_ca.root
else
    echo "o2trac_trk_ca.root already exists, skipping reconstruction"
fi
root.exe -b -q 'CheckTracksCA.C+("o2trac_trk_ca.root")' >&CheckTracksCA_ca.log
