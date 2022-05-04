<!-- doxy
\page refMFT MFT
/doxy -->

# The Muon Forward Tracker

This documentation describes the main workflows used for MFT standalone reconstruction and global forward matching. The MFT is a Silicon pixel tracking detector designed to add vertexing capabilities to the MUON spectrometer. More details about the MFT are available on the [Technical Design Report for the Muon Forward Tracker](https://cds.cern.ch/record/1981898).

## MFT Standalone workflows in O2

`o2-mft-reco-workflow` is the main workflow used in MFT standalone tracking, reconstructing MFT tracks stored in `mfttracks.root`. The workflow options are defined in [mft-reco-workflow.cxx](./workflow/src/mft-reco-workflow.cxx) and tracking configurable parameters at [`MFTTrackingParam.h`](./tracking/include/MFTTracking/MFTTrackingParam.h). During normal data taking periods, this workflows find and fit tracks from clusters provided by upstream devices. For simulated data this workflows reads simulated digits, runs clusterization, track-finding and track-fitting.

### Running MFT Standalone reconstruction from a simulation

```bash
# 0) Enter O2 environment
alienv enter O2/latest-dev-o2

# 1) Run MFT standalone simulation of 10 pp events
o2-sim -n 10 -g pythia8pp -m MFT

# 2) Run MFT digitizer: produces `mftdigits.root`
o2-sim-digitizer-workflow -b

# 3) Run MFT reconstruction workflow from digits
#    produces mftclusters.root and mfttracks.root
o2-mft-reco-workflow  --configKeyValues "MFTTracking.forceZeroField=false;MFTTracking.LTFclsRCut=0.0100;"
```

### MFT Standalone reconstruction from CTFs

Workflow: `CTF reader workflow -> MFT reconstruction workflow`

Example:
```bash
o2-ctf-reader-workflow --onlyDet MFT  --ctf-input o2_ctf_run00000000_orbit0000000000_tf0000000001.root | o2-mft-reco-workflow --clusters-from-upstream --disable-mc  -b
```

### Reconstruction from MFT clusters (real data)

Workflow: `MFT cluster reader workflow -> MFT reconstruction workflow`

Example:
```bash
o2-mft-cluster-reader-workflow | o2-mft-reco-workflow --clusters-from-upstream --disable-mc --mft-cluster-writer "--outfile /dev/null"
```

### MFT Assessment

The `o2-mft-assessment-workflow` evaluates MFT standalone tracking. By default the workflow operates in data collection mode and stores `MFTAssessment.root`.

Usage modes:
  1. Piped with matching workflow:
    -  `o2-mft-reco-workflow | o2-mft-assessment-workflow`
  2. Reading data on disk:
    - `o2-global-track-cluster-reader --track-types MFT --cluster-types MFT | o2-mft-assessment-workflow`

Analysis of collected data can be executed calling the workflow with `--    `, producing objects that cannot be merged. For grid-compatible workflows, data collected in several files can be merged with `hadd` and the finalization of merged objects can be steered by a macro as bellow.

```cpp
void finalizeMFTAssessment()
{
  o2::mft::MFTAssessment analyser(true);
  analyser.loadHistos(); // loads MFFAssessment.root produced by the DPL workflow
  analyser.finalizeAnalysis();
  TFile *fout = new TFile("MFTAssessmentFinalized.root", "RECREATE");
  TObjArray objarOut;
  analyser.getHistos(objarOut); // Output objects
  objarOut.Write();
  fout->Close();
}
```

## MFT in Global Forward Workflows

Global forward workflows produce tracks by matching different detector systems:
* Standalone reconstruction workflows: MFT, MCH and MID
* Matching workflows: MCH-MID and MFT-MCH-MID

A compreensive execution of ALICE workflows from simulation up to AOD files is implemented on [`sim_challenge.sh`](../../../prodtests/sim_challenge.sh). Bellow a summary of the reconstruction workflows used to produce global forward tracks.

### MCH and MID reconstruction workflows

* MCH reconstruction workflow:
    * `o2-mch-reco-workflow`
    * produces `mchtracks.root`
* MID reconstruction workflow:
    * `o2-mid-digits-reader-workflow | o2-mid-reco-workflow`
    * produces `mid-reco.root`

More details about these workflows are available at the documentation for [MCH](../../MUON/MCH/Workflow/README.md) and the [MID](../../MUON/MID/Workflow/README.md).

### MCH-MID matching

`o2-muon-tracks-matcher-workflow` runs MCH-MID matching, using MCH tracks (`mchtracks.root`) and MID tracks (`mid-reco.root`). By default it produces `muontracks.root`, which contains only track-matching information, as defined in [TrackMCHMID.h](../../../DataFormats/Reconstruction/include/ReconstructionDataFormats/TrackMCHMID.h).

### MFT-MCH-MID matching

`o2-globalfwd-matcher-workflow` runs MFT-MCH-MID matching, using MFT tracks (`mfttracks.root`), MCH Tracks (`mchtracks.root`) and MCH-MID matches (`muontracks.root)`. MID information is used to identify GlobalMuonTracks. By default `o2-globalfwd-matcher-workflow` produces `globalfwdtracks.root` which contains matched forward tracks.

Global forward matching can be configured at runtime with options defined in [MatchGlobalFwdParam.h](../../GlobalTracking/include/GlobalTracking/MatchGlobalFwdParam.h). Example:

```
o2-globalfwd-matcher-workflow --configKeyValues "FwdMatching.useMIDMatch=true"
```

### Global Forward Assessment Workflow

The `o2-globalfwd-assessment-workflow` evaluates reconstruction performance, the pairing efficiency and purity of global muon tracks. By default the workflow only collects data, creating mergeable objects.

Usage modes:
  1. Piped with matching workflow:
    -  `o2-globalfwd-matcher-workflow | o2-globalfwd-assessment-workflow`
  2. Reading data on disk:
    - `o2-global-track-cluster-reader --track-types MFT,MCH,MFT-MCH | o2-globalfwd-assessment-workflow`

Analysis of collected data can be executed calling the workflow with `--finalize-analysis`, producing objects that cannot be merged. The finalization of merged objects can be steered by a macro as bellow.

```cpp
void finalizeGlobalFwdAssessment()
{
    o2::globaltracking::GloFwdAssessment analyser(true);
    analyser.loadHistos(); // loads GlobalForwardAssessment.root produced by the DPL workflow
    analyser.finalizeCutConfig(1., 15., 15); // finalizeCutConfig(float minCut, float maxCut, int nSteps)
    analyser.finalizeAnalysis();
    TFile *fout = new TFile("GlobalForwardAssessmentFinalized.root", "RECREATE");
    TObjArray objarOut;
    analyser.getHistos(objarOut);
    objarOut.Write();
    fout->Close();
}
```
