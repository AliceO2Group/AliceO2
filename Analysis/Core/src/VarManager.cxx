#include "Analysis/VarManager.h"

//#include "Framework/runDataProcessing.h"
//#include "Framework/ASoA.h"
//#include "Framework/ASoAHelpers.h"
//#include "Framework/AnalysisTask.h"
//#include "Framework/AnalysisDataModel.h"
//#include "Analysis/ReducedInfoTables.h"


//using namespace o2;
//using namespace o2::framework;
//using namespace o2::framework::expressions;
//using namespace o2::aod;

//using Event = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>::iterator;
//using Track = soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel>::iterator;

ClassImp(VarManager)

TString VarManager::fgVariableNames[VarManager::kNVars] = {""};
TString VarManager::fgVariableUnits[VarManager::kNVars] = {""};
Bool_t  VarManager::fgUsedVars[VarManager::kNVars] = {kFALSE};

//__________________________________________________________________
VarManager::VarManager() :
  TObject()
{
  //
  // constructor
  //
  SetDefaultVarNames();
}

//__________________________________________________________________
VarManager::~VarManager() {
  //
  // destructor
  //
}

//__________________________________________________________________
void VarManager::SetVariableDependencies() {
  //
  // Set as used those variables on which other variables calculation depends
  //
}


//__________________________________________________________________
void VarManager::FillEvent(o2::aod::ReducedEvent event, float* values) {
  
  values[kRunNo] = event.runNumber();
  values[kVtxX] = event.posX();
  values[kVtxY] = event.posY();
  values[kVtxZ] = event.posZ();
  values[kVtxNcontrib] = event.numContrib();
  /*values[kVtxChi2] = event.chi2();
  values[kBC] = event.bc();
  values[kCentVZERO] = event.centVZERO();
  values[kVtxCovXX] = event.covXX();
  values[kVtxCovXY] = event.covXY();
  values[kVtxCovXZ] = event.covXZ();
  values[kVtxCovYY] = event.covYY();
  values[kVtxCovYZ] = event.covYZ();
  values[kVtxCovZZ] = event.covZZ();*/
  
}

//__________________________________________________________________
void VarManager::FillTrack(o2::aod::ReducedTrack track, float* values) {
  
  values[kPt] = track.pt();
  values[kEta] = track.eta();
  values[kPhi] = track.phi();
  values[kCharge] = track.charge();
  /*values[kPin] = track.tpcInnerParam();
  if(fgUsedVars[kITSncls]) {
    values[kITSncls] = 0.0;
    for(int i=0; i<6; ++i) 
      values[kITSncls] += ((track.itsClusterMap() & (1<<i)) ? 1 : 0);
  }
  values[kITSchi2] = track.itsChi2NCl();
  values[kTPCncls] = track.tpcNCls();
  values[kTPCchi2] = track.tpcChi2NCl();
  values[kTPCsignal] = track.tpcSignal();
  values[kTRDsignal] = track.trdSignal();
  values[kTOFsignal] = track.tofSignal();
  values[kTrackLength] = track.barrelLength();*/
  
}


//__________________________________________________________________
void VarManager::SetDefaultVarNames() {
  //
  // Set default variable names
  //
  for(Int_t ivar=0; ivar<kNVars; ++ivar) {
    fgVariableNames[ivar] = "DEFAULT NOT DEFINED"; fgVariableUnits[ivar] = "n/a";
  }
  
  fgVariableNames[kRunNo] = "Run number";  fgVariableUnits[kRunNo] = "";
  fgVariableNames[kRunNo] = "Run number";  fgVariableUnits[kRunNo] = "";
  fgVariableNames[kRunTimeStart] = "Run start time";  fgVariableUnits[kRunTimeStart] = "";
  fgVariableNames[kRunTimeStop] = "Run stop time";  fgVariableUnits[kRunTimeStop] = "";
  fgVariableNames[kLHCFillNumber] = "LHC fill number";  fgVariableUnits[kLHCFillNumber] = "";
  fgVariableNames[kDipolePolarity] = "Dipole polarity";  fgVariableUnits[kDipolePolarity] = "";
  fgVariableNames[kL3Polarity] = "L3 polarity";  fgVariableUnits[kL3Polarity] = "";
  fgVariableNames[kTimeStamp] = "Time stamp";  fgVariableUnits[kTimeStamp] = "";
  fgVariableNames[kBC] = "Bunch crossing";  fgVariableUnits[kBC] = "";
  fgVariableNames[kInstLumi] = "Instantaneous luminosity";  fgVariableUnits[kInstLumi] = "Hz/mb";
  fgVariableNames[kEventType] = "Event type";  fgVariableUnits[kEventType] = "";
  fgVariableNames[kIsPhysicsSelection] = "Physics selection";  fgVariableUnits[kIsPhysicsSelection] = "";
  fgVariableNames[kVtxX] = "Vtx X ";  fgVariableUnits[kVtxX] = "cm";
  fgVariableNames[kVtxY] = "Vtx Y ";  fgVariableUnits[kVtxY] = "cm";
  fgVariableNames[kVtxZ] = "Vtx Z ";  fgVariableUnits[kVtxZ] = "cm";
  fgVariableNames[kVtxNcontrib] = "Vtx contrib.";  fgVariableUnits[kVtxNcontrib] = "";
  fgVariableNames[kVtxCovXX] = "Vtx covXX";  fgVariableUnits[kVtxCovXX] = "cm";
  fgVariableNames[kVtxCovXY] = "Vtx covXY";  fgVariableUnits[kVtxCovXY] = "cm";
  fgVariableNames[kVtxCovXZ] = "Vtx covXZ";  fgVariableUnits[kVtxCovXZ] = "cm";
  fgVariableNames[kVtxCovYY] = "Vtx covYY";  fgVariableUnits[kVtxCovYY] = "cm";
  fgVariableNames[kVtxCovYZ] = "Vtx covYZ";  fgVariableUnits[kVtxCovYZ] = "cm";
  fgVariableNames[kVtxCovZZ] = "Vtx covZZ";  fgVariableUnits[kVtxCovZZ] = "cm";
  fgVariableNames[kVtxChi2] = "Vtx chi2";  fgVariableUnits[kVtxChi2] = "";
  fgVariableNames[kCentVZERO] = "Centrality VZERO";  fgVariableUnits[kCentVZERO] = "%";
  fgVariableNames[kPt] = "p_{T}";  fgVariableUnits[kPt] = "GeV/c";
  fgVariableNames[kEta] = "#eta";  fgVariableUnits[kEta] = "";
  fgVariableNames[kPhi] = "#varphi";  fgVariableUnits[kPhi] = "rad.";
  fgVariableNames[kRap] = "y";  fgVariableUnits[kRap] = "";
  fgVariableNames[kMass] = "mass";  fgVariableUnits[kMass] = "GeV/c2";
  fgVariableNames[kCharge] = "charge";  fgVariableUnits[kCharge] = "";
  fgVariableNames[kPin] = "p_{IN}";  fgVariableUnits[kPin] = "GeV/c";
  fgVariableNames[kITSncls] = "ITS #cls";  fgVariableUnits[kITSncls] = "";
  fgVariableNames[kITSchi2] = "ITS chi2";  fgVariableUnits[kITSchi2] = "";
  fgVariableNames[kITSlayerHit] = "ITS layer";  fgVariableUnits[kITSlayerHit] = "";
  fgVariableNames[kTPCncls] = "TPC #cls";  fgVariableUnits[kTPCncls] = "";
  fgVariableNames[kTPCchi2] = "TPC chi2";  fgVariableUnits[kTPCchi2] = "";
  fgVariableNames[kTPCsignal] = "TPC dE/dx";  fgVariableUnits[kTPCsignal] = "";
  fgVariableNames[kTRDsignal] = "TRD dE/dx";  fgVariableUnits[kTRDsignal] = "";
  fgVariableNames[kTOFsignal] = "TOF signal";  fgVariableUnits[kTOFsignal] = "";
  fgVariableNames[kTrackLength] = "track length";  fgVariableUnits[kTrackLength] = "cm";
  fgVariableNames[kCandidateId] = "";  fgVariableUnits[kCandidateId] = "";
  fgVariableNames[kPairType] = "Pair type";  fgVariableUnits[kPairType] = "";
  fgVariableNames[kPairLxy] = "Pair Lxy";  fgVariableUnits[kPairLxy] = "cm";
  fgVariableNames[kDeltaEta] = "#Delta#eta";  fgVariableUnits[kDeltaEta] = "";
  fgVariableNames[kDeltaPhi] = "#Delta#phi";  fgVariableUnits[kDeltaPhi] = "";
}
