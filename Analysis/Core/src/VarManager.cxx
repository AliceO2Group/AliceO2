// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Analysis/VarManager.h"

ClassImp(VarManager)

  TString VarManager::fgVariableNames[VarManager::kNVars] = {""};
TString VarManager::fgVariableUnits[VarManager::kNVars] = {""};
bool VarManager::fgUsedVars[VarManager::kNVars] = {kFALSE};
float VarManager::fgValues[VarManager::kNVars] = {0.0};
std::map<int, int> VarManager::fgRunMap;

//__________________________________________________________________
VarManager::VarManager() : TObject()
{
  //
  // constructor
  //
  SetDefaultVarNames();
}

//__________________________________________________________________
VarManager::~VarManager() = default;

//__________________________________________________________________
void VarManager::SetVariableDependencies()
{
  //
  // Set as used those variables on which other variables calculation depends
  //
}

//__________________________________________________________________
void VarManager::ResetValues(int startValue, int endValue)
{
  //
  // reset all variables to an "innocent" value
  // NOTE: here we use -9999.0 as a neutral value, but depending on situation, this may not be the case
  for (Int_t i = startValue; i < endValue; ++i)
    fgValues[i] = -9999.;
}

//__________________________________________________________________
void VarManager::SetRunNumbers(int n, int* runs)
{
  //
  // maps the list of runs such that one can plot the list of runs nicely in a histogram axis
  //
  for (int i = 0; i < n; ++i)
    fgRunMap[runs[i]] = i + 1;
}

//__________________________________________________________________
void VarManager::FillEvent(vector<float> event, float* values)
{

  //TODO: the Fill function should take as argument an aod::ReducedEvent iterator, this is just a temporary fix
  if (!values)
    values = fgValues;
  values[kRunNo] = event[0];
  values[kRunId] = (fgRunMap.size() > 0 ? fgRunMap[int(values[kRunNo])] : 0);
  values[kVtxX] = event[1];
  values[kVtxY] = event[2];
  values[kVtxZ] = event[3];
  values[kVtxNcontrib] = event[4];
  /*values[kRunNo] = event.runNumber();
  values[kVtxX] = event.posX();
  values[kVtxY] = event.posY();
  values[kVtxZ] = event.posZ();
  values[kVtxNcontrib] = event.numContrib();*/
  //values[kVtxChi2] = event.chi2();
  //values[kBC] = event.bc();
  //values[kCentVZERO] = event.centVZERO();
  //values[kVtxCovXX] = event.covXX();
  //values[kVtxCovXY] = event.covXY();
  //values[kVtxCovXZ] = event.covXZ();
  //values[kVtxCovYY] = event.covYY();
  //values[kVtxCovYZ] = event.covYZ();
  //values[kVtxCovZZ] = event.covZZ();
}

//__________________________________________________________________
void VarManager::FillTrack(vector<float> track, float* values)
{

  if (!values)
    values = fgValues;

  values[kPt] = track[0];
  values[kEta] = track[1];
  values[kPhi] = track[2];
  values[kCharge] = track[3];
  /*values[kPt] = track.pt();
  values[kEta] = track.eta();
  values[kPhi] = track.phi();
  values[kCharge] = track.charge();*/
  //values[kPin] = track.tpcInnerParam();
  //if(fgUsedVars[kITSncls]) {
  //  values[kITSncls] = 0.0;
  //  for(int i=0; i<6; ++i)
  //    values[kITSncls] += ((track.itsClusterMap() & (1<<i)) ? 1 : 0);
  //}
  //values[kITSchi2] = track.itsChi2NCl();
  //values[kTPCncls] = track.tpcNCls();
  //values[kTPCchi2] = track.tpcChi2NCl();
  //values[kTPCsignal] = track.tpcSignal();
  //values[kTRDsignal] = track.trdSignal();
  //values[kTOFsignal] = track.tofSignal();
  //values[kTrackLength] = track.barrelLength();
}

//__________________________________________________________________
void VarManager::SetDefaultVarNames()
{
  //
  // Set default variable names
  //
  for (Int_t ivar = 0; ivar < kNVars; ++ivar) {
    fgVariableNames[ivar] = "DEFAULT NOT DEFINED";
    fgVariableUnits[ivar] = "n/a";
  }

  fgVariableNames[kRunNo] = "Run number";
  fgVariableUnits[kRunNo] = "";
  fgVariableNames[kRunId] = "Run number";
  fgVariableUnits[kRunId] = "";
  fgVariableNames[kRunTimeStart] = "Run start time";
  fgVariableUnits[kRunTimeStart] = "";
  fgVariableNames[kRunTimeStop] = "Run stop time";
  fgVariableUnits[kRunTimeStop] = "";
  fgVariableNames[kLHCFillNumber] = "LHC fill number";
  fgVariableUnits[kLHCFillNumber] = "";
  fgVariableNames[kDipolePolarity] = "Dipole polarity";
  fgVariableUnits[kDipolePolarity] = "";
  fgVariableNames[kL3Polarity] = "L3 polarity";
  fgVariableUnits[kL3Polarity] = "";
  fgVariableNames[kTimeStamp] = "Time stamp";
  fgVariableUnits[kTimeStamp] = "";
  fgVariableNames[kBC] = "Bunch crossing";
  fgVariableUnits[kBC] = "";
  fgVariableNames[kInstLumi] = "Instantaneous luminosity";
  fgVariableUnits[kInstLumi] = "Hz/mb";
  fgVariableNames[kEventType] = "Event type";
  fgVariableUnits[kEventType] = "";
  fgVariableNames[kIsPhysicsSelection] = "Physics selection";
  fgVariableUnits[kIsPhysicsSelection] = "";
  fgVariableNames[kVtxX] = "Vtx X ";
  fgVariableUnits[kVtxX] = "cm";
  fgVariableNames[kVtxY] = "Vtx Y ";
  fgVariableUnits[kVtxY] = "cm";
  fgVariableNames[kVtxZ] = "Vtx Z ";
  fgVariableUnits[kVtxZ] = "cm";
  fgVariableNames[kVtxNcontrib] = "Vtx contrib.";
  fgVariableUnits[kVtxNcontrib] = "";
  fgVariableNames[kVtxCovXX] = "Vtx covXX";
  fgVariableUnits[kVtxCovXX] = "cm";
  fgVariableNames[kVtxCovXY] = "Vtx covXY";
  fgVariableUnits[kVtxCovXY] = "cm";
  fgVariableNames[kVtxCovXZ] = "Vtx covXZ";
  fgVariableUnits[kVtxCovXZ] = "cm";
  fgVariableNames[kVtxCovYY] = "Vtx covYY";
  fgVariableUnits[kVtxCovYY] = "cm";
  fgVariableNames[kVtxCovYZ] = "Vtx covYZ";
  fgVariableUnits[kVtxCovYZ] = "cm";
  fgVariableNames[kVtxCovZZ] = "Vtx covZZ";
  fgVariableUnits[kVtxCovZZ] = "cm";
  fgVariableNames[kVtxChi2] = "Vtx chi2";
  fgVariableUnits[kVtxChi2] = "";
  fgVariableNames[kCentVZERO] = "Centrality VZERO";
  fgVariableUnits[kCentVZERO] = "%";
  fgVariableNames[kPt] = "p_{T}";
  fgVariableUnits[kPt] = "GeV/c";
  fgVariableNames[kEta] = "#eta";
  fgVariableUnits[kEta] = "";
  fgVariableNames[kPhi] = "#varphi";
  fgVariableUnits[kPhi] = "rad.";
  fgVariableNames[kRap] = "y";
  fgVariableUnits[kRap] = "";
  fgVariableNames[kMass] = "mass";
  fgVariableUnits[kMass] = "GeV/c2";
  fgVariableNames[kCharge] = "charge";
  fgVariableUnits[kCharge] = "";
  fgVariableNames[kPin] = "p_{IN}";
  fgVariableUnits[kPin] = "GeV/c";
  fgVariableNames[kITSncls] = "ITS #cls";
  fgVariableUnits[kITSncls] = "";
  fgVariableNames[kITSchi2] = "ITS chi2";
  fgVariableUnits[kITSchi2] = "";
  fgVariableNames[kITSlayerHit] = "ITS layer";
  fgVariableUnits[kITSlayerHit] = "";
  fgVariableNames[kTPCncls] = "TPC #cls";
  fgVariableUnits[kTPCncls] = "";
  fgVariableNames[kTPCchi2] = "TPC chi2";
  fgVariableUnits[kTPCchi2] = "";
  fgVariableNames[kTPCsignal] = "TPC dE/dx";
  fgVariableUnits[kTPCsignal] = "";
  fgVariableNames[kTRDsignal] = "TRD dE/dx";
  fgVariableUnits[kTRDsignal] = "";
  fgVariableNames[kTOFsignal] = "TOF signal";
  fgVariableUnits[kTOFsignal] = "";
  fgVariableNames[kTrackLength] = "track length";
  fgVariableUnits[kTrackLength] = "cm";
  fgVariableNames[kCandidateId] = "";
  fgVariableUnits[kCandidateId] = "";
  fgVariableNames[kPairType] = "Pair type";
  fgVariableUnits[kPairType] = "";
  fgVariableNames[kPairLxy] = "Pair Lxy";
  fgVariableUnits[kPairLxy] = "cm";
  fgVariableNames[kDeltaEta] = "#Delta#eta";
  fgVariableUnits[kDeltaEta] = "";
  fgVariableNames[kDeltaPhi] = "#Delta#phi";
  fgVariableUnits[kDeltaPhi] = "";
}
