// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//
// Class to handle analysis variables
//

#ifndef VarManager_H
#define VarManager_H

#include <TObject.h>
#include <TString.h>
#include <Math/Vector4D.h>
#include <TMath.h>

#include <vector>
#include <map>
#include <cmath>

#include "Framework/DataTypes.h"
#include "AnalysisCore/TriggerAliases.h"

// TODO: create an array holding these constants for all needed particles or check for a place where these are already defined
static const float fgkElectronMass = 0.000511; // GeV
static const float fgkMuonMass = 0.105;        // GeV

//_________________________________________________________________________
class VarManager : public TObject
{
 public:
  // map the information contained in the objects passed to the Fill functions
  enum ObjTypes {
    BC = BIT(0),
    Collision = BIT(1),
    CollisionCent = BIT(2),
    ReducedEvent = BIT(3),
    ReducedEventExtended = BIT(4),
    ReducedEventVtxCov = BIT(5),
    Track = BIT(0),
    TrackCov = BIT(1),
    TrackExtra = BIT(2),
    TrackPID = BIT(3),
    TrackDCA = BIT(4),
    TrackSelection = BIT(5),
    // TODO: Central model MUON variables to be added
    ReducedTrack = BIT(6),
    ReducedTrackBarrel = BIT(7),
    ReducedTrackBarrelCov = BIT(8),
    ReducedTrackBarrelPID = BIT(9),
    ReducedTrackMuon = BIT(10),
    Pair = BIT(11)
  };

  enum PairCandidateType {
    kJpsiToEE,   // J/psi        -> e+ e-
    kJpsiToMuMu, // J/psi        -> mu+ mu-
    kNMaxCandidateTypes
  };

 public:
  enum Variables {
    kNothing = -1,
    // Run wise variables
    kRunNo = 0,
    kRunId,
    kNRunWiseVariables,

    // Event wise variables   // Daria: imedjat ai ziua mja
    kCollisionTime,
    kBC,
    kIsPhysicsSelection,
    kIsINT7,
    kIsEMC7,
    kIsINT7inMUON,
    kIsMuonSingleLowPt7,
    kIsMuonUnlikeLowPt7,
    kIsMuonLikeLowPt7,
    kVtxX,
    kVtxY,
    kVtxZ,
    kVtxNcontrib,
    kVtxCovXX,
    kVtxCovXY,
    kVtxCovXZ,
    kVtxCovYY,
    kVtxCovYZ,
    kVtxCovZZ,
    kVtxChi2,
    kCentVZERO,
    kNEventWiseVariables,

    // Basic track(pair) wise variables
    kPt,
    kEta,
    kPhi,
    kP,
    kPx,
    kPy,
    kPz,
    kRap,
    kMass,
    kCharge,
    kNBasicTrackVariables,

    // Barrel track variables
    kPin,
    kIsGlobalTrack,
    kIsGlobalTrackSDD,
    kIsITSrefit,
    kIsSPDany,
    kIsSPDfirst,
    kIsSPDboth,
    kITSncls,
    kITSchi2,
    kITSlayerHit,
    kIsTPCrefit,
    kTPCncls,
    kTPCchi2,
    kTPCsignal,
    kTRDsignal,
    kTOFsignal,
    kTOFbeta,
    kTrackLength,
    kTrackDCAxy,
    kTrackDCAz,
    kTrackCYY,
    kTrackCZZ,
    kTrackCSnpSnp,
    kTrackCTglTgl,
    kTrackC1Pt21Pt2,
    kTPCnSigmaEl,
    kTPCnSigmaMu,
    kTPCnSigmaPi,
    kTPCnSigmaKa,
    kTPCnSigmaPr,
    kTOFnSigmaEl,
    kTOFnSigmaMu,
    kTOFnSigmaPi,
    kTOFnSigmaKa,
    kTOFnSigmaPr,
    kNBarrelTrackVariables,

    // Muon track variables
    kMuonInvBendingMomentum,
    kMuonThetaX,
    kMuonThetaY,
    kMuonZMu,
    kMuonBendingCoor,
    kMuonNonBendingCoor,
    kMuonRAtAbsorberEnd,
    kMuonPDca,
    kMuonChi2,
    kMuonChi2MatchTrigger,
    kNMuonTrackVariables,

    // Pair variables
    kCandidateId,
    kPairType,
    kPairLxy,
    kNPairVariables,

    // Candidate-track correlation variables
    kPairMass,
    kPairPt,
    kPairEta,
    kPairPhi,
    kDeltaEta,
    kDeltaPhi,
    kDeltaPhiSym,
    kNCorrelationVariables,

    kNVars
  }; // end of Variables enumeration

  static TString fgVariableNames[kNVars]; // variable names
  static TString fgVariableUnits[kNVars]; // variable units
  static void SetDefaultVarNames();

  static void SetUseVariable(int var)
  {
    if (var >= 0 && var < kNVars) {
      fgUsedVars[var] = kTRUE;
    }
    SetVariableDependencies();
  }
  static void SetUseVars(const bool* usedVars)
  {
    for (int i = 0; i < kNVars; ++i) {
      if (usedVars[i]) {
        fgUsedVars[i] = true; // overwrite only the variables that are being used since there are more channels to modify the used variables array, independently
      }
    }
    SetVariableDependencies();
  }
  static void SetUseVars(const std::vector<int> usedVars)
  {
    for (auto& var : usedVars) {
      fgUsedVars[var] = true;
    }
  }
  static bool GetUsedVar(int var)
  {
    if (var >= 0 && var < kNVars) {
      return fgUsedVars[var];
    }
    return false;
  }

  static void SetRunNumbers(int n, int* runs);
  static void SetRunNumbers(std::vector<int> runs);
  static int GetNRuns()
  {
    return fgRunMap.size();
  }
  static TString GetRunStr()
  {
    return fgRunStr;
  }

  template <uint32_t fillMap, typename T>
  static void FillEvent(T const& event, float* values = nullptr);
  template <uint32_t fillMap, typename T>
  static void FillTrack(T const& track, float* values = nullptr);
  template <typename T>
  static void FillPair(T const& t1, T const& t2, float* values = nullptr, PairCandidateType pairType = kJpsiToEE);
  template <typename T1, typename T2>
  static void FillDileptonHadron(T1 const& dilepton, T2 const& hadron, float* values = nullptr, float hadronMass = 0.0f);

 public:
  VarManager();
  ~VarManager() override;

  static float fgValues[kNVars]; // array holding all variables computed during analysis
  static void ResetValues(int startValue = 0, int endValue = kNVars, float* values = nullptr);

 private:
  static bool fgUsedVars[kNVars];        // holds flags for when the corresponding variable is needed (e.g., in the histogram manager, in cuts, mixing handler, etc.)
  static void SetVariableDependencies(); // toggle those variables on which other used variables might depend

  static std::map<int, int> fgRunMap; // map of runs to be used in histogram axes
  static TString fgRunStr;            // semi-colon separated list of runs, to be used for histogram axis labels

  static void FillEventDerived(float* values = nullptr);
  static void FillTrackDerived(float* values = nullptr);

  VarManager& operator=(const VarManager& c);
  VarManager(const VarManager& c);

  ClassDef(VarManager, 1)
};

template <uint32_t fillMap, typename T>
void VarManager::FillEvent(T const& event, float* values)
{
  if (!values) {
    values = fgValues;
  }

  if constexpr ((fillMap & BC) > 0) {
    values[kRunNo] = event.bc().runNumber(); // accessed via Collisions table
    values[kBC] = event.bc().globalBC();
  }

  if constexpr ((fillMap & Collision) > 0) {
    if (fgUsedVars[kIsINT7]) {
      values[kIsINT7] = (event.alias()[kINT7] > 0);
    }
    if (fgUsedVars[kIsEMC7]) {
      values[kIsEMC7] = (event.alias()[kEMC7] > 0);
    }
    if (fgUsedVars[kIsINT7inMUON]) {
      values[kIsINT7inMUON] = (event.alias()[kINT7inMUON] > 0);
    }
    if (fgUsedVars[kIsMuonSingleLowPt7]) {
      values[kIsMuonSingleLowPt7] = (event.alias()[kMuonSingleLowPt7] > 0);
    }
    if (fgUsedVars[kIsMuonUnlikeLowPt7]) {
      values[kIsMuonUnlikeLowPt7] = (event.alias()[kMuonUnlikeLowPt7] > 0);
    }
    if (fgUsedVars[kIsMuonLikeLowPt7]) {
      values[kIsMuonLikeLowPt7] = (event.alias()[kMuonLikeLowPt7] > 0);
    }
    values[kVtxX] = event.posX();
    values[kVtxY] = event.posY();
    values[kVtxZ] = event.posZ();
    values[kVtxNcontrib] = event.numContrib();
    values[kCollisionTime] = event.collisionTime();
    values[kVtxCovXX] = event.covXX();
    values[kVtxCovXY] = event.covXY();
    values[kVtxCovXZ] = event.covXZ();
    values[kVtxCovYY] = event.covYY();
    values[kVtxCovYZ] = event.covYZ();
    values[kVtxCovZZ] = event.covZZ();
    values[kVtxChi2] = event.chi2();
  }

  if constexpr ((fillMap & CollisionCent) > 0) {
    values[kCentVZERO] = event.centV0M();
  }

  // TODO: need to add EvSels and Cents tables, etc. in case of the central data model

  if constexpr ((fillMap & ReducedEvent) > 0) {
    values[kRunNo] = event.runNumber();
    values[kVtxX] = event.posX();
    values[kVtxY] = event.posY();
    values[kVtxZ] = event.posZ();
    values[kVtxNcontrib] = event.numContrib();
  }

  if constexpr ((fillMap & ReducedEventExtended) > 0) {
    values[kBC] = event.globalBC();
    values[kCentVZERO] = event.centV0M();
    if (fgUsedVars[kIsINT7]) {
      values[kIsINT7] = (event.triggerAlias() & (uint32_t(1) << kINT7)) > 0;
    }
    if (fgUsedVars[kIsEMC7]) {
      values[kIsEMC7] = (event.triggerAlias() & (uint32_t(1) << kEMC7)) > 0;
    }
    if (fgUsedVars[kIsINT7inMUON]) {
      values[kIsINT7inMUON] = (event.triggerAlias() & (uint32_t(1) << kINT7inMUON)) > 0;
    }
    if (fgUsedVars[kIsMuonSingleLowPt7]) {
      values[kIsMuonSingleLowPt7] = (event.triggerAlias() & (uint32_t(1) << kMuonSingleLowPt7)) > 0;
    }
    if (fgUsedVars[kIsMuonUnlikeLowPt7]) {
      values[kIsMuonUnlikeLowPt7] = (event.triggerAlias() & (uint32_t(1) << kMuonUnlikeLowPt7)) > 0;
    }
    if (fgUsedVars[kIsMuonLikeLowPt7]) {
      values[kIsMuonLikeLowPt7] = (event.triggerAlias() & (uint32_t(1) << kMuonLikeLowPt7)) > 0;
    }
  }

  if constexpr ((fillMap & ReducedEventVtxCov) > 0) {
    values[kVtxCovXX] = event.covXX();
    values[kVtxCovXY] = event.covXY();
    values[kVtxCovXZ] = event.covXZ();
    values[kVtxCovYY] = event.covYY();
    values[kVtxCovYZ] = event.covYZ();
    values[kVtxCovZZ] = event.covZZ();
    values[kVtxChi2] = event.chi2();
  }

  FillEventDerived(values);
}

template <uint32_t fillMap, typename T>
void VarManager::FillTrack(T const& track, float* values)
{
  if (!values) {
    values = fgValues;
  }

  if constexpr ((fillMap & Track) > 0 || (fillMap & ReducedTrack) > 0) {
    values[kPt] = track.pt();
    if (fgUsedVars[kPx]) {
      values[kPx] = track.px();
    }
    if (fgUsedVars[kPy]) {
      values[kPy] = track.py();
    }
    if (fgUsedVars[kPz]) {
      values[kPz] = track.pz();
    }
    values[kEta] = track.eta();
    values[kPhi] = track.phi();
    values[kCharge] = track.charge();
  }

  if constexpr ((fillMap & TrackExtra) > 0 || (fillMap & ReducedTrackBarrel) > 0) {
    values[kPin] = track.tpcInnerParam();
    if (fgUsedVars[kIsITSrefit]) {
      values[kIsITSrefit] = (track.flags() & (uint32_t(1) << 0)) > 0;
    }
    if (fgUsedVars[kIsTPCrefit]) {
      values[kIsTPCrefit] = (track.flags() & (uint32_t(1) << 1)) > 0;
    }
    if (fgUsedVars[kIsSPDfirst]) {
      values[kIsSPDfirst] = (track.itsClusterMap() & uint8_t(1)) > 0;
    }
    if (fgUsedVars[kIsSPDboth]) {
      values[kIsSPDboth] = (track.itsClusterMap() & uint8_t(3)) > 0;
    }
    if (fgUsedVars[kIsSPDany]) {
      values[kIsSPDany] = (track.itsClusterMap() & uint8_t(1)) || (track.itsClusterMap() & uint8_t(2));
    }
    values[kITSchi2] = track.itsChi2NCl();
    values[kTPCncls] = track.tpcNClsFound();
    values[kTPCchi2] = track.tpcChi2NCl();
    values[kTrackLength] = track.length();

    if constexpr ((fillMap & TrackExtra) > 0) {
      if (fgUsedVars[kITSncls]) {
        values[kITSncls] = track.itsNCls(); // dynamic column
      }
    }
    if constexpr ((fillMap & ReducedTrackBarrel) > 0) {
      if (fgUsedVars[kITSncls]) {
        values[kITSncls] = 0.0;
        for (int i = 0; i < 6; ++i) {
          values[kITSncls] += ((track.itsClusterMap() & (1 << i)) ? 1 : 0);
        }
      }
      values[kTrackDCAxy] = track.dcaXY();
      values[kTrackDCAz] = track.dcaZ();
    }
  }

  if constexpr ((fillMap & TrackDCA) > 0) {
    values[kTrackDCAxy] = track.dcaXY();
    values[kTrackDCAz] = track.dcaZ();
  }

  if constexpr ((fillMap & TrackSelection) > 0) {
    values[kIsGlobalTrack] = track.isGlobalTrack();
    values[kIsGlobalTrackSDD] = track.isGlobalTrackSDD();
  }

  if constexpr ((fillMap & TrackCov) > 0 || (fillMap & ReducedTrackBarrelCov) > 0) {
    values[kTrackCYY] = track.cYY();
    values[kTrackCZZ] = track.cZZ();
    values[kTrackCSnpSnp] = track.cSnpSnp();
    values[kTrackCTglTgl] = track.cTglTgl();
    values[kTrackC1Pt21Pt2] = track.c1Pt21Pt2();
  }

  if constexpr ((fillMap & TrackPID) > 0 || (fillMap & ReducedTrackBarrelPID) > 0) {
    values[kTPCnSigmaEl] = track.tpcNSigmaEl();
    values[kTPCnSigmaMu] = track.tpcNSigmaMu();
    values[kTPCnSigmaPi] = track.tpcNSigmaPi();
    values[kTPCnSigmaKa] = track.tpcNSigmaKa();
    values[kTPCnSigmaPr] = track.tpcNSigmaPr();
    values[kTOFnSigmaEl] = track.tofNSigmaEl();
    values[kTOFnSigmaMu] = track.tofNSigmaMu();
    values[kTOFnSigmaPi] = track.tofNSigmaPi();
    values[kTOFnSigmaKa] = track.tofNSigmaKa();
    values[kTOFnSigmaPr] = track.tofNSigmaPr();
    values[kTPCsignal] = track.tpcSignal();
    values[kTRDsignal] = track.trdSignal();
    values[kTOFsignal] = track.tofSignal();
    values[kTOFbeta] = track.beta();
  }

  if constexpr ((fillMap & ReducedTrackMuon) > 0) {
    values[kMuonInvBendingMomentum] = track.inverseBendingMomentum();
    values[kMuonThetaX] = track.thetaX();
    values[kMuonThetaY] = track.thetaY();
    values[kMuonZMu] = track.zMu();
    values[kMuonBendingCoor] = track.bendingCoor();
    values[kMuonNonBendingCoor] = track.nonBendingCoor();
    values[kMuonRAtAbsorberEnd] = track.rAtAbsorberEnd();
    values[kMuonPDca] = track.pDca();
    values[kMuonChi2] = track.chi2();
    values[kMuonChi2MatchTrigger] = track.chi2MatchTrigger();
  }

  if constexpr ((fillMap & Pair) > 0) {
    values[kMass] = track.mass();
  }

  FillTrackDerived(values);
}

template <typename T>
void VarManager::FillPair(T const& t1, T const& t2, float* values, PairCandidateType pairType)
{
  if (!values) {
    values = fgValues;
  }

  float m1 = fgkElectronMass;
  float m2 = fgkElectronMass;
  if (pairType == kJpsiToMuMu) {
    m1 = fgkMuonMass;
    m2 = fgkMuonMass;
  }

  ROOT::Math::PtEtaPhiMVector v1(t1.pt(), t1.eta(), t1.phi(), m1);
  ROOT::Math::PtEtaPhiMVector v2(t2.pt(), t2.eta(), t2.phi(), m2);
  ROOT::Math::PtEtaPhiMVector v12 = v1 + v2;
  values[kMass] = v12.M();
  values[kPt] = v12.Pt();
  values[kEta] = v12.Eta();
  values[kPhi] = v12.Phi();
  values[kRap] = -v12.Rapidity();
}

template <typename T1, typename T2>
void VarManager::FillDileptonHadron(T1 const& dilepton, T2 const& hadron, float* values, float hadronMass)
{
  if (!values) {
    values = fgValues;
  }

  if (fgUsedVars[kPairMass] || fgUsedVars[kPairPt] || fgUsedVars[kPairEta] || fgUsedVars[kPairPhi]) {
    ROOT::Math::PtEtaPhiMVector v1(dilepton.pt(), dilepton.eta(), dilepton.phi(), dilepton.mass());
    ROOT::Math::PtEtaPhiMVector v2(hadron.pt(), hadron.eta(), hadron.phi(), hadronMass);
    ROOT::Math::PtEtaPhiMVector v12 = v1 + v2;
    values[kPairMass] = v12.M();
    values[kPairPt] = v12.Pt();
    values[kPairEta] = v12.Eta();
    values[kPairPhi] = v12.Phi();
  }
  if (fgUsedVars[kDeltaPhi]) {
    double delta = dilepton.phi() - hadron.phi();
    if (delta > 3.0 / 2.0 * TMath::Pi()) {
      delta -= 2.0 * TMath::Pi();
    }
    if (delta < -0.5 * TMath::Pi()) {
      delta += 2.0 * TMath::Pi();
    }
    values[kDeltaPhi] = delta;
  }
  if (fgUsedVars[kDeltaPhiSym]) {
    double delta = TMath::Abs(dilepton.phi() - hadron.phi());
    if (delta > TMath::Pi()) {
      delta = 2 * TMath::Pi() - delta;
    }
    values[kDeltaPhiSym] = delta;
  }
  if (fgUsedVars[kDeltaEta]) {
    values[kDeltaEta] = dilepton.eta() - hadron.eta();
  }
}

#endif
