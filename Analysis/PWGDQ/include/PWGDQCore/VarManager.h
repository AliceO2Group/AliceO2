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
#include "Math/Vector3D.h"
#include "Math/GenVector/Boost.h"
#include <TRandom.h>

#include <vector>
#include <map>
#include <cmath>
#include <iostream>

#include "Framework/DataTypes.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "AnalysisCore/TriggerAliases.h"
#include "ReconstructionDataFormats/DCA.h"

using std::cout;
using std::endl;

// TODO: create an array holding these constants for all needed particles or check for a place where these are already defined
static const float fgkElectronMass = 0.000511; // GeV
static const float fgkMuonMass = 0.105;        // GeV

//_________________________________________________________________________
class VarManager : public TObject
{
 public:
  // map the information contained in the objects passed to the Fill functions
  enum ObjTypes {
    // NOTE: Elements containing "Reduced" in their name refer to skimmed data tables
    //       and the ones that don't refer to tables from the Framework data model
    BC = BIT(0),
    Collision = BIT(1),
    CollisionCent = BIT(2),
    CollisionTimestamp = BIT(3),
    ReducedEvent = BIT(4),
    ReducedEventExtended = BIT(5),
    ReducedEventVtxCov = BIT(6),
    Track = BIT(0),
    TrackCov = BIT(1),
    TrackExtra = BIT(2),
    TrackPID = BIT(3),
    TrackDCA = BIT(4),
    TrackSelection = BIT(5),
    ReducedTrack = BIT(6),
    ReducedTrackBarrel = BIT(7),
    ReducedTrackBarrelCov = BIT(8),
    ReducedTrackBarrelPID = BIT(9),
    Muon = BIT(10),
    MuonCov = BIT(11),
    ReducedMuon = BIT(12),
    ReducedMuonExtra = BIT(13),
    ReducedMuonCov = BIT(14),
    Pair = BIT(15)
  };

  enum PairCandidateType {
    // TODO: need to agree on a scheme to incorporate all various hypotheses (e.g. e - mu, jpsi - K+, Jpsi - pipi,...)
    kJpsiToEE = 0, // J/psi        -> e+ e-
    kJpsiToMuMu,   // J/psi        -> mu+ mu-
    kElectronMuon, // Electron - muon correlations
    kNMaxCandidateTypes
  };

 public:
  enum Variables {
    kNothing = -1,
    // Run wise variables
    kRunNo = 0,
    kRunId,
    kNRunWiseVariables,

    // Event wise variables
    kTimestamp,
    kBC,
    kIsPhysicsSelection,
    kIsINT7,
    kIsEMC7,
    kIsINT7inMUON,
    kIsMuonSingleLowPt7,
    kIsMuonSingleHighPt7,
    kIsMuonUnlikeLowPt7,
    kIsMuonLikeLowPt7,
    kIsCUP8,
    kIsCUP9,
    kIsMUP10,
    kIsMUP11,
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

    // Basic track/muon/pair wise variables
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
    kTPCnclsCR,
    kTPCchi2,
    kTPCsignal,
    kTPCsignalRandomized,
    kTPCsignalRandomizedDelta,
    kTRDsignal,
    kTRDPattern,
    kTOFbeta,
    kTrackLength,
    kTrackDCAxy,
    kTrackDCAz,
    kIsGoldenChi2,
    kTrackCYY,
    kTrackCZZ,
    kTrackCSnpSnp,
    kTrackCTglTgl,
    kTrackC1Pt21Pt2,
    kTPCnSigmaEl,
    kTPCnSigmaElRandomized,
    kTPCnSigmaElRandomizedDelta,
    kTPCnSigmaMu,
    kTPCnSigmaPi,
    kTPCnSigmaPiRandomized,
    kTPCnSigmaPiRandomizedDelta,
    kTPCnSigmaKa,
    kTPCnSigmaPr,
    kTPCnSigmaPrRandomized,
    kTPCnSigmaPrRandomizedDelta,
    kTOFnSigmaEl,
    kTOFnSigmaMu,
    kTOFnSigmaPi,
    kTOFnSigmaKa,
    kTOFnSigmaPr,
    kNBarrelTrackVariables,

    // Muon track variables
    kMuonNClusters,
    kMuonPDca,
    kMuonRAtAbsorberEnd,
    kMuonChi2,
    kMuonChi2MatchMCHMID,
    kMuonChi2MatchMCHMFT,
    kMuonMatchScoreMCHMFT,
    kMuonCXX,
    kMuonCYY,
    kMuonCPhiPhi,
    kMuonCTglTgl,
    kMuonC1Pt21Pt2,
    kNMuonTrackVariables,

    // Pair variables
    kCandidateId,
    kPairType,
    kVertexingLxy,
    kVertexingLxyErr,
    kVertexingLxyz,
    kVertexingLxyzErr,
    kVertexingProcCode,
    kVertexingChi2PCA,
    kCosThetaHE,
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

  // Setup the 2 prong DCAFitterN
  static void SetupTwoProngDCAFitter(float magField, bool propagateToPCA, float maxR, float maxDZIni, float minParamChange, float minRelChi2Change, bool useAbsDCA)
  {
    fgFitterTwoProng.setBz(magField);
    fgFitterTwoProng.setPropagateToPCA(propagateToPCA);
    fgFitterTwoProng.setMaxR(maxR);
    fgFitterTwoProng.setMaxDZIni(maxDZIni);
    fgFitterTwoProng.setMinParamChange(minParamChange);
    fgFitterTwoProng.setMinRelChi2Change(minRelChi2Change);
    fgFitterTwoProng.setUseAbsDCA(useAbsDCA);
  }

  template <uint32_t fillMap, typename T>
  static void FillEvent(T const& event, float* values = nullptr);
  template <uint32_t fillMap, typename T>
  static void FillTrack(T const& track, float* values = nullptr);
  template <typename T1, typename T2>
  static void FillPair(T1 const& t1, T2 const& t2, float* values = nullptr, PairCandidateType pairType = kJpsiToEE);
  template <typename C, typename T>
  static void FillPairVertexing(C const& collision, T const& t1, T const& t2, float* values = nullptr, PairCandidateType pairType = kJpsiToEE);
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

  template <typename T, typename U, typename V>
  static auto getRotatedCovMatrixXX(const T& matrix, U phi, V theta);

  static o2::vertexing::DCAFitterN<2> fgFitterTwoProng;

  VarManager& operator=(const VarManager& c);
  VarManager(const VarManager& c);

  ClassDef(VarManager, 1)
};

template <typename T, typename U, typename V>
auto VarManager::getRotatedCovMatrixXX(const T& matrix, U phi, V theta)
{
  auto cp = std::cos(phi);
  auto sp = std::sin(phi);
  auto ct = std::cos(theta);
  auto st = std::sin(theta);
  return matrix[0] * cp * cp * ct * ct        // covXX
         + matrix[1] * 2. * cp * sp * ct * ct // covXY
         + matrix[2] * sp * sp * ct * ct      // covYY
         + matrix[3] * 2. * cp * ct * st      // covXZ
         + matrix[4] * 2. * sp * ct * st      // covYZ
         + matrix[5] * st * st;               // covZZ
}

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

  if constexpr ((fillMap & CollisionTimestamp) > 0) {
    values[kTimestamp] = event.timestamp();
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
    if (fgUsedVars[kIsMuonSingleHighPt7]) {
      values[kIsMuonSingleHighPt7] = (event.alias()[kMuonSingleHighPt7] > 0);
    }
    if (fgUsedVars[kIsMuonUnlikeLowPt7]) {
      values[kIsMuonUnlikeLowPt7] = (event.alias()[kMuonUnlikeLowPt7] > 0);
    }
    if (fgUsedVars[kIsMuonLikeLowPt7]) {
      values[kIsMuonLikeLowPt7] = (event.alias()[kMuonLikeLowPt7] > 0);
    }
    if (fgUsedVars[kIsCUP8]) {
      values[kIsCUP8] = (event.alias()[kCUP8] > 0);
    }
    if (fgUsedVars[kIsCUP9]) {
      values[kIsCUP9] = (event.alias()[kCUP9] > 0);
    }
    if (fgUsedVars[kIsMUP10]) {
      values[kIsMUP10] = (event.alias()[kMUP10] > 0);
    }
    if (fgUsedVars[kIsMUP11]) {
      values[kIsMUP11] = (event.alias()[kMUP11] > 0);
    }
    values[kVtxX] = event.posX();
    values[kVtxY] = event.posY();
    values[kVtxZ] = event.posZ();
    values[kVtxNcontrib] = event.numContrib();
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
    values[kTimestamp] = event.timestamp();
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
    if (fgUsedVars[kIsMuonSingleHighPt7]) {
      values[kIsMuonSingleHighPt7] = (event.triggerAlias() & (uint32_t(1) << kMuonSingleHighPt7)) > 0;
    }
    if (fgUsedVars[kIsMuonUnlikeLowPt7]) {
      values[kIsMuonUnlikeLowPt7] = (event.triggerAlias() & (uint32_t(1) << kMuonUnlikeLowPt7)) > 0;
    }
    if (fgUsedVars[kIsMuonLikeLowPt7]) {
      values[kIsMuonLikeLowPt7] = (event.triggerAlias() & (uint32_t(1) << kMuonLikeLowPt7)) > 0;
    }
    if (fgUsedVars[kIsCUP8]) {
      values[kIsCUP8] = (event.triggerAlias() & (uint32_t(1) << kCUP8)) > 0;
    }
    if (fgUsedVars[kIsCUP9]) {
      values[kIsCUP9] = (event.triggerAlias() & (uint32_t(1) << kCUP9)) > 0;
    }
    if (fgUsedVars[kIsMUP10]) {
      values[kIsMUP10] = (event.triggerAlias() & (uint32_t(1) << kMUP10)) > 0;
    }
    if (fgUsedVars[kIsMUP11]) {
      values[kIsMUP11] = (event.triggerAlias() & (uint32_t(1) << kMUP11)) > 0;
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

  // Quantities based on the basic table (contains just kine information and filter bits)
  if constexpr ((fillMap & Track) > 0 || (fillMap & Muon) > 0 || (fillMap & ReducedTrack) > 0 || (fillMap & ReducedMuon) > 0) {
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
    values[kCharge] = track.sign();

    if constexpr ((fillMap & ReducedTrack) > 0 && !((fillMap & Pair) > 0)) {
      values[kIsGlobalTrack] = track.filteringFlags() & (uint64_t(1) << 0);
      values[kIsGlobalTrackSDD] = track.filteringFlags() & (uint64_t(1) << 1);
    }
  }

  // Quantities based on the barrel tables
  if constexpr ((fillMap & TrackExtra) > 0 || (fillMap & ReducedTrackBarrel) > 0) {
    values[kPin] = track.tpcInnerParam();
    if (fgUsedVars[kIsITSrefit]) {
      values[kIsITSrefit] = (track.flags() & o2::aod::track::ITSrefit) > 0;
    }
    if (fgUsedVars[kIsTPCrefit]) {
      values[kIsTPCrefit] = (track.flags() & o2::aod::track::TPCrefit) > 0;
    }
    if (fgUsedVars[kIsGoldenChi2]) {
      values[kIsGoldenChi2] = (track.flags() & o2::aod::track::GoldenChi2) > 0;
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
    values[kTPCnclsCR] = track.tpcNClsCrossedRows();
    values[kTRDPattern] = track.trdPattern();

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

  // Quantities based on the barrel track selection table
  if constexpr ((fillMap & TrackDCA) > 0) {
    values[kTrackDCAxy] = track.dcaXY();
    values[kTrackDCAz] = track.dcaZ();
  }

  // Quantities based on the barrel track selection table
  if constexpr ((fillMap & TrackSelection) > 0) {
    values[kIsGlobalTrack] = track.isGlobalTrack();
    values[kIsGlobalTrackSDD] = track.isGlobalTrackSDD();
  }

  // Quantities based on the barrel covariance tables
  if constexpr ((fillMap & TrackCov) > 0 || (fillMap & ReducedTrackBarrelCov) > 0) {
    values[kTrackCYY] = track.cYY();
    values[kTrackCZZ] = track.cZZ();
    values[kTrackCSnpSnp] = track.cSnpSnp();
    values[kTrackCTglTgl] = track.cTglTgl();
    values[kTrackC1Pt21Pt2] = track.c1Pt21Pt2();
  }

  // Quantities based on the barrel PID tables
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
    values[kTOFbeta] = track.beta();
    if (fgUsedVars[kTPCsignalRandomized] || fgUsedVars[kTPCnSigmaElRandomized] || fgUsedVars[kTPCnSigmaPiRandomized] || fgUsedVars[kTPCnSigmaPrRandomized]) {
      // NOTE: this is needed temporarilly for the study of the impact of TPC pid degradation on the quarkonium triggers in high lumi pp
      //     This study involves a degradation from a dE/dx resolution of 5% to one of 6% (20% worsening)
      //     For this we smear the dE/dx and n-sigmas using a gaus distribution with a width of 3.3%
      //         which is approx the needed amount to get dE/dx to a resolution of 6%
      double randomX = gRandom->Gaus(0.0, 0.033);
      values[kTPCsignalRandomized] = values[kTPCsignal] * (1.0 + randomX);
      values[kTPCsignalRandomizedDelta] = values[kTPCsignal] * randomX;
      values[kTPCnSigmaElRandomized] = values[kTPCnSigmaEl] * (1.0 + randomX);
      values[kTPCnSigmaElRandomizedDelta] = values[kTPCnSigmaEl] * randomX;
      values[kTPCnSigmaPiRandomized] = values[kTPCnSigmaPi] * (1.0 + randomX);
      values[kTPCnSigmaPiRandomizedDelta] = values[kTPCnSigmaPi] * randomX;
      values[kTPCnSigmaPrRandomized] = values[kTPCnSigmaPr] * (1.0 + randomX);
      values[kTPCnSigmaPrRandomizedDelta] = values[kTPCnSigmaPr] * randomX;
    }
  }

  // Quantities based on the muon extra table
  if constexpr ((fillMap & ReducedMuonExtra) > 0 || (fillMap & Muon) > 0) {
    values[kMuonNClusters] = track.nClusters();
    values[kMuonPDca] = track.pDca();
    values[kMuonRAtAbsorberEnd] = track.rAtAbsorberEnd();
    values[kMuonChi2] = track.chi2();
    values[kMuonChi2MatchMCHMID] = track.chi2MatchMCHMID();
    values[kMuonChi2MatchMCHMFT] = track.chi2MatchMCHMFT();
    values[kMuonMatchScoreMCHMFT] = track.matchScoreMCHMFT();
  }
  // Quantities based on the muon covariance table
  if constexpr ((fillMap & ReducedMuonCov) > 0 || (fillMap & MuonCov) > 0) {
    values[kMuonCXX] = track.cXX();
    values[kMuonCYY] = track.cYY();
    values[kMuonCPhiPhi] = track.cPhiPhi();
    values[kMuonCTglTgl] = track.cTglTgl();
    values[kMuonC1Pt21Pt2] = track.c1Pt21Pt2();
  }

  // Quantities based on the pair table(s)
  if constexpr ((fillMap & Pair) > 0) {
    values[kMass] = track.mass();
  }

  // Derived quantities which can be computed based on already filled variables
  FillTrackDerived(values);
}

template <typename T1, typename T2>
void VarManager::FillPair(T1 const& t1, T2 const& t2, float* values, PairCandidateType pairType)
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

  if (pairType == kElectronMuon) {
    m1 = fgkElectronMass;
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
  // CosTheta Helicity calculation
  ROOT::Math::Boost boostv12{v12.BoostToCM()};
  ROOT::Math::XYZVectorF v1_CM{(boostv12(v1).Vect()).Unit()};
  ROOT::Math::XYZVectorF v2_CM{(boostv12(v2).Vect()).Unit()};
  ROOT::Math::XYZVectorF zaxis{(v12.Vect()).Unit()};

  double cosTheta = 0;
  if (t1.sign() > 0) {
    cosTheta = zaxis.Dot(v1_CM);
  } else {
    cosTheta = zaxis.Dot(v2_CM);
  }
  values[kCosThetaHE] = cosTheta;
}

template <typename C, typename T>
void VarManager::FillPairVertexing(C const& collision, T const& t1, T const& t2, float* values, PairCandidateType pairType)
{
  if (!values) {
    values = fgValues;
  }

  // TODO: use trackUtilities functions to initialize the various matrices to avoid code duplication
  //auto pars1 = getTrackParCov(t1);
  //auto pars2 = getTrackParCov(t2);
  std::array<float, 5> t1pars = {t1.y(), t1.z(), t1.snp(), t1.tgl(), t1.signed1Pt()};
  std::array<float, 15> t1covs = {t1.cYY(), t1.cZY(), t1.cZZ(), t1.cSnpY(), t1.cSnpZ(),
                                  t1.cSnpSnp(), t1.cTglY(), t1.cTglZ(), t1.cTglSnp(), t1.cTglTgl(),
                                  t1.c1PtY(), t1.c1PtZ(), t1.c1PtSnp(), t1.c1PtTgl(), t1.c1Pt21Pt2()};
  o2::track::TrackParCov pars1{t1.x(), t1.alpha(), t1pars, t1covs};
  std::array<float, 5> t2pars = {t2.y(), t2.z(), t2.snp(), t2.tgl(), t2.signed1Pt()};
  std::array<float, 15> t2covs = {t2.cYY(), t2.cZY(), t2.cZZ(), t2.cSnpY(), t2.cSnpZ(),
                                  t2.cSnpSnp(), t2.cTglY(), t2.cTglZ(), t2.cTglSnp(), t2.cTglTgl(),
                                  t2.c1PtY(), t2.c1PtZ(), t2.c1PtSnp(), t2.c1PtTgl(), t2.c1Pt21Pt2()};
  o2::track::TrackParCov pars2{t2.x(), t2.alpha(), t2pars, t2covs};

  // reconstruct the 2-prong secondary vertex
  int procCode = fgFitterTwoProng.process(pars1, pars2);
  values[kVertexingProcCode] = procCode;
  if (procCode == 0) {
    // TODO: set the other variables to appropriate values and return
    values[kVertexingChi2PCA] = -999.;
    values[kVertexingLxy] = -999.;
    values[kVertexingLxyz] = -999.;
    values[kVertexingLxyErr] = -999.;
    values[kVertexingLxyzErr] = -999.;
    return;
  }

  const auto& secondaryVertex = fgFitterTwoProng.getPCACandidate();
  auto chi2PCA = fgFitterTwoProng.getChi2AtPCACandidate();
  auto covMatrixPCA = fgFitterTwoProng.calcPCACovMatrix().Array();
  auto trackParVar0 = fgFitterTwoProng.getTrack(0);
  auto trackParVar1 = fgFitterTwoProng.getTrack(1);
  values[kVertexingChi2PCA] = chi2PCA;

  // get track momenta
  std::array<float, 3> pvec0;
  std::array<float, 3> pvec1;
  trackParVar0.getPxPyPzGlo(pvec0);
  trackParVar1.getPxPyPzGlo(pvec1);

  // get track impact parameters
  // This modifies track momenta!
  o2::math_utils::Point3D<float> vtxXYZ(collision.posX(), collision.posY(), collision.posZ());
  std::array<float, 6> vtxCov{collision.covXX(), collision.covXY(), collision.covYY(), collision.covXZ(), collision.covYZ(), collision.covZZ()};
  o2::dataformats::VertexBase primaryVertex = {std::move(vtxXYZ), std::move(vtxCov)};
  //auto primaryVertex = getPrimaryVertex(collision);
  auto covMatrixPV = primaryVertex.getCov();
  o2::dataformats::DCA impactParameter0;
  o2::dataformats::DCA impactParameter1;
  trackParVar0.propagateToDCA(primaryVertex, fgFitterTwoProng.getBz(), &impactParameter0);
  trackParVar1.propagateToDCA(primaryVertex, fgFitterTwoProng.getBz(), &impactParameter1);

  // get uncertainty of the decay length
  //double phi, theta;
  //getPointDirection(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex, phi, theta);
  double phi = std::atan2(secondaryVertex[1] - collision.posY(), secondaryVertex[0] - collision.posX());
  double theta = std::atan2(secondaryVertex[2] - collision.posZ(),
                            std::sqrt((secondaryVertex[0] - collision.posX()) * (secondaryVertex[0] - collision.posX()) +
                                      (secondaryVertex[1] - collision.posY()) * (secondaryVertex[1] - collision.posY())));

  values[kVertexingLxyzErr] = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, theta) + getRotatedCovMatrixXX(covMatrixPCA, phi, theta));
  values[kVertexingLxyErr] = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, 0.) + getRotatedCovMatrixXX(covMatrixPCA, phi, 0.));

  values[kVertexingLxy] = (collision.posX() - secondaryVertex[0]) * (collision.posX() - secondaryVertex[0]) +
                          (collision.posY() - secondaryVertex[1]) * (collision.posY() - secondaryVertex[1]);
  values[kVertexingLxyz] = values[kVertexingLxy] + (collision.posZ() - secondaryVertex[2]) * (collision.posZ() - secondaryVertex[2]);
  values[kVertexingLxy] = std::sqrt(values[kVertexingLxy]);
  values[kVertexingLxyz] = std::sqrt(values[kVertexingLxyz]);
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
    if (delta > 3.0 / 2.0 * M_PI) {
      delta -= 2.0 * M_PI;
    }
    if (delta < -0.5 * M_PI) {
      delta += 2.0 * M_PI;
    }
    values[kDeltaPhi] = delta;
  }
  if (fgUsedVars[kDeltaPhiSym]) {
    double delta = std::abs(dilepton.phi() - hadron.phi());
    if (delta > M_PI) {
      delta = 2 * M_PI - delta;
    }
    values[kDeltaPhiSym] = delta;
  }
  if (fgUsedVars[kDeltaEta]) {
    values[kDeltaEta] = dilepton.eta() - hadron.eta();
  }
}
#endif
