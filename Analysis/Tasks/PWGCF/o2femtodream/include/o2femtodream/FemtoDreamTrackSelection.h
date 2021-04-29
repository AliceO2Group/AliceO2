// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FemtoDreamTrackCuts.h
/// \brief Definition of the FemtoDreamTrackCuts
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMTRACKSELECTION_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMTRACKSELECTION_H_

#include "ReconstructionDataFormats/PID.h"
//#include "Framework/HistogramRegistry.h"
//#include "Framework/Expressions.h"
#include <Rtypes.h>
#include <algorithm>
#include <string>
#include <cmath>
#include <iostream>

//using namespace o2::framework;
//using namespace o2::framework::expressions;

namespace o2::analysis
{
namespace femtoDream
{

/// \class FemtoDreamTrackCuts
/// \brief Cut class to contain and execute all cuts applied to tracks
class FemtoDreamTrackSelection
{
 public:
  FemtoDreamTrackSelection();
  FemtoDreamTrackSelection(int charge, float ptMin, float pTmax,
                           float etaMax, int tpcNcls, float tpcFcls,
                           int tpcNrows, bool tpcShareRej, float dcaXYMax,
                           float dcaZMax, float pidNsigmaMax, float pidTPCmom, o2::track::PID::ID part);
  virtual ~FemtoDreamTrackSelection() = default;

  /// Initialized histograms for the task
  /// \todo folder naming in case of two tracks per task - addon to the folder probably
  void init(); //HistogramRegistry* registry = nullptr);

  template <typename T>
  auto getNsigmaTPC(T const& track);

  template <typename T>
  auto getNsigmaTOF(T const& track);

  //const Filter AODFilter()
  //{
  //  return (o2::aod::track::pt > mPtMin) &&
  //         (o2::aod::track::pt < mPtMax) &&
  //         (o2::nabs(aod::track::eta) < mEtaMax) &&
  //         (nabs(o2::aod::track::dcaZ) < mDCAzMax);
  //};

  template <typename T>
  bool isSelected(T const& track);

  template <typename T>
  int getCutContainer(T const& track);

  template <typename T>
  void fillQA(T const& track);

  static std::string getCutHelp();
  void printCuts();

  void SetTPCCut(int i) { mTPCclsCut.push_back(i); }
  void SetTPCCut(std::vector<int> i) { mTPCclsCut = i; }

 private:
  std::vector<int> mTPCclsCut;

  int mTrackCharge;    ///< Charge of the track
  float mPtMin;        ///< Min. pT (GeV/c)
  float mPtMax;        ///< Max. pT (GeV/c)
  float mEtaMax;       ///< Max. eta
  int mTPCnClsMin;     ///< Min. TPC cluster
  float mTPCfClsMin;   ///< Min. TPC findable cluster fraction
  int mTPCcRowMin;     ///< Min. TPC crossed rows
  bool mTPCsClsRej;    ///< Shared cluster rejection
  float mDCAxyMax;     ///< Max. DCA to PV in xy (cm)
  float mDCAzMax;      ///< Max. DCA to PV in z (cm)
  float mPIDnSigmaMax; ///< Max. nSigma value;
  float mPIDmomTPC;    ///< Max. p for TPC-only PID (GeV/c)
  int mPIDParticle;    ///< Particle species to select

  //HistogramRegistry* mHistogramRegistry; ///< For QA output
  bool mDoQA;                            ///< Switch for protection

  ClassDefNV(FemtoDreamTrackSelection, 1);
};

template <typename T>
auto FemtoDreamTrackSelection::getNsigmaTPC(T const& track)
{
  switch (mPIDParticle) {
    case o2::track::PID::Electron:
      return track.tpcNSigmaEl();
      break;
    case o2::track::PID::Muon:
      return track.tpcNSigmaMu();
      break;
    case o2::track::PID::Pion:
      return track.tpcNSigmaPi();
      break;
    case o2::track::PID::Kaon:
      return track.tpcNSigmaKa();
      break;
    case o2::track::PID::Proton:
      return track.tpcNSigmaPr();
      break;
    case o2::track::PID::Deuteron:
      return track.tpcNSigmaDe();
      break;
    default:
      return 999.f;
      break;
  }
}

template <typename T>
auto FemtoDreamTrackSelection::getNsigmaTOF(T const& track)
{
  /// skip tracks without TOF signal
  /// \todo not sure what the error flags mean...
  if (track.tofSignal() <= 0.f || std::abs(track.tofSignal() - 99998) < 0.01 || std::abs(track.tofSignal() - 99999) < 0.01) {
    return 999.f;
  }

  switch (mPIDParticle) {
    case o2::track::PID::Electron:
      return track.tofNSigmaEl();
      break;
    case o2::track::PID::Muon:
      return track.tofNSigmaMu();
      break;
    case o2::track::PID::Pion:
      return track.tofNSigmaPi();
      break;
    case o2::track::PID::Kaon:
      return track.tofNSigmaKa();
      break;
    case o2::track::PID::Proton:
      return track.tofNSigmaPr();
      break;
    case o2::track::PID::Deuteron:
      return track.tofNSigmaDe();
      break;
    default:
      return 999.f;
      break;
  }
}

template <typename T>
inline bool FemtoDreamTrackSelection::isSelected(T const& track)
{
  if (track.sign() != mTrackCharge) {
    return false;
  }
  if (track.pt() < mPtMin || track.pt() > mPtMax) {
    return false;
  }
  if (std::abs(track.eta()) > mEtaMax) {
    return false;
  }
  if (track.tpcNClsFound() < mTPCclsCut[0]) {
    return false;
  }
  if (track.tpcCrossedRowsOverFindableCls() < mTPCfClsMin) {
    return false;
  }
  if (track.tpcNClsCrossedRows() < mTPCcRowMin) {
    return false;
  }
  if (mTPCsClsRej == true && track.tpcNClsShared() != 0) {
    return false;
  }

  const auto nSigmaTPC = getNsigmaTPC(track);
  if (track.tpcInnerParam() < mPIDmomTPC) {
    if (std::abs(nSigmaTPC) > mPIDnSigmaMax) {
      return false;
    }
  } else {
    const auto nSigmaTOF = getNsigmaTOF(track);
    const auto nSigmaComb = std::sqrt(nSigmaTPC * nSigmaTPC + nSigmaTOF * nSigmaTOF);
    if (std::abs(nSigmaComb) > mPIDnSigmaMax) {
      return false;
    }
  }
  if (std::abs(track.dcaZ()) > mDCAzMax) {
    return false;
  }

  if (mDoQA) {
    /// this needs to be done before the DCAxy cut, otherwise no template fitting
    //mHistogramRegistry->fill(HIST("TrackCuts/dcaXYhistBefore"), track.pt(), track.dcaXY());
  }

  if (std::abs(track.dcaXY()) > mDCAxyMax) {
    return false;
  }

  return true;
}

template <typename T>
int FemtoDreamTrackSelection::getCutContainer(T const& track)
{
  int output = 0;
  const auto nCls = track.tpcNClsFound();
  for (const auto it : mTPCclsCut) {
    if (nCls > it) {
      output = 1;
      // do something with the cut data type
    }
  }
  return output;
}

template <typename T>
inline void FemtoDreamTrackSelection::fillQA(T const& track)
{ /*
  if (mDoQA) {
    mHistogramRegistry->fill(HIST("TrackCuts/pThist"), track.pt());
    mHistogramRegistry->fill(HIST("TrackCuts/etahist"), track.eta());
    mHistogramRegistry->fill(HIST("TrackCuts/phihist"), track.phi());
    mHistogramRegistry->fill(HIST("TrackCuts/tpcnclshist"), track.tpcNClsFound());
    mHistogramRegistry->fill(HIST("TrackCuts/tpcfclshist"), track.tpcCrossedRowsOverFindableCls());
    mHistogramRegistry->fill(HIST("TrackCuts/tpcnrowshist"), track.tpcNClsCrossedRows());
    mHistogramRegistry->fill(HIST("TrackCuts/tpcnsharedhist"), track.tpcNClsShared());
    mHistogramRegistry->fill(HIST("TrackCuts/dcaXYhist"), track.pt(), track.dcaXY());
    mHistogramRegistry->fill(HIST("TrackCuts/dcaZhist"), track.pt(), track.dcaZ());
    mHistogramRegistry->fill(HIST("TrackCuts/tpcdEdx"), track.tpcInnerParam(), track.tpcSignal());
    mHistogramRegistry->fill(HIST("TrackCuts/tofSignal"), track.p(), track.tofSignal());
  } */
}

} // namespace femtoDream
} // namespace o2::analysis

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMTRACKSELECTION_H_ */
