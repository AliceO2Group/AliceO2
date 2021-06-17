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

#include "FemtoDreamObjectSelection.h"

#include "ReconstructionDataFormats/PID.h"
#include "Framework/HistogramRegistry.h"
#include <Rtypes.h>
#include <cmath>

using namespace o2::framework;

namespace o2::analysis
{
namespace femtoDream
{
namespace femtoDreamTrackSelection
{
enum TrackSel { kSign,
                kpTMin,
                kpTMax,
                kEtaMax,
                kTPCnClsMin,
                kTPCfClsMin,
                kTPCcRowsMin,
                kTPCsClsMax,
                kDCAxyMax,
                kDCAzMax,
                kPIDnSigmaMax };
}

/// \class FemtoDreamTrackCuts
/// \brief Cut class to contain and execute all cuts applied to tracks
class FemtoDreamTrackSelection : public FemtoDreamObjectSelection<float, femtoDreamTrackSelection::TrackSel>
{
 public:
  /// Initializes histograms for the task
  void init(HistogramRegistry* registry);

  template <typename T>
  void setPIDSpecies(T& pids)
  {
    std::vector<int> tmpPids = pids; // necessary due to some features of the configurable
    for (const o2::track::PID& pid : tmpPids) {
      mPIDspecies.push_back(pid);
    }
  }

  template <typename T>
  auto getNsigmaTPC(T const& track, o2::track::PID pid);

  template <typename T>
  auto getNsigmaTOF(T const& track, o2::track::PID pid);

  template <typename T>
  bool isSelectedMinimal(T const& track);

  template <typename T>
  uint64_t getCutContainer(T const& track);

  template <typename T>
  void fillQA(T const& track);

  template <typename T>
  void fillCutQA(T const& track, uint64_t cutContainer);

 private:
  std::vector<o2::track::PID> mPIDspecies;

  ClassDefNV(FemtoDreamTrackSelection, 1);
}; // namespace femtoDream

void FemtoDreamTrackSelection::init(HistogramRegistry* registry)
{
  if (registry) {
    mHistogramRegistry = registry;
    fillSelectionHistogram("TrackCuts/cuthist");

    /// \todo this should be an automatic check in the parent class, and the return type should be templated
    int nSelections = 2 + getNSelections() + mPIDspecies.size() * (getNSelections(femtoDreamTrackSelection::kPIDnSigmaMax) - 1);
    if (8 * sizeof(uint64_t) < nSelections) {
      LOGF(error, "Number of selections to large for your container - quitting!");
    }

    mHistogramRegistry->add("TrackCuts/pThist", "; #it{p}_{T} (GeV/#it{c}); Entries", kTH1F, {{1000, 0, 10}});
    mHistogramRegistry->add("TrackCuts/etahist", "; #eta; Entries", kTH1F, {{1000, -1, 1}});
    mHistogramRegistry->add("TrackCuts/phihist", "; #phi; Entries", kTH1F, {{1000, 0, 2. * M_PI}});
    mHistogramRegistry->add("TrackCuts/tpcnclshist", "; TPC Cluster; Entries", kTH1F, {{163, 0, 163}});
    mHistogramRegistry->add("TrackCuts/tpcfclshist", "; TPC ratio findable; Entries", kTH1F, {{100, 0.5, 1.5}});
    mHistogramRegistry->add("TrackCuts/tpcnrowshist", "; TPC crossed rows; Entries", kTH1F, {{163, 0, 163}});
    mHistogramRegistry->add("TrackCuts/tpcnsharedhist", "; TPC shared clusters; Entries", kTH1F, {{163, 0, 163}});
    mHistogramRegistry->add("TrackCuts/dcaXYhist", "; #it{p}_{T} (GeV/#it{c}); DCA_{xy} (cm)", kTH2F, {{100, 0, 10}, {301, -1.5, 1.5}});
    mHistogramRegistry->add("TrackCuts/dcaZhist", "; #it{p}_{T} (GeV/#it{c}); DCA_{z} (cm)", kTH2F, {{100, 0, 10}, {301, -1.5, 1.5}});
    mHistogramRegistry->add("TrackCuts/tpcdEdx", "; #it{p} (GeV/#it{c}); TPC Signal", kTH2F, {{100, 0, 10}, {1000, 0, 1000}});
    mHistogramRegistry->add("TrackCuts/tofSignal", "; #it{p} (GeV/#it{c}); TOF Signal", kTH2F, {{100, 0, 10}, {1000, 0, 100e3}});

    const int nChargeSel = getNSelections(femtoDreamTrackSelection::kSign);
    const int nPtMinSel = getNSelections(femtoDreamTrackSelection::kpTMin);
    const int nPtMaxSel = getNSelections(femtoDreamTrackSelection::kpTMax);
    const int nEtaMaxSel = getNSelections(femtoDreamTrackSelection::kEtaMax);
    const int nTPCnMinSel = getNSelections(femtoDreamTrackSelection::kTPCnClsMin);
    const int nTPCfMinSel = getNSelections(femtoDreamTrackSelection::kTPCfClsMin);
    const int nTPCcMinSel = getNSelections(femtoDreamTrackSelection::kTPCcRowsMin);
    const int nTPCsMaxSel = getNSelections(femtoDreamTrackSelection::kTPCsClsMax);
    const int nDCAxyMaxSel = getNSelections(femtoDreamTrackSelection::kDCAxyMax);
    const int nDCAzMaxSel = getNSelections(femtoDreamTrackSelection::kDCAzMax);
    const int nPIDnSigmaSel = getNSelections(femtoDreamTrackSelection::kPIDnSigmaMax);

    if (nChargeSel > 0) {
      mHistogramRegistry->add("TrackCutsQA/Charge", "; Cut; Charge", kTH2F, {{nChargeSel, 0, static_cast<double>(nChargeSel)}, {3, -1.5, 1.5}});
    }
    if (nPtMinSel > 0) {
      mHistogramRegistry->add("TrackCutsQA/pTMin", "; Cut; #it{p}_{T} (GeV/#it{c})", kTH2F, {{nPtMinSel, 0, static_cast<double>(nPtMinSel)}, {1000, 0, 10}});
    }
    if (nPtMaxSel > 0) {
      mHistogramRegistry->add("TrackCutsQA/pTMax", "; Cut; #it{p}_{T} (GeV/#it{c})", kTH2F, {{nPtMaxSel, 0, static_cast<double>(nPtMaxSel)}, {1000, 0, 10}});
    }
    if (nEtaMaxSel > 0) {
      mHistogramRegistry->add("TrackCutsQA/etaMax", "; Cut; #eta", kTH2F, {{nEtaMaxSel, 0, static_cast<double>(nEtaMaxSel)}, {1000, -1, 1}});
    }
    if (nTPCnMinSel > 0) {
      mHistogramRegistry->add("TrackCutsQA/tpcnClsMin", "; Cut; TPC Cluster", kTH2F, {{nTPCnMinSel, 0, static_cast<double>(nTPCnMinSel)}, {163, 0, 163}});
    }
    if (nTPCfMinSel > 0) {
      mHistogramRegistry->add("TrackCutsQA/tpcfClsMin", "; Cut; TPC ratio findable", kTH2F, {{nTPCfMinSel, 0, static_cast<double>(nTPCfMinSel)}, {100, 0.5, 1.5}});
    }
    if (nTPCcMinSel > 0) {
      mHistogramRegistry->add("TrackCutsQA/tpcnRowsMin", "; Cut; TPC crossed rows", kTH2F, {{nTPCcMinSel, 0, static_cast<double>(nTPCcMinSel)}, {163, 0, 163}});
    }
    if (nTPCsMaxSel > 0) {
      mHistogramRegistry->add("TrackCutsQA/tpcnSharedMax", "; Cut; TPC shared clusters", kTH2F, {{nTPCsMaxSel, 0, static_cast<double>(nTPCsMaxSel)}, {163, 0, 163}});
    }
    if (nDCAxyMaxSel > 0) {
      mHistogramRegistry->add("TrackCutsQA/dcaXYMax", "; Cut; DCA_{xy} (cm)", kTH2F, {{nDCAxyMaxSel, 0, static_cast<double>(nDCAxyMaxSel)}, {51, -3, 3}});
    }
    if (nDCAzMaxSel > 0) {
      mHistogramRegistry->add("TrackCutsQA/dcaZMax", "; Cut; DCA_{z} (cm)", kTH2F, {{nDCAzMaxSel, 0, static_cast<double>(nDCAzMaxSel)}, {51, -1.5, 1.5}});
    }
    if (nPIDnSigmaSel > 0) {
      int nSpecies = mPIDspecies.size();
      mHistogramRegistry->add("TrackCutsQA/pidCombnsigmaMax", "; #it{n}_{#sigma, comb.}; Cut", kTH2F, {{nPIDnSigmaSel * nSpecies, 0, static_cast<double>(nSpecies * nPIDnSigmaSel)}, {60, 0, 6}});
      mHistogramRegistry->add("TrackCutsQA/pidTPCnsigmaMax", "; #it{n}_{#sigma, TPC}; Cut", kTH2F, {{nPIDnSigmaSel * nSpecies, 0, static_cast<double>(nSpecies * nPIDnSigmaSel)}, {121, -6, 6}});
    }
  }
}

template <typename T>
auto FemtoDreamTrackSelection::getNsigmaTPC(T const& track, o2::track::PID pid)
{
  switch (pid) {
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
auto FemtoDreamTrackSelection::getNsigmaTOF(T const& track, o2::track::PID pid)
{
  /// skip tracks without TOF signal
  /// \todo not sure what the error flags mean...
  if (track.tofSignal() <= 0.f || std::abs(track.tofSignal() - 99998) < 0.01 || std::abs(track.tofSignal() - 99999) < 0.01) {
    return 999.f;
  }

  switch (pid) {
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
bool FemtoDreamTrackSelection::isSelectedMinimal(T const& track)
{
  const auto pT = track.pt();
  const auto eta = track.eta();
  const auto tpcNClsF = track.tpcNClsFound();
  const auto tpcRClsC = track.tpcCrossedRowsOverFindableCls();
  const auto tpcNClsC = track.tpcNClsCrossedRows();
  const auto tpcNClsS = track.tpcNClsShared();
  const auto dcaXY = track.dcaXY();
  const auto dcaZ = track.dcaZ();

  /// check whether the most open cuts are fulfilled - most of this should have already be done by the filters

  const static int nPtMinSel = getNSelections(femtoDreamTrackSelection::kpTMin);
  const static int nPtMaxSel = getNSelections(femtoDreamTrackSelection::kpTMax);
  const static int nEtaSel = getNSelections(femtoDreamTrackSelection::kEtaMax);
  const static int nTPCnMinSel = getNSelections(femtoDreamTrackSelection::kTPCnClsMin);
  const static int nTPCfMinSel = getNSelections(femtoDreamTrackSelection::kTPCfClsMin);
  const static int nTPCcMinSel = getNSelections(femtoDreamTrackSelection::kTPCcRowsMin);
  const static int nTPCsMaxSel = getNSelections(femtoDreamTrackSelection::kTPCsClsMax);
  const static int nDCAxyMaxSel = getNSelections(femtoDreamTrackSelection::kDCAxyMax);
  const static int nDCAzMaxSel = getNSelections(femtoDreamTrackSelection::kDCAzMax);

  const static float pTMin = getMinimalSelection(femtoDreamTrackSelection::kpTMin, femtoDreamSelection::kLowerLimit);
  const static float pTMax = getMinimalSelection(femtoDreamTrackSelection::kpTMax, femtoDreamSelection::kUpperLimit);
  const static float etaMax = getMinimalSelection(femtoDreamTrackSelection::kEtaMax, femtoDreamSelection::kAbsUpperLimit);
  const static float nClsMin = getMinimalSelection(femtoDreamTrackSelection::kTPCnClsMin, femtoDreamSelection::kLowerLimit);
  const static float fClsMin = getMinimalSelection(femtoDreamTrackSelection::kTPCfClsMin, femtoDreamSelection::kLowerLimit);
  const static float cTPCMin = getMinimalSelection(femtoDreamTrackSelection::kTPCcRowsMin, femtoDreamSelection::kLowerLimit);
  const static float sTPCMax = getMinimalSelection(femtoDreamTrackSelection::kTPCsClsMax, femtoDreamSelection::kUpperLimit);
  const static float dcaXYMax = getMinimalSelection(femtoDreamTrackSelection::kDCAxyMax, femtoDreamSelection::kAbsUpperLimit);
  const static float dcaZMax = getMinimalSelection(femtoDreamTrackSelection::kDCAzMax, femtoDreamSelection::kAbsUpperLimit);

  if (nPtMinSel > 0 && pT < pTMin) {
    return false;
  }
  if (nPtMaxSel > 0 && pT > pTMax) {
    return false;
  }
  if (nEtaSel > 0 && std::abs(eta) > etaMax) {
    return false;
  }
  if (nTPCnMinSel > 0 && tpcNClsF < nClsMin) {
    return false;
  }
  if (nTPCfMinSel > 0 && tpcRClsC < fClsMin) {
    return false;
  }
  if (nTPCcMinSel > 0 && tpcNClsC < cTPCMin) {
    return false;
  }
  if (nTPCsMaxSel > 0 && tpcNClsS > sTPCMax) {
    return false;
  }
  if (nDCAxyMaxSel > 0 && std::abs(dcaXY) > dcaXYMax) {
    return false;
  }
  if (nDCAzMaxSel > 0 && std::abs(dcaZ) > dcaZMax) {
    return false;
  }
  return true;
}

template <typename T>
uint64_t FemtoDreamTrackSelection::getCutContainer(T const& track)
{
  uint64_t output = 0;
  size_t counter = 2; // first two slots reserved for track/v0/cascade encoding
  const auto sign = track.sign();
  const auto pT = track.pt();
  const auto eta = track.eta();
  const auto tpcNClsF = track.tpcNClsFound();
  const auto tpcRClsC = track.tpcCrossedRowsOverFindableCls();
  const auto tpcNClsC = track.tpcNClsCrossedRows();
  const auto tpcNClsS = track.tpcNClsShared();
  const auto dcaXY = track.dcaXY();
  const auto dcaZ = track.dcaZ();
  std::vector<float> pidTPC, pidTOF;
  for (auto it : mPIDspecies) {
    pidTPC.push_back(getNsigmaTPC(track, it));
    pidTOF.push_back(getNsigmaTOF(track, it));
  }

  float observable;
  for (auto& sel : mSelections) {
    const auto selVariable = sel.getSelectionVariable();
    if (selVariable == femtoDreamTrackSelection::kPIDnSigmaMax) {
      /// PID needs to be handled a bit differently since we may need more than one species
      for (size_t i = 0; i < pidTPC.size(); ++i) {
        auto pidTPCVal = pidTPC.at(i);
        auto pidTOFVal = pidTOF.at(i);
        sel.checkSelectionSetBit(pidTPCVal, output, counter);
        auto pidComb = std::sqrt(pidTPCVal * pidTPCVal + pidTOFVal * pidTOFVal);
        sel.checkSelectionSetBit(pidComb, output, counter);
      }

    } else {
      /// for the rest it's all the same
      switch (selVariable) {
        case (femtoDreamTrackSelection::kSign):
          observable = sign;
          break;
        case (femtoDreamTrackSelection::kpTMin):
        case (femtoDreamTrackSelection::kpTMax):
          observable = pT;
          break;
        case (femtoDreamTrackSelection::kEtaMax):
          observable = eta;
          break;
        case (femtoDreamTrackSelection::kTPCnClsMin):
          observable = tpcNClsF;
          break;
        case (femtoDreamTrackSelection::kTPCfClsMin):
          observable = tpcRClsC;
          break;
        case (femtoDreamTrackSelection::kTPCcRowsMin):
          observable = tpcNClsC;
          break;
        case (femtoDreamTrackSelection::kTPCsClsMax):
          observable = tpcNClsS;
          break;
        case (femtoDreamTrackSelection::kDCAxyMax):
          observable = dcaXY;
          break;
        case (femtoDreamTrackSelection::kDCAzMax):
          observable = dcaZ;
          break;
        case (femtoDreamTrackSelection::kPIDnSigmaMax):
          break;
      }
      sel.checkSelectionSetBit(observable, output, counter);
    }
  }
  return output;
}

template <typename T>
void FemtoDreamTrackSelection::fillQA(T const& track)
{
  if (mHistogramRegistry) {
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
  }
}

template <typename T>
void FemtoDreamTrackSelection::fillCutQA(T const& track, uint64_t cutContainer)
{
  if (mHistogramRegistry) {
    size_t counter = 2; // first two slots reserved for track/v0/cascade encoding
    const auto sign = track.sign();
    const auto pT = track.pt();
    const auto eta = track.eta();
    const auto tpcNClsF = track.tpcNClsFound();
    const auto tpcRClsC = track.tpcCrossedRowsOverFindableCls();
    const auto tpcNClsC = track.tpcNClsCrossedRows();
    const auto tpcNClsS = track.tpcNClsShared();
    const auto dcaXY = track.dcaXY();
    const auto dcaZ = track.dcaZ();
    std::vector<float> pidTPC, pidTOF;
    for (auto it : mPIDspecies) {
      pidTPC.push_back(getNsigmaTPC(track, it));
      pidTOF.push_back(getNsigmaTOF(track, it));
    }

    femtoDreamTrackSelection::TrackSel oldTrackSel = femtoDreamTrackSelection::kSign;
    size_t currentTrackSelCounter = 0;
    size_t pidCounter = 0;

    for (auto& sel : mSelections) {
      const auto selVariable = sel.getSelectionVariable();
      if (oldTrackSel != selVariable) {
        currentTrackSelCounter = 0;
      }
      if (selVariable == femtoDreamTrackSelection::kPIDnSigmaMax) {
        /// PID needs to be handled a bit differently since we more than one species
        for (size_t i = 0; i < pidTPC.size(); ++i) {
          bool isTrueTPC = (cutContainer >> counter) & 1;
          ++counter;
          bool isTrueComb = (cutContainer >> counter) & 1;
          ++counter;
          auto pidTPCVal = pidTPC.at(i);
          auto pidTOFVal = pidTOF.at(i);
          auto pidComb = std::sqrt(pidTPCVal * pidTPCVal + pidTOFVal * pidTOFVal);
          if (isTrueTPC) {
            mHistogramRegistry->fill(HIST("TrackCutsQA/pidTPCnsigmaMax"), pidCounter, pidTPCVal);
          }
          if (isTrueComb) {
            mHistogramRegistry->fill(HIST("TrackCutsQA/pidCombnsigmaMax"), pidCounter, pidComb);
          }
          ++pidCounter;
        }
        ++currentTrackSelCounter;
      } else {
        /// for the rest it's all the same
        bool isTrue = (cutContainer >> counter) & 1;
        switch (selVariable) {
          case (femtoDreamTrackSelection::kSign):
            if (isTrue) {
              mHistogramRegistry->fill(HIST("TrackCutsQA/Charge"), currentTrackSelCounter, sign);
            }
            break;
          case (femtoDreamTrackSelection::kpTMin):
            if (isTrue) {
              mHistogramRegistry->fill(HIST("TrackCutsQA/pTMin"), currentTrackSelCounter, pT);
            }
            break;
          case (femtoDreamTrackSelection::kpTMax):
            if (isTrue) {
              mHistogramRegistry->fill(HIST("TrackCutsQA/pTMax"), currentTrackSelCounter, pT);
            }
            break;
          case (femtoDreamTrackSelection::kEtaMax):
            if (isTrue) {
              mHistogramRegistry->fill(HIST("TrackCutsQA/etaMax"), currentTrackSelCounter, eta);
            }
            break;
          case (femtoDreamTrackSelection::kTPCnClsMin):
            if (isTrue) {
              mHistogramRegistry->fill(HIST("TrackCutsQA/tpcnClsMin"), currentTrackSelCounter, tpcNClsF);
            }
            break;
          case (femtoDreamTrackSelection::kTPCfClsMin):
            if (isTrue) {
              mHistogramRegistry->fill(HIST("TrackCutsQA/tpcfClsMin"), currentTrackSelCounter, tpcRClsC);
            }
            break;
          case (femtoDreamTrackSelection::kTPCcRowsMin):
            if (isTrue) {
              mHistogramRegistry->fill(HIST("TrackCutsQA/tpcnRowsMin"), currentTrackSelCounter, tpcNClsC);
            }
            break;
          case (femtoDreamTrackSelection::kTPCsClsMax):
            if (isTrue) {
              mHistogramRegistry->fill(HIST("TrackCutsQA/tpcnSharedMax"), currentTrackSelCounter, tpcNClsS);
            }
            break;
          case (femtoDreamTrackSelection::kDCAxyMax):
            if (isTrue) {
              mHistogramRegistry->fill(HIST("TrackCutsQA/dcaXYMax"), currentTrackSelCounter, dcaXY);
            }
            break;
          case (femtoDreamTrackSelection::kDCAzMax):
            if (isTrue) {
              mHistogramRegistry->fill(HIST("TrackCutsQA/dcaZMax"), currentTrackSelCounter, dcaZ);
            }
            break;
          case (femtoDreamTrackSelection::kPIDnSigmaMax):
            break;
        }
        ++currentTrackSelCounter;
        ++counter;
        oldTrackSel = selVariable;
      }
    }
  }
} // namespace femtoDream

} // namespace femtoDream
} // namespace o2::analysis

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMTRACKSELECTION_H_ */
