// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FemtoDreamV0Selection.h
/// \brief Definition of the FemtoDreamV0Selection
/// \author Valentina Mantovani Sarti, TU München valentina.mantovani-sarti@tum.de and Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMV0SELECTION_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMV0SELECTION_H_

#include "FemtoDreamObjectSelection.h"
#include "FemtoDreamTrackSelection.h"
#include "FemtoDreamSelection.h"

#include "ReconstructionDataFormats/PID.h"
#include "AnalysisCore/RecoDecay.h"
#include "Framework/HistogramRegistry.h"
#include <Rtypes.h>
#include <cmath>

using namespace o2::framework;

namespace o2::analysis
{
namespace femtoDream
{
namespace femtoDreamV0Selection
{
enum V0Sel { kpTV0Min,
             kpTV0Max,
             kDCAV0DaughMax,
             kCPAV0Min,
             kTranRadV0Min,
             kTranRadV0Max,
             kDecVtxMax };
enum ChildTrackType { kPosTrack,
                      kNegTrack };
} // namespace femtoDreamV0Selection

/// \class FemtoDreamV0Selection
/// \brief Cut class to contain and execute all cuts applied to V0s
class FemtoDreamV0Selection : public FemtoDreamObjectSelection<float, femtoDreamV0Selection::V0Sel>
{
 public:
  /// Initializes histograms for the task
  void init(HistogramRegistry* registry);

  template <typename C, typename V, typename T>
  bool isSelectedMinimal(C const& col, V const& v0, T const& posTrack, T const& negTrack);

  template <typename C, typename V, typename T>
  std::vector<uint64_t> getCutContainer(C const& col, V const& v0, T const& posTrack, T const& negTrack);

  template <typename C, typename V>
  void fillQA(C const& col, V const& v0);

  template <typename T1, typename T2>
  void setChildCuts(femtoDreamV0Selection::ChildTrackType child, T1 selVal, T2 selVar, femtoDreamSelection::SelectionType selType)
  {
    if (child == femtoDreamV0Selection::kPosTrack) {
      PosDaughTrack.setSelection(selVal, selVar, selType);
    } else if (child == femtoDreamV0Selection::kNegTrack) {
      NegDaughTrack.setSelection(selVal, selVar, selType);
    }
  }

 private:
  FemtoDreamTrackSelection PosDaughTrack;
  FemtoDreamTrackSelection NegDaughTrack;

  ClassDefNV(FemtoDreamV0Selection, 1);
}; // namespace femtoDream

void FemtoDreamV0Selection::init(HistogramRegistry* registry)
{
  if (registry) {
    mHistogramRegistry = registry;
    fillSelectionHistogram("V0Cuts/cuthist");

    /// \todo this should be an automatic check in the parent class, and the return type should be templated
    int nSelections = getNSelections();
    if (8 * sizeof(uint64_t) < nSelections) {
      LOGF(error, "Number of selections to large for your container - quitting!");
    }
    /// \todo initialize histograms for children tracks of v0s
    mHistogramRegistry->add("V0Cuts/pThist", "; #it{p}_{T} (GeV/#it{c}); Entries", kTH1F, {{1000, 0, 10}});
    mHistogramRegistry->add("V0Cuts/etahist", "; #eta; Entries", kTH1F, {{1000, -1, 1}});
    mHistogramRegistry->add("V0Cuts/phihist", "; #phi; Entries", kTH1F, {{1000, 0, 2. * M_PI}});
    mHistogramRegistry->add("V0Cuts/dcaDauToVtx", "; DCADaug_{Vtx} (cm); Entries", kTH1F, {{1000, 0, 10}});
    mHistogramRegistry->add("V0Cuts/transRadius", "; #it{r}_{xy} (cm); Entries", kTH1F, {{1500, 0, 150}});
    mHistogramRegistry->add("V0Cuts/decayVtxXPV", "; #it{iVtx}_{x} (cm); Entries", kTH1F, {{2000, 0, 200}});
    mHistogramRegistry->add("V0Cuts/decayVtxYPV", "; #it{iVtx}_{y} (cm)); Entries", kTH1F, {{2000, 0, 200}});
    mHistogramRegistry->add("V0Cuts/decayVtxZPV", "; #it{iVtx}_{z} (cm); Entries", kTH1F, {{2000, 0, 200}});
    mHistogramRegistry->add("V0Cuts/cpa", "; #it{cos(#alpha)}; Entries", kTH1F, {{1000, 0.9, 1.}});
    mHistogramRegistry->add("V0Cuts/cpapTBins", "; #it{p}_{T} (GeV/#it{c}); #it{cos(#alpha)}", kTH2F, {{8, 0.3, 4.3}, {1000, 0.9, 1.}});
  }
}

template <typename C, typename V, typename T>
bool FemtoDreamV0Selection::isSelectedMinimal(C const& col, V const& v0, T const& posTrack, T const& negTrack)
{
  const auto signPos = posTrack.sign();
  const auto signNeg = negTrack.sign();
  // printf("pos sign = %i --- neg sign = %i\n", signPos, signNeg);
  if (signPos < 0 || signNeg > 0) {
    printf("-Something wrong in isSelectedMinimal--\n");
    printf("ERROR - Wrong sign for V0 daughters\n");
  }
  const float pT = v0.pt();
  const std::vector<float> decVtx = {v0.x(), v0.y(), v0.z()};
  const float tranRad = v0.v0radius();
  const float dcaDaughv0 = v0.dcaV0daughters();
  const float cpav0 = v0.v0cosPA(col.posX(), col.posY(), col.posZ());

  /// check whether the most open cuts are fulfilled - most of this should have already be done by the filters
  const static int nPtV0MinSel = getNSelections(femtoDreamV0Selection::kpTV0Min);
  const static int nPtV0MaxSel = getNSelections(femtoDreamV0Selection::kpTV0Max);
  const static int nDCAV0DaughMax = getNSelections(femtoDreamV0Selection::kDCAV0DaughMax);
  const static int nCPAV0Min = getNSelections(femtoDreamV0Selection::kCPAV0Min);
  const static int nTranRadV0Min = getNSelections(femtoDreamV0Selection::kTranRadV0Min);
  const static int nTranRadV0Max = getNSelections(femtoDreamV0Selection::kTranRadV0Max);
  const static int nDecVtxMax = getNSelections(femtoDreamV0Selection::kDecVtxMax);

  const static float pTV0Min = getMinimalSelection(femtoDreamV0Selection::kpTV0Min, femtoDreamSelection::kLowerLimit);
  const static float pTV0Max = getMinimalSelection(femtoDreamV0Selection::kpTV0Max, femtoDreamSelection::kUpperLimit);
  const static float DCAV0DaughMax = getMinimalSelection(femtoDreamV0Selection::kDCAV0DaughMax, femtoDreamSelection::kUpperLimit);
  const static float CPAV0Min = getMinimalSelection(femtoDreamV0Selection::kCPAV0Min, femtoDreamSelection::kLowerLimit);
  const static float TranRadV0Min = getMinimalSelection(femtoDreamV0Selection::kTranRadV0Min, femtoDreamSelection::kLowerLimit);
  const static float TranRadV0Max = getMinimalSelection(femtoDreamV0Selection::kTranRadV0Max, femtoDreamSelection::kUpperLimit);
  const static float DecVtxMax = getMinimalSelection(femtoDreamV0Selection::kDecVtxMax, femtoDreamSelection::kAbsUpperLimit);

  if (nPtV0MinSel > 0 && pT < pTV0Min) {
    return false;
  }
  if (nPtV0MaxSel > 0 && pT > pTV0Max) {
    return false;
  }
  if (nDCAV0DaughMax > 0 && dcaDaughv0 > DCAV0DaughMax) {
    return false;
  }
  if (nCPAV0Min > 0 && cpav0 < CPAV0Min) {
    return false;
  }
  if (nTranRadV0Min > 0 && tranRad < TranRadV0Min) {
    return false;
  }
  if (nTranRadV0Max > 0 && tranRad > TranRadV0Max) {
    return false;
  }
  for (int i = 0; i < decVtx.size(); i++) {
    if (nDecVtxMax > 0 && decVtx.at(i) > DecVtxMax) {
      return false;
    }
  }
  if (!PosDaughTrack.isSelectedMinimal(posTrack)) {
    return false;
  }
  if (!NegDaughTrack.isSelectedMinimal(negTrack)) {
    return false;
  }
  return true;
}

/// the CosPA of V0 needs as argument the posXYZ of collisions vertex so we need to pass the collsion as well
template <typename C, typename V, typename T>
std::vector<uint64_t> FemtoDreamV0Selection::getCutContainer(C const& col, V const& v0, T const& posTrack, T const& negTrack)
{
  uint64_t outputPosTrack = PosDaughTrack.getCutContainer(posTrack);
  uint64_t outputNegTrack = NegDaughTrack.getCutContainer(negTrack);
  uint64_t output = 0;
  size_t counter = 0;

  const auto pT = v0.pt();
  const auto tranRad = v0.v0radius();
  const auto dcaDaughv0 = v0.dcaV0daughters();
  const auto cpav0 = v0.v0cosPA(col.posX(), col.posY(), col.posZ());
  const std::vector<float> decVtx = {v0.x(), v0.y(), v0.z()};

  float observable;
  for (auto& sel : mSelections) {
    const auto selVariable = sel.getSelectionVariable();
    if (selVariable == femtoDreamV0Selection::kDecVtxMax) {
      for (size_t i = 0; i < decVtx.size(); ++i) {
        auto decVtxValue = decVtx.at(i);
        sel.checkSelectionSetBit(decVtxValue, output, counter);
      }
    } else {
      switch (selVariable) {
        case (femtoDreamV0Selection::kpTV0Min):
        case (femtoDreamV0Selection::kpTV0Max):
          observable = pT;
          break;
        case (femtoDreamV0Selection::kDCAV0DaughMax):
          observable = dcaDaughv0;
          break;
        case (femtoDreamV0Selection::kCPAV0Min):
          observable = cpav0;
          break;
        case (femtoDreamV0Selection::kTranRadV0Min):
        case (femtoDreamV0Selection::kTranRadV0Max):
          observable = tranRad;
          break;
        case (femtoDreamV0Selection::kDecVtxMax):
          break;
      }
      sel.checkSelectionSetBit(observable, output, counter);
    }
  }
  return {{outputPosTrack, outputNegTrack, output}};
}

template <typename C, typename V>
void FemtoDreamV0Selection::fillQA(C const& col, V const& v0)
{
  if (mHistogramRegistry) {
    mHistogramRegistry->fill(HIST("V0Cuts/pThist"), v0.pt());
    mHistogramRegistry->fill(HIST("V0Cuts/etahist"), v0.eta());
    mHistogramRegistry->fill(HIST("V0Cuts/phihist"), v0.phi());
    mHistogramRegistry->fill(HIST("V0Cuts/dcaDauToVtx"), v0.dcaV0daughters());
    mHistogramRegistry->fill(HIST("V0Cuts/transRadius"), v0.v0radius());
    mHistogramRegistry->fill(HIST("V0Cuts/decayVtxXPV"), v0.x());
    mHistogramRegistry->fill(HIST("V0Cuts/decayVtxYPV"), v0.y());
    mHistogramRegistry->fill(HIST("V0Cuts/decayVtxZPV"), v0.z());
    mHistogramRegistry->fill(HIST("V0Cuts/cpa"), v0.v0cosPA(col.posX(), col.posY(), col.posZ()));
    mHistogramRegistry->fill(HIST("V0Cuts/cpapTBins"), v0.pt(), v0.v0cosPA(col.posX(), col.posY(), col.posZ()));
  }
}

} // namespace femtoDream
} // namespace o2::analysis

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMV0SELECTION_H_ */
