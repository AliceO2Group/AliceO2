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

/// \file  FastMultEst.h
/// \brief Fast multiplicity estimator for ITS
/// \author ruben.shahoyan@cern.ch

#include "ITStracking/FastMultEst.h"
#include "Framework/Logger.h"
#include <ctime>
#include <cstring>
#include <algorithm>
#include <TRandom.h>

using namespace o2::its;

namespace
{

// Convert trigger IR to ROF index on a given layer using LayerTiming
int findROFForIR(const o2::InteractionRecord& ir,
                 const o2::InteractionRecord& tfStartIR,
                 const LayerTiming& layerTiming)
{
  // Convert IR to BC-from-TF-start, which is the time base expected by LayerTiming.
  const int64_t bcFromTFStart = ir.differenceInBC(tfStartIR);
  if (bcFromTFStart < 0) {
    return -1;
  }
  return layerTiming.getROF(static_cast<LayerTiming::BCType>(bcFromTFStart));
}

template <int NLayers>
void enableCompatibleROFs(int baseLayer,
                          int baseRof,
                          const typename o2::its::ROFOverlapTable<NLayers>::View& overlapView,
                          o2::its::ROFMaskTable<NLayers>& sel)
{
  sel.setROFEnabled(baseLayer, baseRof);
  for (int layer = 0; layer < NLayers; ++layer) {
    if (layer == baseLayer) {
      continue;
    }
    const auto& overlap = overlapView.getOverlap(baseLayer, layer, baseRof);
    if (overlap.getEntries() > 0) {
      sel.setROFsEnabled(layer, overlap.getFirstEntry(), overlap.getEntries());
    }
  }
}

template <int NLayers>
std::vector<int> buildMultiplicityCounts(const std::array<gsl::span<const o2::itsmft::ROFRecord>, NLayers>& rofs,
                                         const std::array<gsl::span<const o2::itsmft::CompClusterExt>, NLayers>& clus,
                                         bool doStaggering,
                                         int multLayer)
{
  std::vector<int> multCounts;
  if (doStaggering) {
    multCounts.resize(rofs[multLayer].size());
    for (size_t iRof = 0; iRof < rofs[multLayer].size(); ++iRof) {
      multCounts[iRof] = rofs[multLayer][iRof].getNEntries();
    }
    return multCounts;
  }

  static const o2::itsmft::ChipMappingITS chipMapping;
  multCounts.resize(rofs[0].size(), 0);
  for (size_t iRof = 0; iRof < rofs[0].size(); ++iRof) {
    for (const auto& cluster : rofs[0][iRof].getROFData(clus[0])) {
      if (chipMapping.getLayer(cluster.getSensorID()) == multLayer) {
        ++multCounts[iRof];
      }
    }
  }
  return multCounts;
}
} // namespace

bool FastMultEst::sSeedSet = false;

///______________________________________________________
FastMultEst::FastMultEst()
{
  if (!sSeedSet && FastMultEstConfig::Instance().cutRandomFraction > 0.f) {
    sSeedSet = true;
    if (FastMultEstConfig::Instance().randomSeed > 0) {
      gRandom->SetSeed(FastMultEstConfig::Instance().randomSeed);
    } else if (FastMultEstConfig::Instance().randomSeed < 0) {
      gRandom->SetSeed(std::time(nullptr) % 0xffff);
    }
  }
}

///______________________________________________________
/// count clusters on the configured multiplicity layer
int FastMultEst::countClustersOnLayer(const gsl::span<const o2::itsmft::CompClusterExt>& clusters) const
{
  const int targetLayer = std::clamp(FastMultEstConfig::Instance().cutMultClusLayer, 0, NLayers - 1);
  int count = 0;
  int lr = FastMultEst::NLayers - 1;
  int nchAcc = o2::itsmft::ChipMappingITS::getNChips() - o2::itsmft::ChipMappingITS::getNChipsPerLr(lr);
  for (int i = clusters.size(); i--;) { // profit from clusters being ordered in chip increasing order
    while (clusters[i].getSensorID() < nchAcc) {
      assert(lr >= 0);
      nchAcc -= o2::itsmft::ChipMappingITS::getNChipsPerLr(--lr);
    }
    if (lr == targetLayer) {
      ++count;
    }
  }
  return count;
}

///______________________________________________________
/// find multiplicity for given number of clusters per layer
float FastMultEst::processNoiseFree(int nClusters)
{
  // Single-layer regime: estimate multiplicity from one configured layer only.
  const auto& conf = FastMultEstConfig::Instance();
  const int layer = std::clamp(conf.cutMultClusLayer, 0, NLayers - 1);
  const float acc = conf.accCorr[layer];
  nLayersUsed = nClusters > 0 ? 1 : 0;
  noisePerChip = 0.f;
  chi2 = 0.f;
  cov[0] = cov[1] = cov[2] = 0.f;
  if (nLayersUsed == 0 || acc <= 0.f) {
    mult = -1.f;
    return -1.f;
  }
  mult = nClusters / acc;
  return mult > 0 ? mult : 0;
}

///______________________________________________________
/// find multiplicity for given number of clusters per layer with mean noise imposed
float FastMultEst::processNoiseImposed(int nClusters)
{
  // Single-layer regime with imposed noise subtraction.
  const auto& conf = FastMultEstConfig::Instance();
  const int layer = std::clamp(conf.cutMultClusLayer, 0, NLayers - 1);
  const float acc = conf.accCorr[layer];
  const float nch = static_cast<float>(o2::itsmft::ChipMappingITS::getNChipsPerLr(layer));
  nLayersUsed = nClusters > 0 ? 1 : 0;
  chi2 = 0.f;
  cov[0] = cov[1] = cov[2] = 0.f;
  if (nLayersUsed == 0 || acc <= 0.f) {
    mult = -1.f;
    return -1.f;
  }
  mult = (nClusters - noisePerChip * nch) / acc;
  return mult;
}

int FastMultEst::selectROFs(const std::array<gsl::span<const o2::itsmft::ROFRecord>, NLayers>& rofs,
                            const std::array<gsl::span<const o2::itsmft::CompClusterExt>, NLayers>& clus,
                            const gsl::span<const o2::itsmft::PhysTrigger> trig,
                            uint32_t firstTForbit,
                            bool doStaggering,
                            const ROFOverlapTableN::View& overlapView,
                            ROFMaskTableN& sel)
{
  const auto& multEstConf = FastMultEstConfig::Instance(); // parameters for mult estimation and cuts
  const int selectionLayer = multEstConf.isMultCutRequested() ? std::clamp(multEstConf.cutMultClusLayer, 0, NLayers - 1) : overlapView.getClock();
  const auto multCounts = buildMultiplicityCounts<NLayers>(rofs, clus, doStaggering, selectionLayer);
  const int selectionRofCount = doStaggering ? static_cast<int>(rofs[selectionLayer].size()) : static_cast<int>(rofs[0].size());

  sel.resetMask();
  lastRandomSeed = gRandom->GetSeed();
  const o2::InteractionRecord tfStartIR{0, firstTForbit};
  // mask ROFs which are not good from the multiplicity selection (if any) POV
  struct ROFStatus {
    int entry = 0, priority = 0;
  };
  std::vector<ROFStatus> selROFs;
  selROFs.reserve(selectionRofCount);
  bool selmult = multEstConf.isMultCutRequested();
  for (int selectionRof = 0; selectionRof < selectionRofCount; ++selectionRof) {
    selROFs.emplace_back(selectionRof, (selmult && !multEstConf.isPassingMultCut(process(multCounts[selectionRof]))) ? -1 : 0);
  }
  if (!trig.empty() && multEstConf.preferTriggered) {
    const auto& selectionLayerTiming = overlapView.getLayer(selectionLayer);
    for (const auto& trigger : trig) {
      const int selectionRof = findROFForIR(trigger.ir, tfStartIR, selectionLayerTiming);
      if (selectionRof < 0 || selROFs[selectionRof].priority < 0) {
        continue;
      }
      selROFs[selectionRof].priority++; // increment trigger counter
    }
    sort(selROFs.begin(), selROFs.end(), [](const ROFStatus& a, const ROFStatus& b) { return a.priority > b.priority; }); // order in number of triggers, masked will go to the end
  }
  int nsel = 0;
  for (auto& rof : selROFs) {
    if (rof.priority >= 0 && (multEstConf.cutRandomFraction <= 0.f || (gRandom->Rndm() > multEstConf.cutRandomFraction))) {
      enableCompatibleROFs<NLayers>(selectionLayer, rof.entry, overlapView, sel);
      nsel++;
    }
  }
  LOGP(debug, "NSel = {} of {} rofs on layer {} Seeds: before {} after {}", nsel, selectionRofCount, selectionLayer, lastRandomSeed, gRandom->GetSeed());
  return nsel;
}
