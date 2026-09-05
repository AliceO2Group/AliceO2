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
///

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/TrackingKernels.h"
#include "ITStrackingGPU/LaunchGeometry.h"
#include "ITSMFTTracking/MathUtils.h"
#include "ITStracking/Configuration.h"

namespace o2::its
{
using o2::itsmft::tracking::runOnSlab;
using o2::itsmft::tracking::SlabSite;

template <int NLayers>
void TrackerTraitsGPU<NLayers>::initialiseTimeFrame(const int iteration)
{
  this->mTaskArena->execute([&] {
    mTimeFrameGPU->initialise(this->mTrkParams[iteration], this->mTrkParams[iteration].NLayers, iteration);
  });
  // load iteration parameters
  mTimeFrameGPU->loadIterationParameters(this->mTrkParams[iteration]);

  if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
    // on default stream
    mTimeFrameGPU->loadVertices();
    if (this->mTrkParams[iteration].PassFlags[IterationStep::LoadPersistentTables]) {
      mTimeFrameGPU->loadROFOverlapTable(); // this can be put in constant memory actually
      mTimeFrameGPU->loadTrackingTopologies();
    }
    mTimeFrameGPU->loadROFVertexLookupTable();
    // once the tables are in persistent memory just re-upload the vertex one
    // mTimeFrameGPU->uploadROFVertexLookupTable();
    mTimeFrameGPU->loadIndexTableUtils();
    // pinned on host
    mTimeFrameGPU->createUsedClustersDeviceArray();
    mTimeFrameGPU->createClustersDeviceArray();
    mTimeFrameGPU->createUnsortedClustersDeviceArray();
    mTimeFrameGPU->createClustersIndexTablesArray();
    mTimeFrameGPU->createTrackingFrameInfoDeviceArray();
    mTimeFrameGPU->createROFrameClustersDeviceArray();
    // device array
    mTimeFrameGPU->createClusterRadiiDevice();
    mTimeFrameGPU->uploadClusterRadii();
    mTimeFrameGPU->createTrackletsLUTDeviceArray();
    mTimeFrameGPU->createTrackletsBuffersArray();
    mTimeFrameGPU->createCellsBuffersArray();
    mTimeFrameGPU->createCellsLUTDeviceArray();
    mTimeFrameGPU->createClusterOwnersDeviceArray();
  }
  if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass] || this->mTrkParams[iteration].PassFlags[IterationStep::UseUPCMask]) {
    mTimeFrameGPU->loadROFCutMask(iteration);
  }
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::adoptTimeFrame(TimeFrame<NLayers>* tf)
{
  mTimeFrameGPU = static_cast<gpu::TimeFrameGPU<NLayers>*>(tf);
  this->mTimeFrame = static_cast<TimeFrame<NLayers>*>(tf);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeLayerTracklets(const int iteration, int iVertex)
{
  const auto topology = mTimeFrameGPU->getDeviceTrackingTopologyView();
  const auto hostTopology = mTimeFrameGPU->getTrackingTopologyView();
  const bool loadFirstPassData = this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass] && iVertex <= 0; // load data only on first pass and first vertex
  for (int iLayer{0}; iLayer < this->mTrkParams[iteration].NLayers; ++iLayer) {
    if (loadFirstPassData) {
      mTimeFrameGPU->createUsedClustersDevice(iLayer);
      mTimeFrameGPU->loadUnsortedClustersDevice(iLayer);
      mTimeFrameGPU->loadROFrameClustersDevice(iLayer);
      mTimeFrameGPU->sortClustersDevice(iLayer, this->mTrkParams[iteration]);
    }
    mTimeFrameGPU->recordEvent(iLayer);
  }

  for (int linkId{0}; linkId < hostTopology.nLinks; ++linkId) {
    mTimeFrameGPU->createTrackletsLUTDevice(loadFirstPassData, linkId); // on first pass allocates, then only clears memory
  }

  // Stack allocations created from trackleting through road finding are scoped to one tracker pass.
  // With per-primary-vertex processing, the chain is called once per vertex while initialisation is only done once.
  mTimeFrameGPU->pushMemoryStack(iteration);

  const auto nClusters = mTimeFrameGPU->getClusterSizes();
  const bool vtxMode = this->mTrkParams[iteration].PassFlags[IterationStep::SeedingVertexPass];
  const bool useDiamond = this->mTrkParams[iteration].UseDiamond;
  if (useDiamond) {
    const Vertex diamondVert(this->mTrkParams[iteration].Diamond, this->mTrkParams[iteration].DiamondCov, 1, 1.f);
    mTimeFrameGPU->createDiamondDevice(diamondVert);
    mTimeFrameGPU->recordEvent(0);
  }
  const Vertex* deviceVertices = useDiamond ? mTimeFrameGPU->getDeviceDiamond() : mTimeFrameGPU->getDeviceVertices();
  bounded_vector<float> vtxPhiCuts(vtxMode ? hostTopology.nLinks : 0,
                                   this->mTrkParams[iteration].VtxPhiCut,
                                   this->getMemoryPool().get());
  auto& linkPhiCuts = vtxMode ? vtxPhiCuts : mTimeFrameGPU->getLinkPhiCuts();

  for (int linkId{0}; linkId < hostTopology.nLinks; ++linkId) {
    const auto link = hostTopology.getLink(linkId);
    mTimeFrameGPU->waitEvent(linkId, link.fromLayer);
    mTimeFrameGPU->waitEvent(linkId, link.toLayer);
    if (useDiamond) {
      mTimeFrameGPU->waitEvent(linkId, 0); // links not anchored on layer 0 must still wait for the diamond upload
    }
    const auto key = CapacityEstimator::makeKey(SlabSite::Tracklets, iteration, iVertex + 1, linkId);
    const auto scale = static_cast<double>(nClusters[link.fromLayer]);
    runOnSlab(mTimeFrameGPU->getCapacityEstimator(), key, scale, [&](const int capacity) {
      mTimeFrameGPU->createTrackletsBuffers(linkId, capacity);
      return TrackingKernels<NLayers>::computeTrackletsInROFsHandler(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                                                     mTimeFrameGPU->getDeviceROFMaskTableView(),
                                                                     linkId,
                                                                     link.fromLayer,
                                                                     link.toLayer,
                                                                     mTimeFrameGPU->getDeviceROFOverlapTableView(),
                                                                     mTimeFrameGPU->getDeviceROFVertexLookupTableView(),
                                                                     iVertex,
                                                                     deviceVertices,
                                                                     vtxMode,
                                                                     mTimeFrameGPU->getDeviceArrayClusters(),
                                                                     nClusters,
                                                                     mTimeFrameGPU->getDeviceROFrameClusters(),
                                                                     (const uint8_t**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                                                     mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                                                     mTimeFrameGPU->getDeviceArrayTracklets(),
                                                                     mTimeFrameGPU->getDeviceTracklets(),
                                                                     mTimeFrameGPU->getNTracklets(),
                                                                     capacity,
                                                                     mTimeFrameGPU->getDeviceTrackletsLUTs(),
                                                                     this->mTrkParams[iteration].PassFlags[IterationStep::SelectUPCVertices],
                                                                     this->mTrkParams[iteration].NSigmaCut,
                                                                     topology,
                                                                     linkPhiCuts,
                                                                     this->mTrkParams[iteration].PVres,
                                                                     mTimeFrameGPU->getDeviceMinRs(),
                                                                     mTimeFrameGPU->getDeviceMaxRs(),
                                                                     mTimeFrameGPU->getPositionResolutions(),
                                                                     this->mTrkParams[iteration].LayerRadii,
                                                                     mTimeFrameGPU->getLinkMSAngles(),
                                                                     mTimeFrameGPU->getFrameworkAllocator(),
                                                                     mTimeFrameGPU->getStreams());
    });
    mTimeFrameGPU->recordEvent(linkId);
  }
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeLayerCells(const int iteration)
{
  const auto topology = mTimeFrameGPU->getDeviceTrackingTopologyView();
  const auto hostTopology = mTimeFrameGPU->getTrackingTopologyView();
  for (int iLayer{0}; iLayer < this->mTrkParams[iteration].NLayers; ++iLayer) {
    if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
      mTimeFrameGPU->loadUnsortedClustersDevice(iLayer); // latched: a no-op if trackleting already did it
      mTimeFrameGPU->loadTrackingFrameInfoDevice(iLayer);
    }
    mTimeFrameGPU->recordEvent(iLayer);
  }

  for (int cellTopologyId{hostTopology.nCells}; cellTopologyId--;) {
    const auto cellTopology = hostTopology.getCell(cellTopologyId);
    const auto first = hostTopology.getLink(cellTopology.firstLink);
    const auto second = hostTopology.getLink(cellTopology.secondLink);
    const float cellDeltaPhiCut = this->mTrkParams[iteration].PassFlags[IterationStep::SeedingVertexPass] ? math_utils::cellDeltaPhiBound(this->mBz, this->mTrkParams[iteration].CellDeltaPhiMinPt,
                                                                                                                                          this->mTrkParams[iteration].LayerRadii[first.fromLayer],
                                                                                                                                          this->mTrkParams[iteration].LayerRadii[first.toLayer],
                                                                                                                                          this->mTrkParams[iteration].LayerRadii[second.toLayer],
                                                                                                                                          mTimeFrameGPU->getLinkMSAngle(cellTopology.firstLink))
                                                                                                          : -1.f;
    const float cellTanLNSigma = this->mTrkParams[iteration].CellDeltaTanLambdaNSigma > 0.f
                                   ? this->mTrkParams[iteration].CellDeltaTanLambdaNSigma
                                   : this->mTrkParams[iteration].NSigmaCut;
    const int currentLayerTrackletsNum{static_cast<int>(mTimeFrameGPU->getNTracklets()[cellTopology.firstLink])};
    if (!currentLayerTrackletsNum || !mTimeFrameGPU->getNTracklets()[cellTopology.secondLink]) {
      mTimeFrameGPU->getNCells()[cellTopologyId] = 0;
      continue;
    }

    mTimeFrameGPU->createCellsLUTDevice(cellTopologyId);
    mTimeFrameGPU->waitEvent(cellTopologyId, cellTopology.firstLink);
    mTimeFrameGPU->waitEvent(cellTopologyId, cellTopology.secondLink);
    mTimeFrameGPU->waitEvent(cellTopologyId, first.fromLayer);
    mTimeFrameGPU->waitEvent(cellTopologyId, first.toLayer);
    mTimeFrameGPU->waitEvent(cellTopologyId, second.toLayer);
    const auto key = CapacityEstimator::makeKey(SlabSite::Cells, iteration, 0, cellTopologyId);
    const auto scale = static_cast<double>(currentLayerTrackletsNum);
    const int emitted = runOnSlab(mTimeFrameGPU->getCapacityEstimator(), key, scale, [&](const int capacity) {
      mTimeFrameGPU->createCellsBuffers(cellTopologyId, capacity);
      return TrackingKernels<NLayers>::computeCellsHandler(mTimeFrameGPU->getDeviceArrayClusters(),
                                                           mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                                                           mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                                           mTimeFrameGPU->getDeviceArrayTracklets(),
                                                           mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                                                           currentLayerTrackletsNum,
                                                           cellTopologyId,
                                                           topology,
                                                           mTimeFrameGPU->getDeviceCells()[cellTopologyId],
                                                           capacity,
                                                           mTimeFrameGPU->getDeviceCellLUTs()[cellTopologyId],
                                                           this->mBz,
                                                           this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                                           this->mTrkParams[iteration].CellDeltaTanLambdaSigma,
                                                           cellDeltaPhiCut,
                                                           cellTanLNSigma,
                                                           mTimeFrameGPU->getDeviceLayerxX0(),
                                                           mTimeFrameGPU->getFrameworkAllocator(),
                                                           mTimeFrameGPU->getStreams());
    });
    mTimeFrameGPU->getNCells()[cellTopologyId] = emitted;
    mTimeFrameGPU->recordEvent(cellTopologyId);
  }
  mTimeFrameGPU->syncStreams(false);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeVertexCandidates(const int iteration)
{
  const int nCells = mTimeFrameGPU->getNCells()[0];
  if (!nCells) {
    mTimeFrameGPU->setNLinesTotal(0);
    return;
  }
  mTimeFrameGPU->createClusterOwnersDevice();
  mTimeFrameGPU->createLinesDevice(nCells);
  mTimeFrameGPU->resetClusterOwnersDevice();
  mTimeFrameGPU->syncStreams(false);

  TrackingKernels<NLayers>::registerClusterOwnershipHandler(mTimeFrameGPU->getDeviceCells()[0],
                                                            nCells,
                                                            mTimeFrameGPU->getDeviceArrayClusterOwners(),
                                                            mTimeFrameGPU->getStream(0));

  TrackingKernels<NLayers>::linearizeCellsToLinesHandler(nCells,
                                                         mTimeFrameGPU->getDeviceCells()[0],
                                                         mTimeFrameGPU->getDeviceArrayClusterOwners(),
                                                         mTimeFrameGPU->getDeviceROFramesClusters(1),
                                                         mTimeFrameGPU->getNrof(1),
                                                         this->mTrkParams[iteration].CellLineSharedClusterCut,
                                                         mTimeFrameGPU->getDeviceLines(),
                                                         mTimeFrameGPU->getDeviceLineRof(),
                                                         mTimeFrameGPU->getDeviceLineClusters(),
                                                         mTimeFrameGPU->getDeviceLineSlots(),
                                                         mTimeFrameGPU->getBeamX(),
                                                         mTimeFrameGPU->getBeamY(),
                                                         this->mTrkParams[iteration].VtxMaxZPositionAllowed,
                                                         this->mTrkParams[iteration].VtxLineMinPt,
                                                         mTimeFrameGPU->getDeviceLineZs(),
                                                         mTimeFrameGPU->getDeviceLineTimes(),
                                                         mTimeFrameGPU->getDeviceLineChi2(),
                                                         mTimeFrameGPU->getDeviceLinePt(),
                                                         mTimeFrameGPU->getFrameworkAllocator(),
                                                         mTimeFrameGPU->getStream(0));
  const unsigned int nLines = mTimeFrameGPU->downloadLinesDevice();

  TrackingKernels<NLayers>::sortLinesHandler(nLines,
                                             mTimeFrameGPU->getNrof(1),
                                             mTimeFrameGPU->getLineProjSoA(),
                                             mTimeFrameGPU->getLineProjSortedSoA(),
                                             mTimeFrameGPU->getDeviceLineRof(),
                                             mTimeFrameGPU->getDeviceRofLineOffsets(),
                                             mTimeFrameGPU->getFrameworkAllocator(),
                                             mTimeFrameGPU->getStream(0));

  const auto& tp = this->mTrkParams[iteration];
  const float zWindow = 0.5f * tp.VtxClusterCut;
  TrackingKernels<NLayers>::scanDensityHandler(static_cast<int>(nLines),
                                               mTimeFrameGPU->getLineProjSortedSoA(),
                                               mTimeFrameGPU->getDeviceRofLineOffsets(),
                                               mTimeFrameGPU->getDeviceLineDensity(),
                                               mTimeFrameGPU->getDeviceLineWin(),
                                               zWindow,
                                               mTimeFrameGPU->getStream(0));

  const float fineZWindow = tp.VtxFineZWindow;
  const bool doFine = fineZWindow > 0.f && fineZWindow < zWindow;
  if (doFine) {
    TrackingKernels<NLayers>::scanDensityHandler(static_cast<int>(nLines),
                                                 mTimeFrameGPU->getLineProjSortedSoA(),
                                                 mTimeFrameGPU->getDeviceRofLineOffsets(),
                                                 mTimeFrameGPU->getDeviceLineDensityFine(),
                                                 mTimeFrameGPU->getDeviceLineWinFine(),
                                                 fineZWindow,
                                                 mTimeFrameGPU->getStream(0));
  }

  TrackingKernels<NLayers>::findPeaksHandler(nLines,
                                             mTimeFrameGPU->getNrof(1),
                                             mTimeFrameGPU->getLineProjSortedSoA(),
                                             mTimeFrameGPU->getDeviceRofLineOffsets(),
                                             mTimeFrameGPU->getDeviceLineDensity(),
                                             mTimeFrameGPU->getDeviceLineWin(),
                                             mTimeFrameGPU->getDeviceLineIsPeak(),
                                             doFine ? mTimeFrameGPU->getDeviceLineDensityFine() : nullptr,
                                             doFine ? mTimeFrameGPU->getDeviceLineWinFine() : nullptr,
                                             tp.VtxFineMinDensity,
                                             doFine ? mTimeFrameGPU->getDeviceLineIsPeakFine() : nullptr,
                                             mTimeFrameGPU->getDevicePeakScan(),
                                             mTimeFrameGPU->getDevicePeakLineIdx(),
                                             mTimeFrameGPU->getDevicePeakOffsets(),
                                             mTimeFrameGPU->getFrameworkAllocator(),
                                             mTimeFrameGPU->getStream(0));

  const float goodLinePtCut = std::abs(mTimeFrameGPU->getBz()) > 0.01f ? tp.VtxGoodLinePtCut : -1.f;
  TrackingKernels<NLayers>::fitPeaksHandler(mTimeFrameGPU->getDeviceNPeaks(),
                                            mTimeFrameGPU->getDevicePeakLineIdx(),
                                            mTimeFrameGPU->getDeviceLineWin(),
                                            mTimeFrameGPU->getLineProjSortedSoA(),
                                            mTimeFrameGPU->getDeviceLines(),
                                            mTimeFrameGPU->getDeviceLineChi2(),
                                            mTimeFrameGPU->getDeviceLinePt(),
                                            tp.VtxGoodLineChi2Cut,
                                            goodLinePtCut,
                                            tp.VtxPairCut * tp.VtxPairCut,
                                            tp.VtxNSigmaCut,
                                            tp.VtxClusterContributorsCut,
                                            mTimeFrameGPU->getBeamX(),
                                            mTimeFrameGPU->getBeamY(),
                                            doFine ? mTimeFrameGPU->getDeviceLineIsPeakFine() : nullptr,
                                            tp.VtxFineMaxDrift,
                                            mTimeFrameGPU->getDeviceVertexCands(),
                                            mTimeFrameGPU->getStream(0));

  const float duplicateZCut = tp.VtxDuplicateZCut > 0.f
                                ? tp.VtxDuplicateZCut
                                : std::max(4.f * tp.VtxPairCut, 0.5f * tp.VtxClusterCut);
  TrackingKernels<NLayers>::dedupVertexCandidatesHandler(mTimeFrameGPU->getDeviceNPeaks(),
                                                         mTimeFrameGPU->getDevicePeakLineIdx(),
                                                         mTimeFrameGPU->getDevicePeakOffsets(),
                                                         mTimeFrameGPU->getLineProjSortedSoA(),
                                                         duplicateZCut,
                                                         tp.VtxDuplicateZScale,
                                                         mTimeFrameGPU->getDeviceVertexCands(),
                                                         mTimeFrameGPU->getStream(0));

  const bool withMC = mTimeFrameGPU->hasMCinformation() && this->mTrkParams[iteration].CreateArtefactLabels;
  mTimeFrameGPU->downloadVertexCandsDevice(); // sets nPeaks + host candidate/peak-offset mirrors
  if (withMC) {
    // pull back the peak windows and re-run the membership test here
    mTimeFrameGPU->downloadPeakMembershipInputs();
  }
  const auto& lines = mTimeFrameGPU->getHostLines();
  const auto& lineRof = mTimeFrameGPU->getHostLineRof();
  const auto& lineClusters = mTimeFrameGPU->getHostLineClusters();
  const int nRofs = mTimeFrameGPU->getNrof(1);
  if (withMC) {
    mTimeFrameGPU->getLineLabelFlat().assign(nLines, o2::MCCompLabel()); // global-indexed, for the vertex-label vote
  }
  auto lineLabel = [&](const int* cl) -> o2::MCCompLabel {
    const auto l0 = mTimeFrameGPU->getClusterLabels(0, cl[0]);
    const auto l1 = mTimeFrameGPU->getClusterLabels(1, cl[1]);
    const auto l2 = mTimeFrameGPU->getClusterLabels(2, cl[2]);
    for (const auto& a : l0) {
      if (!a.isValid()) {
        continue;
      }
      bool in1{false}, in2{false};
      for (const auto& b : l1) {
        if (b == a) {
          in1 = true;
          break;
        }
      }
      for (const auto& c : l2) {
        if (c == a) {
          in2 = true;
          break;
        }
      }
      if (in1 && in2) {
        return a;
      }
    }
    return o2::MCCompLabel();
  };
  for (unsigned int i{0}; i < nLines; ++i) {
    const int rof = lineRof[i];
    if (rof < 0 || rof >= nRofs) {
      LOGP(fatal, "ITS GPU linearizer: line {} carries out-of-range ROF {} (nRofs={}).", i, rof, nRofs);
    }
    const auto& l = lines[i];
    mTimeFrameGPU->getLines(rof).emplace_back(std::array<float, 3>{l.originPoint[0], l.originPoint[1], l.originPoint[2]},
                                              std::array<float, 3>{l.cosinesDirector[0], l.cosinesDirector[1], l.cosinesDirector[2]},
                                              l.mTime);
    if (withMC) {
      const auto lbl = lineLabel(&lineClusters[3 * i]);
      mTimeFrameGPU->getLinesLabel(rof).emplace_back(lbl);
      mTimeFrameGPU->getLineLabelFlat()[i] = lbl; // same label, global-indexed for the vote
    }
  }
  mTimeFrameGPU->setNLinesTotal(nLines);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeVertices(const int iteration)
{
  const int nRofs = mTimeFrameGPU->getNrof(1);
  const bool withMC = mTimeFrameGPU->hasMCinformation() && this->mTrkParams[iteration].CreateArtefactLabels;
  const int suppressLowMultDebris = this->mTrkParams[iteration].VtxSuppressLowMultDebris;
  const bool skipHighMultRofs = this->mTrkParams[iteration].PassFlags[IterationStep::SkipROFsAboveThreshold];

  const auto& cands = mTimeFrameGPU->getHostVertexCands();       // per compacted peak slot [0, nPeaks)
  const auto& peakOffsets = mTimeFrameGPU->getHostPeakOffsets(); // nRofs+1; per-ROF peak slices

  std::vector<std::vector<Vertex>> rofVertices(nRofs);
  std::vector<std::vector<VertexLabel>> rofLabels(nRofs);
  const float goodSig = this->mTrkParams[iteration].VtxGoodContributorsSignificance;

  for (int rofId = 0; rofId < nRofs; ++rofId) {
    if (skipHighMultRofs &&
        static_cast<int>(mTimeFrameGPU->getROFVertexLookupTableView().getVertices(1, rofId).getEntries()) > this->mTrkParams[iteration].VertPerRofThreshold) {
      continue;
    }
    // Survivors of this ROF, sorted by contributor count desc
    std::vector<int> accepted;
    for (int p = peakOffsets[rofId]; p < peakOffsets[rofId + 1]; ++p) {
      if (cands[p].keep) {
        accepted.push_back(p);
      }
    }
    std::sort(accepted.begin(), accepted.end(), [&](const int a, const int b) { return cands[a].size > cands[b].size; });

    double rofLoad = 0.;
    if (goodSig > 0.f) { // compute the number of contributors in this ROF
      for (int p = peakOffsets[rofId]; p < peakOffsets[rofId + 1]; ++p) {
        if (cands[p].ok && !cands[p].fine) {
          rofLoad += cands[p].size;
        }
      }
    }
    const float sigThreshold = goodSig > 0.f ? goodSig * std::sqrt(static_cast<float>(std::max(rofLoad, 1.))) : 0.f;

    for (const int p : accepted) {
      const auto& c = cands[p];
      if (!rofVertices[rofId].empty()) {
        if (goodSig > 0.f) {
          if (c.nGood <= sigThreshold) {
            continue;
          }
        } else if (c.size < suppressLowMultDebris) {
          continue;
        }
      }
      const float pos[3] = {c.x, c.y, c.z};
      Vertex vertex(pos, c.rms2, static_cast<ushort>(c.size), c.avgDist2);
      vertex.setTimeStamp(c.time);
      if (this->mTrkParams[iteration].PassFlags[IterationStep::MarkVerticesAsUPC]) {
        vertex.setFlags(Vertex::UPCMode);
      }
      rofVertices[rofId].push_back(vertex);
      if (withMC) {
        // Re-apply here the rule the fit used: a line contributes when it is
        // time-compatible with the peak line and within pairCut of the candidate's seed
        const auto& memb = mTimeFrameGPU->getHostPeakMembership();
        const auto& lines = mTimeFrameGPU->getHostLines();
        const auto& lineLabelFlat = mTimeFrameGPU->getLineLabelFlat(); // nLines, global-indexed
        const int nLab = static_cast<int>(lineLabelFlat.size());
        const float pairCut2 = this->mTrkParams[iteration].VtxPairCut * this->mTrkParams[iteration].VtxPairCut;
        const int k = memb.peakLineIdx[p];
        const auto tk = memb.times[k];
        const auto wk = memb.win[k];
        std::vector<o2::MCCompLabel> labels;
        labels.reserve(wk.hi - wk.lo);
        for (int j = wk.lo; j < wk.hi; ++j) {
          const auto tj = memb.times[j];
          if (!tk.isCompatible(tj)) {
            continue;
          }
          const int gi = memb.sortedToLine[j];
          if (gi < 0 || gi >= nLab) {
            LOGP(error, "Seeding vertexer: member line {} out of range (nLines={}), skipping label", gi, nLab);
            continue;
          }
          if (gpu::GPULine::getDistance2FromPoint(lines[gi], c.seed) >= pairCut2) {
            continue;
          }
          labels.push_back(lineLabelFlat[gi]);
        }
        rofLabels[rofId].push_back(computeMainVertexLabel(labels));
      }
    }
  }

  for (int rofId = 0; rofId < nRofs; ++rofId) {
    for (auto& vertex : rofVertices[rofId]) {
      mTimeFrameGPU->addPrimaryVertex(vertex);
    }
    if (withMC) {
      for (auto& label : rofLabels[rofId]) {
        mTimeFrameGPU->addPrimaryVertexLabel(label);
      }
    }
  }

  auto& pvs = mTimeFrameGPU->getPrimaryVertices();
  std::vector<size_t> indices(pvs.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&pvs](const size_t i, const size_t j) {
    const auto aLower = pvs[i].getTimeStamp().lower();
    const auto bLower = pvs[j].getTimeStamp().lower();
    if (aLower != bLower) {
      return aLower < bLower;
    }
    return pvs[i].getNContributors() > pvs[j].getNContributors();
  });
  std::decay_t<decltype(pvs)> sortedVtx(pvs.get_allocator());
  sortedVtx.reserve(pvs.size());
  for (const size_t idx : indices) {
    sortedVtx.push_back(pvs[idx]);
  }
  pvs.swap(sortedVtx);
  if (withMC) {
    auto& mc = mTimeFrameGPU->getPrimaryVerticesLabels();
    std::decay_t<decltype(mc)> sortedMC(mc.get_allocator());
    sortedMC.reserve(mc.size());
    for (const size_t idx : indices) {
      sortedMC.push_back(mc[idx]);
    }
    mc.swap(sortedMC);
  }
  mTimeFrameGPU->updateROFVertexLookupTable();

  mTimeFrameGPU->popMemoryStack(iteration); // frees the whole seeding-pass stack frame
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::findCellsNeighbours(const int iteration)
{
  const auto hostTopology = mTimeFrameGPU->getTrackingTopologyView();
  bounded_vector<int> sourceTopologies(this->getMemoryPool().get());
  sourceTopologies.reserve(hostTopology.nCells);
  for (int outerLayer{0}; outerLayer < NLayers; ++outerLayer) {
    for (int targetCellTopologyId{0}; targetCellTopologyId < hostTopology.nCells; ++targetCellTopologyId) {
      const auto targetCellTopology = hostTopology.getCell(targetCellTopologyId);
      if (targetCellTopology.hitLayerMask.last() != outerLayer) {
        continue;
      }
      const int targetCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[targetCellTopologyId])};
      sourceTopologies.clear();
      size_t sourceCellCount{0};
      for (int sourceCellTopologyId{0}; sourceCellTopologyId < hostTopology.nCells; ++sourceCellTopologyId) {
        const auto sourceCellTopology = hostTopology.getCell(sourceCellTopologyId);
        const int sourceCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[sourceCellTopologyId])};
        if (!sourceCellsNum || sourceCellTopology.secondLink != targetCellTopology.firstLink) {
          continue;
        }
        sourceTopologies.push_back(sourceCellTopologyId);
        sourceCellCount += sourceCellsNum;
      }
      if (!targetCellsNum || sourceTopologies.empty()) {
        mTimeFrameGPU->getNNeighbours()[targetCellTopologyId] = 0;
        mTimeFrameGPU->createNeighboursDevice(targetCellTopologyId, 0);
        mTimeFrameGPU->recordEvent(targetCellTopologyId);
        continue;
      }
      mTimeFrameGPU->createNeighboursLUTDevice(targetCellTopologyId, targetCellsNum);
      auto& stream = mTimeFrameGPU->getStream(targetCellTopologyId);
      int* outputCounter = mTimeFrameGPU->getDeviceNeighboursLUT(targetCellTopologyId) + targetCellsNum;

      const auto key = CapacityEstimator::makeKey(SlabSite::Neighbours, iteration, 0, targetCellTopologyId);
      const auto scale = static_cast<double>(sourceCellCount);
      const int emitted = runOnSlab(mTimeFrameGPU->getCapacityEstimator(), key, scale, [&](const int capacity) {
        mTimeFrameGPU->createNeighboursDevice(targetCellTopologyId, capacity);
        resetOutputCounterHandler(outputCounter, stream);
        for (const int sourceCellTopologyId : sourceTopologies) {
          mTimeFrameGPU->waitEvent(targetCellTopologyId, sourceCellTopologyId);
          TrackingKernels<NLayers>::computeCellNeighboursHandler(mTimeFrameGPU->getDeviceArrayCells(),
                                                                 mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                                                 mTimeFrameGPU->getDeviceNeighbours(targetCellTopologyId),
                                                                 outputCounter,
                                                                 capacity,
                                                                 sourceCellTopologyId,
                                                                 targetCellTopologyId,
                                                                 this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                                                 this->mBz,
                                                                 mTimeFrameGPU->getNCells()[sourceCellTopologyId],
                                                                 mTimeFrameGPU->getFrameworkAllocator(),
                                                                 stream);
        }
        return finalizeCellNeighboursHandler(mTimeFrameGPU->getDeviceNeighbours(targetCellTopologyId),
                                             mTimeFrameGPU->getDeviceNeighboursLUT(targetCellTopologyId),
                                             targetCellsNum,
                                             capacity,
                                             mTimeFrameGPU->getFrameworkAllocator(),
                                             stream);
      });
      mTimeFrameGPU->getNNeighbours()[targetCellTopologyId] = emitted;
      mTimeFrameGPU->recordEvent(targetCellTopologyId);
    }
  }
  mTimeFrameGPU->syncStreams(false);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::findRoads(const int iteration)
{
  bounded_vector<bounded_vector<int>> firstClusters(this->mTrkParams[iteration].NLayers, bounded_vector<int>(this->getMemoryPool().get()), this->getMemoryPool().get());
  firstClusters.resize(this->mTrkParams[iteration].NLayers);
  const auto hostTopology = mTimeFrameGPU->getTrackingTopologyView();
  const bool extendTop = this->mTrkParams[iteration].PassFlags[IterationStep::TrackFollowerTop];
  const bool extendBot = this->mTrkParams[iteration].PassFlags[IterationStep::TrackFollowerBot];
  const bool extendTracks = extendTop || extendBot;
  for (int startLevel{this->mTrkParams[iteration].CellsPerRoad()}; startLevel >= this->mTrkParams[iteration].CellMinimumLevel(); --startLevel) {
    // The cells that may start a road at this level, as the scale the estimator predicts from.
    size_t startCells{0};
    for (int startCellTopologyId{0}; startCellTopologyId < hostTopology.nCells; ++startCellTopologyId) {
      const int startLayer = hostTopology.getCell(startCellTopologyId).hitLayerMask.last();
      if (this->mTrkParams[iteration].StartLayerMask.has(startLayer)) {
        startCells += mTimeFrameGPU->getNCells()[startCellTopologyId];
      }
    }
    if (!startCells) {
      continue;
    }
    const auto key = CapacityEstimator::makeKey(SlabSite::TrackSeeds, iteration, startLevel, 0);
    auto& estimator = mTimeFrameGPU->getCapacityEstimator();
    const int nSeeds = runOnSlab(estimator, key, static_cast<double>(startCells), [&](const int capacity) {
      mTimeFrameGPU->createTrackSeedsDevice(capacity);
      int cursor{0};
      for (int startCellTopologyId{0}; startCellTopologyId < hostTopology.nCells; ++startCellTopologyId) {
        const int startLayer = hostTopology.getCell(startCellTopologyId).hitLayerMask.last();
        if (!(this->mTrkParams[iteration].StartLayerMask.has(startLayer)) || mTimeFrameGPU->getNCells()[startCellTopologyId] == 0) {
          continue;
        }
        TrackingKernels<NLayers>::processNeighboursHandler(startLevel,
                                                           startCellTopologyId,
                                                           mTimeFrameGPU->getDeviceArrayCells(),
                                                           mTimeFrameGPU->getDeviceCells()[startCellTopologyId],
                                                           nullptr,
                                                           nullptr,
                                                           mTimeFrameGPU->getArrayNCells().data(),
                                                           (const uint8_t**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                                           mTimeFrameGPU->getDeviceArrayNeighbours(),
                                                           mTimeFrameGPU->getDeviceArrayNeighboursCellLUT(),
                                                           mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                                           mTimeFrameGPU->getDeviceTrackSeeds(),
                                                           capacity,
                                                           cursor,
                                                           mTimeFrameGPU->getCapacityEstimator(),
                                                           iteration,
                                                           this->mBz,
                                                           this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                                           this->mTrkParams[iteration].MaxChi2NDF,
                                                           this->mTrkParams[iteration].MaxHoles,
                                                           this->mTrkParams[iteration].getMinSeedingClusters(),
                                                           this->mTrkParams[iteration].HoleLayerMask,
                                                           this->mTrkParams[iteration].getNonSeedingLayerMask(),
                                                           mTimeFrameGPU->getDeviceLayerxX0(),
                                                           mTimeFrameGPU->getDevicePropagator(),
                                                           this->mTrkParams[iteration].CorrType,
                                                           mTimeFrameGPU->getFrameworkAllocator());
      }
      return cursor; }, estimator.peakCapacity(key));
    if (!nSeeds) {
      LOGP(debug, "No track seeds found, skipping track finding");
      continue;
    }
    if (extendTracks) { // independent of the slab size, so it must not be redone on a retry
      mTimeFrameGPU->createTrackExtensionScratchDevice(gpu::gridThreads(gpu::ResidentBlocks.fitTrackSeedsExtended),
                                                       this->mTrkParams[iteration].TrackFollowerMaxHypotheses);
    }
    const auto trackKey = CapacityEstimator::makeKey(extendTracks ? SlabSite::TracksExtended : SlabSite::Tracks,
                                                     iteration, startLevel, 0);
    const int nTracks = runOnSlab(estimator, trackKey, static_cast<double>(nSeeds), [&](const int capacity) {
      mTimeFrameGPU->createTrackITSExtDevice(capacity);
      return TrackingKernels<NLayers>::computeTrackSeedHandler(mTimeFrameGPU->getDeviceTrackSeeds(),
                                                               mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                                               mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                                                               mTimeFrameGPU->getDeviceIndexTableUtils(),
                                                               mTimeFrameGPU->getDeviceROFMaskTableView(),
                                                               mTimeFrameGPU->getDeviceROFOverlapTableView(),
                                                               mTimeFrameGPU->getDeviceArrayClusters(),
                                                               (const unsigned char**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                                               mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                                               mTimeFrameGPU->getDeviceROFrameClusters(),
                                                               mTimeFrameGPU->getDeviceTrackITSExt(),
                                                               mTimeFrameGPU->getDeviceTrackIndices(),
                                                               mTimeFrameGPU->getDeviceTrackSeedIndices(),
                                                               mTimeFrameGPU->getDeviceTrackCounter(),
                                                               capacity,
                                                               extendTracks ? mTimeFrameGPU->getDeviceActiveTrackExtensionHypotheses() : nullptr,
                                                               extendTracks ? mTimeFrameGPU->getDeviceNextTrackExtensionHypotheses() : nullptr,
                                                               mTimeFrameGPU->getDeviceLayerRadii(),
                                                               mTimeFrameGPU->getDeviceMinPts(),
                                                               mTimeFrameGPU->getDeviceLayerxX0(),
                                                               static_cast<unsigned int>(nSeeds),
                                                               this->mBz,
                                                               this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                                               this->mTrkParams[iteration].MaxChi2NDF,
                                                               this->mTrkParams[iteration].ReseedIfShorter,
                                                               this->mTrkParams[iteration].RepeatRefitOut,
                                                               this->mTrkParams[iteration].ShiftRefToCluster,
                                                               this->mTrkParams[iteration].NLayers,
                                                               this->mTrkParams[iteration].PhiBins,
                                                               this->mTrkParams[iteration].TrackFollowerMaxHypotheses,
                                                               extendTop,
                                                               extendBot,
                                                               this->mTrkParams[iteration].TrackFollowerNSigmaCutPhi,
                                                               this->mTrkParams[iteration].TrackFollowerNSigmaCutZ,
                                                               mTimeFrameGPU->getDevicePropagator(),
                                                               this->mTrkParams[iteration].CorrType,
                                                               mTimeFrameGPU->getFrameworkAllocator()); }, estimator.peakCapacity(trackKey));
    mTimeFrameGPU->createTrackITSExtHost(nTracks);
    mTimeFrameGPU->downloadTrackITSExtDevice();

    auto& tracks = mTimeFrameGPU->getTrackITSExt();
    const auto& trackIndices = mTimeFrameGPU->getTrackIndices();
    this->acceptTracks(iteration, tracks, trackIndices, firstClusters);
    mTimeFrameGPU->loadUsedClustersDevice();
  }
  this->markTracks(iteration);
  // wipe the artefact memory
  mTimeFrameGPU->popMemoryStack(iteration);
};

template <int NLayers>
int TrackerTraitsGPU<NLayers>::getTFNumberOfClusters() const
{
  return mTimeFrameGPU->getNumberOfClusters();
}

template <int NLayers>
int TrackerTraitsGPU<NLayers>::getTFNumberOfTracklets() const
{
  return std::accumulate(mTimeFrameGPU->getNTracklets().begin(), mTimeFrameGPU->getNTracklets().end(), 0);
}

template <int NLayers>
int TrackerTraitsGPU<NLayers>::getTFNumberOfCells() const
{
  return mTimeFrameGPU->getNumberOfCells();
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::setBz(float bz)
{
  this->mBz = bz;
  mTimeFrameGPU->setBz(bz);
}

template class TrackerTraitsGPU<7>;
#ifdef ENABLE_UPGRADES
template class TrackerTraitsGPU<11>;
template class TrackerTraitsGPU<13>;
#endif
} // namespace o2::its
