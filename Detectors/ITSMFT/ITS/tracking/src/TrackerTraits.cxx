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
/// \file TrackerTraits.cxx
/// \brief
///

#include "ITStracking/TrackerTraits.h"

#include <cassert>
#include <iostream>

#include <fmt/format.h>

#include "CommonConstants/MathConstants.h"
#include "DetectorsBase/Propagator.h"
#include "GPUCommonMath.h"
#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Tracklet.h"
#include "ReconstructionDataFormats/Track.h"

using o2::base::PropagatorF;

namespace
{
float Sq(float q)
{
  return q * q;
}
} // namespace

namespace o2
{
namespace its
{

constexpr int debugLevel{0};

void TrackerTraits::computeLayerTracklets()
{
  TimeFrame* tf = mTimeFrame;

#ifdef OPTIMISATION_OUTPUT
  static int iteration{0};
  std::ofstream off(fmt::format("tracklets{}.txt", iteration++));
#endif

  const Vertex diamondVert({mTrkParams.Diamond[0], mTrkParams.Diamond[1], mTrkParams.Diamond[2]}, {25.e-6f, 0.f, 0.f, 25.e-6f, 0.f, 36.f}, 1, 1.f);
  gsl::span<const Vertex> diamondSpan(&diamondVert, 1);
  for (int rof0{0}; rof0 < tf->getNrof(); ++rof0) {
    gsl::span<const Vertex> primaryVertices = mTrkParams.UseDiamond ? diamondSpan : tf->getPrimaryVertices(rof0);
    int minRof = (rof0 >= mTrkParams.DeltaROF) ? rof0 - mTrkParams.DeltaROF : 0;
    int maxRof = (rof0 == tf->getNrof() - mTrkParams.DeltaROF) ? rof0 : rof0 + mTrkParams.DeltaROF;
    for (int iLayer{0}; iLayer < mTrkParams.TrackletsPerRoad(); ++iLayer) {
      gsl::span<const Cluster> layer0 = tf->getClustersOnLayer(rof0, iLayer);
      if (layer0.empty()) {
        continue;
      }
      float meanDeltaR{mTrkParams.LayerRadii[iLayer + 1] - mTrkParams.LayerRadii[iLayer]};

      const int currentLayerClustersNum{static_cast<int>(layer0.size())};
      for (int iCluster{0}; iCluster < currentLayerClustersNum; ++iCluster) {
        const Cluster& currentCluster{layer0[iCluster]};
        const int currentSortedIndex{tf->getSortedIndex(rof0, iLayer, iCluster)};

        if (tf->isClusterUsed(iLayer, currentCluster.clusterId)) {
          continue;
        }
        const float inverseR0{1.f / currentCluster.radius};

        for (auto& primaryVertex : primaryVertices) {
          const float resolution = std::sqrt(Sq(mTrkParams.PVres) / primaryVertex.getNContributors() + Sq(tf->getPositionResolution(iLayer)));

          const float tanLambda{(currentCluster.zCoordinate - primaryVertex.getZ()) * inverseR0};

          const float zAtRmin{tanLambda * (tf->getMinR(iLayer + 1) - currentCluster.radius) + currentCluster.zCoordinate};
          const float zAtRmax{tanLambda * (tf->getMaxR(iLayer + 1) - currentCluster.radius) + currentCluster.zCoordinate};

          const float sqInverseDeltaZ0{1.f / (Sq(currentCluster.zCoordinate - primaryVertex.getZ()) + 2.e-8f)}; /// protecting from overflows adding the detector resolution
          const float sigmaZ{std::sqrt(Sq(resolution) * Sq(tanLambda) * ((Sq(inverseR0) + sqInverseDeltaZ0) * Sq(meanDeltaR) + 1.f) + Sq(meanDeltaR * tf->getMSangle(iLayer)))};

          const int4 selectedBinsRect{getBinsRect(currentCluster, iLayer, zAtRmin, zAtRmax,
                                                  sigmaZ * mTrkParams.NSigmaCut, tf->getPhiCut(iLayer))};

          if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
            continue;
          }

          int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

          if (phiBinsNum < 0) {
            phiBinsNum += mTrkParams.PhiBins;
          }

          for (int rof1{minRof}; rof1 <= maxRof; ++rof1) {
            gsl::span<const Cluster> layer1 = tf->getClustersOnLayer(rof1, iLayer + 1);
            if (layer1.empty()) {
              continue;
            }

            for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
              int iPhiBin = (selectedBinsRect.y + iPhiCount) % mTrkParams.PhiBins;
              const int firstBinIndex{tf->mIndexTableUtils.getBinIndex(selectedBinsRect.x, iPhiBin)};
              const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
              if constexpr (debugLevel) {
                if (firstBinIndex < 0 || firstBinIndex > tf->getIndexTables(rof1)[iLayer + 1].size() ||
                    maxBinIndex < 0 || maxBinIndex > tf->getIndexTables(rof1)[iLayer + 1].size()) {
                  std::cout << iLayer << "\t" << iCluster << "\t" << zAtRmin << "\t" << zAtRmax << "\t" << sigmaZ * mTrkParams.NSigmaCut << "\t" << tf->getPhiCut(iLayer) << std::endl;
                  std::cout << currentCluster.zCoordinate << "\t" << primaryVertex.getZ() << "\t" << currentCluster.radius << std::endl;
                  std::cout << tf->getMinR(iLayer + 1) << "\t" << currentCluster.radius << "\t" << currentCluster.zCoordinate << std::endl;
                  std::cout << "Illegal access to IndexTable " << firstBinIndex << "\t" << maxBinIndex << "\t" << selectedBinsRect.z << "\t" << selectedBinsRect.x << std::endl;
                  exit(1);
                }
              }
              const int firstRowClusterIndex = tf->getIndexTables(rof1)[iLayer + 1][firstBinIndex];
              const int maxRowClusterIndex = tf->getIndexTables(rof1)[iLayer + 1][maxBinIndex];

              for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
                if (iNextCluster >= (int)layer1.size()) {
                  break;
                }
                const Cluster& nextCluster{layer1[iNextCluster]};

                if (tf->isClusterUsed(iLayer + 1, nextCluster.clusterId)) {
                  continue;
                }

                const float deltaPhi{gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi)};
                const float deltaZ{gpu::GPUCommonMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) +
                                                           currentCluster.zCoordinate - nextCluster.zCoordinate)};

#ifdef OPTIMISATION_OUTPUT
                MCCompLabel label;
                int currentId{currentCluster.clusterId};
                int nextId{nextCluster.clusterId};
                for (auto& lab1 : tf->getClusterLabels(iLayer, currentId)) {
                  for (auto& lab2 : tf->getClusterLabels(iLayer + 1, nextId)) {
                    if (lab1 == lab2 && lab1.isValid()) {
                      label = lab1;
                      break;
                    }
                  }
                  if (label.isValid()) {
                    break;
                  }
                }
                off << fmt::format("{}\t{:d}\t{}\t{}\t{}\t{}", iLayer, label.isValid(), (tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate) / sigmaZ, tanLambda, resolution, sigmaZ) << std::endl;
#endif

                if (deltaZ / sigmaZ < mTrkParams.NSigmaCut &&
                    (deltaPhi < tf->getPhiCut(iLayer) ||
                     gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < tf->getPhiCut(iLayer))) {
                  if (iLayer > 0) {
                    tf->getTrackletsLookupTable()[iLayer - 1][currentSortedIndex]++;
                  }
                  const float phi{o2::gpu::GPUCommonMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate,
                                                                currentCluster.xCoordinate - nextCluster.xCoordinate)};
                  const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) /
                                   (currentCluster.radius - nextCluster.radius)};
                  tf->getTracklets()[iLayer].emplace_back(currentSortedIndex, tf->getSortedIndex(rof1, iLayer + 1, iNextCluster), tanL, phi, rof0, rof1);
                }
              }
            }
          }
        }
      }
      if (!tf->checkMemory(mTrkParams.MaxMemory)) {
        return;
      }
    }
  }
  /// Cold code, fixups

  for (int iLayer{0}; iLayer < mTrkParams.CellsPerRoad(); ++iLayer) {
    /// Sort tracklets
    auto& trkl{tf->getTracklets()[iLayer + 1]};
    std::sort(trkl.begin(), trkl.end(), [](const Tracklet& a, const Tracklet& b) {
      return a.firstClusterIndex < b.firstClusterIndex || (a.firstClusterIndex == b.firstClusterIndex && a.secondClusterIndex < b.secondClusterIndex);
    });
    /// Remove duplicates
    auto& lut{tf->getTrackletsLookupTable()[iLayer]};
    int id0{-1}, id1{-1};
    std::vector<Tracklet> newTrk;
    newTrk.reserve(trkl.size());
    for (auto& trk : trkl) {
      if (trk.firstClusterIndex == id0 && trk.secondClusterIndex == id1) {
        lut[id0]--;
      } else {
        id0 = trk.firstClusterIndex;
        id1 = trk.secondClusterIndex;
        newTrk.push_back(trk);
      }
    }
    trkl.swap(newTrk);

    /// Compute LUT
    std::exclusive_scan(lut.begin(), lut.end(), lut.begin(), 0);
    lut.push_back(trkl.size());
  }
  /// Layer 0 is done outside the loop
  std::sort(tf->getTracklets()[0].begin(), tf->getTracklets()[0].end(), [](const Tracklet& a, const Tracklet& b) {
    return a.firstClusterIndex < b.firstClusterIndex || (a.firstClusterIndex == b.firstClusterIndex && a.secondClusterIndex < b.secondClusterIndex);
  });
  int id0{-1}, id1{-1};
  std::vector<Tracklet> newTrk;
  newTrk.reserve(tf->getTracklets()[0].size());
  for (auto& trk : tf->getTracklets()[0]) {
    if (trk.firstClusterIndex != id0 || trk.secondClusterIndex != id1) {
      id0 = trk.firstClusterIndex;
      id1 = trk.secondClusterIndex;
      newTrk.push_back(trk);
    }
  }
  tf->getTracklets()[0].swap(newTrk);

  /// Create tracklets labels
  if (tf->hasMCinformation()) {
    for (int iLayer{0}; iLayer < mTrkParams.TrackletsPerRoad(); ++iLayer) {
      for (auto& trk : tf->getTracklets()[iLayer]) {
        MCCompLabel label;
        int currentId{tf->getClusters()[iLayer][trk.firstClusterIndex].clusterId};
        int nextId{tf->getClusters()[iLayer + 1][trk.secondClusterIndex].clusterId};
        for (auto& lab1 : tf->getClusterLabels(iLayer, currentId)) {
          for (auto& lab2 : tf->getClusterLabels(iLayer + 1, nextId)) {
            if (lab1 == lab2 && lab1.isValid()) {
              label = lab1;
              break;
            }
          }
          if (label.isValid()) {
            break;
          }
        }
        tf->getTrackletsLabel(iLayer).emplace_back(label);
      }
    }
  }
}

void TrackerTraits::computeLayerCells()
{

#ifdef OPTIMISATION_OUTPUT
  static int iteration{0};
  std::ofstream off(fmt::format("cells{}.txt", iteration++));
#endif

  TimeFrame* tf = mTimeFrame;
  for (int iLayer{0}; iLayer < mTrkParams.CellsPerRoad(); ++iLayer) {

    if (tf->getTracklets()[iLayer + 1].empty() ||
        tf->getTracklets()[iLayer].empty()) {
      continue;
    }

    float resolution{std::sqrt(Sq(mTrkParams.LayerMisalignment[iLayer]) + Sq(mTrkParams.LayerMisalignment[iLayer + 1]) + Sq(mTrkParams.LayerMisalignment[iLayer + 2])) / mTrkParams.LayerResolution[iLayer]};
    resolution = resolution > 1.e-12 ? resolution : 1.f;

    const int currentLayerTrackletsNum{static_cast<int>(tf->getTracklets()[iLayer].size())};

    for (int iTracklet{0}; iTracklet < currentLayerTrackletsNum; ++iTracklet) {

      const Tracklet& currentTracklet{tf->getTracklets()[iLayer][iTracklet]};
      const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
      const int nextLayerFirstTrackletIndex{
        tf->getTrackletsLookupTable()[iLayer][nextLayerClusterIndex]};
      const int nextLayerLastTrackletIndex{
        tf->getTrackletsLookupTable()[iLayer][nextLayerClusterIndex + 1]};

      if (nextLayerFirstTrackletIndex == nextLayerLastTrackletIndex) {
        continue;
      }

      for (int iNextTracklet{nextLayerFirstTrackletIndex}; iNextTracklet < nextLayerLastTrackletIndex; ++iNextTracklet) {
        if (tf->getTracklets()[iLayer + 1][iNextTracklet].firstClusterIndex != nextLayerClusterIndex) {
          break;
        }
        const Tracklet& nextTracklet{tf->getTracklets()[iLayer + 1][iNextTracklet]};
        const float deltaTanLambda{std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};
        const float tanLambda{(currentTracklet.tanLambda + nextTracklet.tanLambda) * 0.5f};

#ifdef OPTIMISATION_OUTPUT
        bool good{tf->getTrackletsLabel(iLayer)[iTracklet] == tf->getTrackletsLabel(iLayer + 1)[iNextTracklet]};
        float signedDelta{currentTracklet.tanLambda - nextTracklet.tanLambda};
        off << fmt::format("{}\t{:d}\t{}\t{}\t{}\t{}", iLayer, good, signedDelta, signedDelta / (mTrkParams.CellDeltaTanLambdaSigma), tanLambda, resolution) << std::endl;
#endif

        if (deltaTanLambda / mTrkParams.CellDeltaTanLambdaSigma < mTrkParams.NSigmaCut) {

          if (iLayer > 0 && (int)tf->getCellsLookupTable()[iLayer - 1].size() <= iTracklet) {
            tf->getCellsLookupTable()[iLayer - 1].resize(iTracklet + 1, tf->getCells()[iLayer].size());
          }

          tf->getCells()[iLayer].emplace_back(
            currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex,
            iTracklet, iNextTracklet, tanLambda);
        }
      }
    }
    if (iLayer > 0) {
      tf->getCellsLookupTable()[iLayer - 1].resize(currentLayerTrackletsNum + 1, tf->getCells()[iLayer].size());
    }
    if (!tf->checkMemory(mTrkParams.MaxMemory)) {
      return;
    }
  }

  /// Create cells labels
  if (tf->hasMCinformation()) {
    for (int iLayer{0}; iLayer < mTrkParams.CellsPerRoad(); ++iLayer) {
      for (auto& cell : tf->getCells()[iLayer]) {
        MCCompLabel currentLab{tf->getTrackletsLabel(iLayer)[cell.getFirstTrackletIndex()]};
        MCCompLabel nextLab{tf->getTrackletsLabel(iLayer + 1)[cell.getSecondTrackletIndex()]};
        tf->getCellsLabel(iLayer).emplace_back(currentLab == nextLab ? currentLab : MCCompLabel());
      }
    }
  }

  if constexpr (debugLevel) {
    for (int iLayer{0}; iLayer < mTrkParams.CellsPerRoad(); ++iLayer) {
      std::cout << "Cells on layer " << iLayer << " " << tf->getCells()[iLayer].size() << std::endl;
    }
  }
}

void TrackerTraits::refitTracks(const std::vector<std::vector<TrackingFrameInfo>>& tf, std::vector<TrackITSExt>& tracks)
{
  std::vector<const Cell*> cells;
  for (int iLayer = 0; iLayer < mTrkParams.CellsPerRoad(); iLayer++) {
    cells.push_back(mTimeFrame->getCells()[iLayer].data());
  }
  std::vector<const Cluster*> clusters;
  for (int iLayer = 0; iLayer < mTrkParams.NLayers; iLayer++) {
    clusters.push_back(mTimeFrame->getClusters()[iLayer].data());
  }
  mChainRunITSTrackFit(*mChain, mTimeFrame->getRoads(), clusters, cells, tf, tracks);
}

bool TrackerTraits::trackFollowing(TrackITSExt* track, int rof, bool outward)
{
  auto propInstance = o2::base::Propagator::Instance();
  const int step = -1 + outward * 2;
  const int end = outward ? mTrkParams.NLayers - 1 : 0;
  std::vector<TrackITSExt> hypotheses(1, *track);
  for (auto& hypo : hypotheses) {
    int iLayer = outward ? track->getLastClusterLayer() : track->getFirstClusterLayer();
    while (iLayer != end) {
      iLayer += step;
      const float& r = mTrkParams.LayerRadii[iLayer];
      float x;
      if (!hypo.getXatLabR(r, x, mTimeFrame->getBz(), o2::track::DirAuto)) {
        continue;
      }
      bool success{false};
      auto& hypoParam{outward ? hypo.getParamOut() : hypo.getParamIn()};
      if (!propInstance->propagateToX(hypoParam, x, mTimeFrame->getBz(), PropagatorF::MAX_SIN_PHI,
                                      PropagatorF::MAX_STEP, mTrkParams.CorrType)) {
        continue;
      }

      if (mTrkParams.CorrType == PropagatorF::MatCorrType::USEMatCorrNONE) {
        float radl = 9.36f; // Radiation length of Si [cm]
        float rho = 2.33f;  // Density of Si [g/cm^3]
        if (!hypoParam.correctForMaterial(mTrkParams.LayerxX0[iLayer], mTrkParams.LayerxX0[iLayer] * radl * rho, true)) {
          continue;
        }
      }
      const float phi{hypo.getPhi()};
      const float ePhi{std::sqrt(hypo.getSigmaSnp2() / hypo.getCsp2())};
      const float z{hypo.getZ()};
      const float eZ{std::sqrt(hypo.getSigmaZ2())};
      const int4 selectedBinsRect{getBinsRect(iLayer, phi, mTrkParams.NSigmaCut * ePhi, z, mTrkParams.NSigmaCut * eZ)};

      if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
        continue;
      }

      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

      if (phiBinsNum < 0) {
        phiBinsNum += mTrkParams.PhiBins;
      }

      gsl::span<const Cluster> layer1 = mTimeFrame->getClustersOnLayer(rof, iLayer);
      if (layer1.empty()) {
        continue;
      }

      TrackITSExt currentHypo{hypo}, newHypo{hypo};
      bool first{true};
      for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
        int iPhiBin = (selectedBinsRect.y + iPhiCount) % mTrkParams.PhiBins;
        const int firstBinIndex{mTimeFrame->mIndexTableUtils.getBinIndex(selectedBinsRect.x, iPhiBin)};
        const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
        /// TODO: here we assume that we have the intex tables in layer 0... we currently don't. We must fix this.
        const int firstRowClusterIndex = mTimeFrame->getIndexTables(rof)[iLayer][firstBinIndex];
        const int maxRowClusterIndex = mTimeFrame->getIndexTables(rof)[iLayer][maxBinIndex];

        for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
          if (iNextCluster >= (int)layer1.size()) {
            break;
          }
          const Cluster& nextCluster{layer1[iNextCluster]};

          if (mTimeFrame->isClusterUsed(iLayer, nextCluster.clusterId)) {
            continue;
          }

          const TrackingFrameInfo& trackingHit = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer).at(nextCluster.clusterId);

          TrackITSExt& tbupdated = first ? hypo : newHypo;
          auto& tbuParams = outward ? tbupdated.getParamOut() : tbupdated.getParamIn();
          if (!tbuParams.rotate(trackingHit.alphaTrackingFrame)) {
            continue;
          }

          if (!propInstance->propagateToX(tbuParams, trackingHit.xTrackingFrame, mTimeFrame->getBz(),
                                          PropagatorF::MAX_SIN_PHI, PropagatorF::MAX_STEP, PropagatorF::MatCorrType::USEMatCorrNONE)) {
            continue;
          }

          GPUArray<float, 3> cov{trackingHit.covarianceTrackingFrame};
          cov[0] = std::hypot(cov[0], mTrkParams.LayerMisalignment[iLayer]);
          cov[2] = std::hypot(cov[2], mTrkParams.LayerMisalignment[iLayer]);
          auto predChi2{tbuParams.getPredictedChi2(trackingHit.positionTrackingFrame, cov)};
          if (predChi2 >= track->getChi2() * mTrkParams.NSigmaCut) {
            continue;
          }

          if (!tbuParams.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, cov)) {
            continue;
          }
          tbupdated.setChi2(tbupdated.getChi2() + predChi2); /// This is wrong for outward propagation as the chi2 refers to inward parameters
          tbupdated.setExternalClusterIndex(iLayer, nextCluster.clusterId, true);

          if (!first) {
            hypotheses.emplace_back(tbupdated);
            newHypo = currentHypo;
          }
          first = false;
        }
      }
    }
  }

  TrackITSExt* bestHypo{track};
  bool swapped{false};
  for (auto& hypo : hypotheses) {
    if (hypo.isBetter(*bestHypo, track->getChi2() * mTrkParams.NSigmaCut)) {
      bestHypo = &hypo;
      swapped = true;
    }
  }
  *track = *bestHypo;
  return swapped;
}

TimeFrame* TrackerTraits::getTimeFrameGPU() { return nullptr; }
} // namespace its
} // namespace o2
