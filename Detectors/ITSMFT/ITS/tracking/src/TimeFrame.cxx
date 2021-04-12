// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file TimeFrame.cxx
/// \brief
///

#include "ITStracking/TimeFrame.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include <iostream>

namespace
{
struct ClusterHelper {
  float phi;
  float r;
  int bin;
  int ind;
};
} // namespace

namespace o2
{
namespace its
{

constexpr float DefClusErrorRow = o2::itsmft::SegmentationAlpide::PitchRow * 0.5;
constexpr float DefClusErrorCol = o2::itsmft::SegmentationAlpide::PitchCol * 0.5;
constexpr float DefClusError2Row = DefClusErrorRow * DefClusErrorRow;
constexpr float DefClusError2Col = DefClusErrorCol * DefClusErrorCol;

TimeFrame::TimeFrame(int nLayers)
{
  mMinR.resize(nLayers, 10000.);
  mMaxR.resize(nLayers, -1.);
  mClusters.resize(nLayers);
  mUnsortedClusters.resize(nLayers);
  mTrackingFrameInfo.resize(nLayers);
  mClusterLabels.resize(nLayers);
  mClusterExternalIndices.resize(nLayers);
  mUsedClusters.resize(nLayers);
  mROframesClusters.resize(nLayers);
}

void TimeFrame::addClusterLabelToLayer(int layer, const MCCompLabel label) { mClusterLabels[layer].emplace_back(label.getRawValue()); }

void TimeFrame::addPrimaryVertices(const std::vector<std::pair<float3, int>>& vertices)
{
  for (const auto& vertex : vertices) {
    mPrimaryVertices.emplace_back(vertex.first);
    const int w{vertex.second};
    mBeamPos[0] = (mBeamPos[0] * mBeamPosWeight + vertex.first.x * w) / (mBeamPosWeight + w);
    mBeamPos[1] = (mBeamPos[1] * mBeamPosWeight + vertex.first.y * w) / (mBeamPosWeight + w);
    mBeamPosWeight += w;
  }
  mROframesPV.push_back(mPrimaryVertices.size());
}

int TimeFrame::loadROFrameData(const o2::itsmft::ROFRecord& rof, gsl::span<const itsmft::Cluster> clusters,
                               const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  int clusterId{0};

  auto first = rof.getFirstEntry();
  auto clusters_in_frame = rof.getROFData(clusters);
  for (auto& c : clusters_in_frame) {
    int layer = geom->getLayer(c.getSensorID());

    /// Clusters are stored in the tracking frame
    auto xyz = c.getXYZGloRot(*geom);
    addTrackingFrameInfoToLayer(layer, xyz.x(), xyz.y(), xyz.z(), c.getX(), geom->getSensorRefAlpha(c.getSensorID()),
                                std::array<float, 2>{c.getY(), c.getZ()},
                                std::array<float, 3>{c.getSigmaY2(), c.getSigmaYZ(), c.getSigmaZ2()});

    /// Rotate to the global frame
    addClusterToLayer(layer, xyz.x(), xyz.y(), xyz.z(), mUnsortedClusters[layer].size());
    if (mcLabels) {
      addClusterLabelToLayer(layer, *(mcLabels->getLabels(first + clusterId).begin()));
    }
    addClusterExternalIndexToLayer(layer, first + clusterId);
    clusterId++;
  }

  for (unsigned int iL{0}; iL < mUnsortedClusters.size(); ++iL)
    mROframesClusters[iL].push_back(mUnsortedClusters[iL].size());
  mNrof++;
  return clusters_in_frame.size();
}

int TimeFrame::loadROFrameData(const std::vector<o2::itsmft::ROFRecord>* rofs, gsl::span<const itsmft::CompClusterExt> clusters, gsl::span<const unsigned char>::iterator& pattIt,
                               const itsmft::TopologyDictionary& dict, const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  mNrof = 0;
  for (int clusterId{0}; clusterId < clusters.size() && mNrof < rofs->size(); ++clusterId) {
    auto& c = clusters[clusterId];

    int layer = geom->getLayer(c.getSensorID());

    auto pattID = c.getPatternID();
    o2::math_utils::Point3D<float> locXYZ;
    float sigmaY2 = DefClusError2Row, sigmaZ2 = DefClusError2Col, sigmaYZ = 0; //Dummy COG errors (about half pixel size)
    if (pattID != itsmft::CompCluster::InvalidPatternID) {
      sigmaY2 = dict.getErr2X(pattID);
      sigmaZ2 = dict.getErr2Z(pattID);
      if (!dict.isGroup(pattID)) {
        locXYZ = dict.getClusterCoordinates(c);
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        locXYZ = dict.getClusterCoordinates(c, patt);
      }
    } else {
      o2::itsmft::ClusterPattern patt(pattIt);
      locXYZ = dict.getClusterCoordinates(c, patt, false);
    }
    auto sensorID = c.getSensorID();
    // Inverse transformation to the local --> tracking
    auto trkXYZ = geom->getMatrixT2L(sensorID) ^ locXYZ;
    // Transformation to the local --> global
    auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;

    addTrackingFrameInfoToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), geom->getSensorRefAlpha(sensorID),
                                std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                std::array<float, 3>{sigmaY2, sigmaYZ, sigmaZ2});

    /// Rotate to the global frame
    addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), mUnsortedClusters[layer].size());
    if (mcLabels) {
      addClusterLabelToLayer(layer, *(mcLabels->getLabels(clusterId).begin()));
    }
    addClusterExternalIndexToLayer(layer, clusterId);

    if (clusterId == rofs->at(mNrof).getFirstEntry() + rofs->at(mNrof).getNEntries() - 1) {
      for (unsigned int iL{0}; iL < mUnsortedClusters.size(); ++iL) {
        mROframesClusters[iL].push_back(mUnsortedClusters[iL].size());
      }
      mNrof++;
    }
  }
  return mNrof;
}

int TimeFrame::getTotalClusters() const
{
  size_t totalClusters{0};
  for (auto& clusters : mUnsortedClusters)
    totalClusters += clusters.size();
  return int(totalClusters);
}

void TimeFrame::initialise(const int iteration, const MemoryParameters& memParam, const TrackingParameters& trkParam)
{
  if (iteration == 0) {

    mCells.resize(trkParam.CellsPerRoad());
    mCellsLookupTable.resize(trkParam.CellsPerRoad() - 1);
    mCellsNeighbours.resize(trkParam.CellsPerRoad() - 1);
    mTracklets.resize(trkParam.TrackletsPerRoad());
    mTrackletsLookupTable.resize(trkParam.CellsPerRoad());
    mIndexTableUtils.setTrackingParameters(trkParam);

    for (unsigned int iLayer{0}; iLayer < mClusters.size(); ++iLayer) {
      mClusters[iLayer].clear();
      mClusters[iLayer].resize(mUnsortedClusters[iLayer].size());
      mUsedClusters[iLayer].clear();
      mUsedClusters[iLayer].resize(mUnsortedClusters[iLayer].size(), false);
    }

    mIndexTables.resize(mNrof);
    std::vector<ClusterHelper> cHelper;
    for (int rof{0}; rof < mNrof; ++rof) {
      mIndexTables[rof].resize(trkParam.TrackletsPerRoad(), std::vector<int>(trkParam.ZBins * trkParam.PhiBins + 1, 0));
      for (int iLayer{0}; iLayer < trkParam.NLayers; ++iLayer) {
        std::vector<int> clsPerBin(trkParam.PhiBins * trkParam.ZBins, 0);

        const auto unsortedClusters{getUnsortedClustersOnLayer(rof, iLayer)};
        const int clustersNum{static_cast<int>(unsortedClusters.size())};

        cHelper.clear();
        cHelper.resize(clustersNum);

        for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {
          const Cluster& c = unsortedClusters[iCluster];
          ClusterHelper& h = cHelper[iCluster];
          float x = c.xCoordinate - mBeamPos[0];
          float y = c.yCoordinate - mBeamPos[1];
          float phi = math_utils::computePhi(x, y);
          const int zBin{mIndexTableUtils.getZBinIndex(iLayer, c.zCoordinate)};
          int bin = mIndexTableUtils.getBinIndex(zBin, mIndexTableUtils.getPhiBinIndex(phi));

          h.phi = phi;
          h.r = math_utils::hypot(x, y);
          mMinR[iLayer] = o2::gpu::GPUCommonMath::Min(h.r, mMinR[iLayer]);
          mMaxR[iLayer] = o2::gpu::GPUCommonMath::Max(h.r, mMaxR[iLayer]);
          h.bin = bin;
          h.ind = clsPerBin[bin]++;
        }
        std::vector<int> lutPerBin(clsPerBin.size());
        lutPerBin[0] = 0;
        for (unsigned int iB{1}; iB < lutPerBin.size(); ++iB) {
          lutPerBin[iB] = lutPerBin[iB - 1] + clsPerBin[iB - 1];
        }

        auto clusters2beSorted{getClustersOnLayer(rof, iLayer)};
        for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {
          const ClusterHelper& h = cHelper[iCluster];

          Cluster& c = clusters2beSorted[lutPerBin[h.bin] + h.ind];
          c = unsortedClusters[iCluster];
          c.phi = h.phi;
          c.radius = h.r;
          c.indexTableBinIndex = h.bin;
        }

        if (iLayer > 0) {
          for (unsigned int iB{0}; iB < clsPerBin.size(); ++iB) {
            mIndexTables[rof][iLayer - 1][iB] = lutPerBin[iB];
          }
          for (auto iB{clsPerBin.size()}; iB < (int)mIndexTables[rof][iLayer - 1].size(); iB++) {
            mIndexTables[rof][iLayer - 1][iB] = clustersNum;
          }
        }
      }
    }
  }

  mRoads.clear();

  for (unsigned int iLayer{0}; iLayer < mClusters.size(); ++iLayer) {
    if (iLayer < mCells.size()) {
      mCells[iLayer].clear();
      float cellsMemorySize =
        memParam.MemoryOffset +
        std::ceil(((memParam.CellsMemoryCoefficients[iLayer] * mUnsortedClusters[iLayer].size()) *
                   mUnsortedClusters[iLayer + 1].size()) *
                  mUnsortedClusters[iLayer + 2].size());

      if (cellsMemorySize > mCells[iLayer].capacity()) {
        mCells[iLayer].reserve(cellsMemorySize);
      }
    }

    if (iLayer < mCells.size() - 1) {
      mCellsLookupTable[iLayer].clear();
      mCellsLookupTable[iLayer].resize(
        std::max(mUnsortedClusters[iLayer + 1].size(), mUnsortedClusters[iLayer + 2].size()) +
          std::ceil((memParam.TrackletsMemoryCoefficients[iLayer + 1] *
                     mUnsortedClusters[iLayer + 1].size()) *
                    mUnsortedClusters[iLayer + 2].size()),
        constants::its::UnusedIndex);
      mCellsNeighbours[iLayer].clear();
    }
  }

  for (unsigned int iLayer{0}; iLayer < mTracklets.size(); ++iLayer) {
    float trackletsMemorySize =
      std::max(mUnsortedClusters[iLayer].size(), mUnsortedClusters[iLayer + 1].size()) +
      std::ceil((memParam.TrackletsMemoryCoefficients[iLayer] * mUnsortedClusters[iLayer].size()) *
                mUnsortedClusters[iLayer + 1].size());

    if (trackletsMemorySize > mTracklets[iLayer].capacity()) {
      mTracklets[iLayer].reserve(trackletsMemorySize);
    }

    if (iLayer < mCells.size()) {
      mTrackletsLookupTable[iLayer].clear();
      mTrackletsLookupTable[iLayer].resize(mUnsortedClusters[iLayer + 1].size(), constants::its::UnusedIndex);
    }
  }
}

} // namespace its
} // namespace o2
