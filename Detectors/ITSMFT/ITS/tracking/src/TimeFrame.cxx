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
#include "ITStracking/TrackingConfigParam.h"

#include <iostream>

namespace
{
struct ClusterHelper {
  float phi;
  float r;
  int bin;
  int ind;
};

float MSangle(float mass, float p, float xX0)
{
  float beta = p / std::hypot(mass, p);
  return 0.0136f * std::sqrt(xX0) * (1.f + 0.038f * std::log(xX0)) / (beta * p);
}

float Sq(float v)
{
  return v * v;
}

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
  mClusterExternalIndices.resize(nLayers);
  mUsedClusters.resize(nLayers);
  mROframesClusters.resize(nLayers, {0}); /// TBC: if resetting the timeframe is required, then this has to be done
  mTrackletsIndexROf.resize(2, {0});
}

void TimeFrame::addPrimaryVertices(const std::vector<Vertex>& vertices)
{
  for (const auto& vertex : vertices) {
    mPrimaryVertices.emplace_back(vertex);
    const int w{vertex.getNContributors()};
    mBeamPos[0] = (mBeamPos[0] * mBeamPosWeight + vertex.getX() * w) / (mBeamPosWeight + w);
    mBeamPos[1] = (mBeamPos[1] * mBeamPosWeight + vertex.getY() * w) / (mBeamPosWeight + w);
    mBeamPosWeight += w;
  }
  mROframesPV.push_back(mPrimaryVertices.size());
}

void TimeFrame::addPrimaryVertices(const gsl::span<const Vertex>& vertices)
{
  for (const auto& vertex : vertices) {
    mPrimaryVertices.emplace_back(vertex);
    const int w{vertex.getNContributors()};
    mBeamPos[0] = (mBeamPos[0] * mBeamPosWeight + vertex.getX() * w) / (mBeamPosWeight + w);
    mBeamPos[1] = (mBeamPos[1] * mBeamPosWeight + vertex.getY() * w) / (mBeamPosWeight + w);
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
    addClusterExternalIndexToLayer(layer, first + clusterId);
    clusterId++;
  }

  for (unsigned int iL{0}; iL < mUnsortedClusters.size(); ++iL) {
    mROframesClusters[iL].push_back(mUnsortedClusters[iL].size());
    if (iL < 2) {
      mTrackletsIndexROf[iL].push_back(mUnsortedClusters[1].size()); // Tracklets used in vertexer are always computed starting from L1
    }
  }
  if (mcLabels) {
    mClusterLabels = mcLabels;
  }
  mNrof++;
  return clusters_in_frame.size();
}

int TimeFrame::loadROFrameData(gsl::span<o2::itsmft::ROFRecord> rofs,
                               gsl::span<const itsmft::CompClusterExt> clusters,
                               gsl::span<const unsigned char>::iterator& pattIt,
                               const itsmft::TopologyDictionary* dict,
                               const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  mNrof = 0;
  for (auto& rof : rofs) {
    for (int clusterId{rof.getFirstEntry()}; clusterId < rof.getFirstEntry() + rof.getNEntries(); ++clusterId) {
      auto& c = clusters[clusterId];

      int layer = geom->getLayer(c.getSensorID());

      auto pattID = c.getPatternID();
      o2::math_utils::Point3D<float> locXYZ;
      float sigmaY2 = DefClusError2Row, sigmaZ2 = DefClusError2Col, sigmaYZ = 0; // Dummy COG errors (about half pixel size)
      if (pattID != itsmft::CompCluster::InvalidPatternID) {
        sigmaY2 = dict->getErr2X(pattID);
        sigmaZ2 = dict->getErr2Z(pattID);
        if (!dict->isGroup(pattID)) {
          locXYZ = dict->getClusterCoordinates(c);
        } else {
          o2::itsmft::ClusterPattern patt(pattIt);
          locXYZ = dict->getClusterCoordinates(c, patt);
        }
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        locXYZ = dict->getClusterCoordinates(c, patt, false);
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
      addClusterExternalIndexToLayer(layer, clusterId);
    }
    for (unsigned int iL{0}; iL < mUnsortedClusters.size(); ++iL) {
      mROframesClusters[iL].push_back(mUnsortedClusters[iL].size());
    }
    mNrof++;
  }

  for (auto& v : mNTrackletsPerCluster) {
    v.resize(mUnsortedClusters[1].size());
  }

  if (mcLabels) {
    mClusterLabels = mcLabels;
  }

  return mNrof;
}

int TimeFrame::getTotalClusters() const
{
  size_t totalClusters{0};
  for (auto& clusters : mUnsortedClusters) {
    totalClusters += clusters.size();
  }
  return int(totalClusters);
}

void TimeFrame::initialise(const int iteration, const MemoryParameters& memParam, const TrackingParameters& trkParam, const int maxLayers)
{
  if (iteration == 0) {
    mTracks.clear();
    mTracksLabel.clear();
    mTracks.resize(mNrof);
    mTracksLabel.resize(mNrof);
    mCells.resize(trkParam.CellsPerRoad());
    mCellsLookupTable.resize(trkParam.CellsPerRoad() - 1);
    mCellsNeighbours.resize(trkParam.CellsPerRoad() - 1);
    mCellLabels.resize(trkParam.CellsPerRoad());
    mTracklets.resize(std::min(trkParam.TrackletsPerRoad(), maxLayers - 1));
    mTrackletLabels.resize(trkParam.TrackletsPerRoad());
    mTrackletsLookupTable.resize(trkParam.CellsPerRoad());
    mIndexTables.clear();
    mIndexTablesL0.clear();
    mIndexTableUtils.setTrackingParameters(trkParam);
    mPositionResolution.resize(trkParam.NLayers);
    mBogusClusters.resize(trkParam.NLayers, 0);
    mLines.clear();
    mTrackletClusters.clear();
    for (unsigned int iLayer{0}; iLayer < std::min((int)mClusters.size(), maxLayers); ++iLayer) {
      if (mClusters[iLayer].size()) {
        continue;
      }
      mClusters[iLayer].clear();
      mClusters[iLayer].resize(mUnsortedClusters[iLayer].size());
      mUsedClusters[iLayer].clear();
      mUsedClusters[iLayer].resize(mUnsortedClusters[iLayer].size(), false);
      mPositionResolution[iLayer] = std::hypot(trkParam.LayerMisalignment[iLayer], trkParam.LayerResolution[iLayer]);
    }
    mIndexTables.resize(mNrof);
    mIndexTablesL0.resize(mNrof, std::vector<int>(trkParam.ZBins * trkParam.PhiBins + 1, 0));
    mLines.resize(mNrof);
    mTrackletClusters.resize(mNrof);
    mNTrackletsPerROf.resize(2, std::vector<int>(mNrof + 1, 0));

    std::vector<ClusterHelper> cHelper;
    std::vector<int> clsPerBin(trkParam.PhiBins * trkParam.ZBins, 0);
    for (int rof{0}; rof < mNrof; ++rof) {
      mIndexTables[rof].resize(trkParam.TrackletsPerRoad(), std::vector<int>(trkParam.ZBins * trkParam.PhiBins + 1, 0));
      if (mMultiplicityCutMask.size() == mNrof && !mMultiplicityCutMask[rof]) {
        continue;
      }
      for (int iLayer{0}; iLayer < std::min(trkParam.NLayers, maxLayers); ++iLayer) {
        std::fill(clsPerBin.begin(), clsPerBin.end(), 0);
        const auto unsortedClusters{getUnsortedClustersOnLayer(rof, iLayer)};
        const int clustersNum{static_cast<int>(unsortedClusters.size())};

        cHelper.clear();
        cHelper.resize(clustersNum);

        for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {
          const Cluster& c = unsortedClusters[iCluster];
          ClusterHelper& h = cHelper[iCluster];
          float x = c.xCoordinate - mBeamPos[0];
          float y = c.yCoordinate - mBeamPos[1];
          const float& z = c.zCoordinate;
          float phi = math_utils::computePhi(x, y);
          int zBin{mIndexTableUtils.getZBinIndex(iLayer, z)};
          if (zBin < 0) {
            zBin = 0;
            mBogusClusters[iLayer]++;
          } else if (zBin >= trkParam.ZBins) {
            zBin = trkParam.ZBins - 1;
            mBogusClusters[iLayer]++;
          }
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
        } else { // LUTs on layer 0 are only for vertexer
          for (unsigned int iB{0}; iB < clsPerBin.size(); ++iB) {
            mIndexTablesL0[rof][iB] = lutPerBin[iB];
          }
          for (auto iB{clsPerBin.size()}; iB < (int)mIndexTablesL0[rof].size(); iB++) {
            mIndexTablesL0[rof][iB] = clustersNum;
          }
        }
      }
    }
  }

  mRoads.clear();
  mRoadLabels.clear();

  mMSangles.resize(trkParam.NLayers);
  mPhiCuts.resize(mClusters.size() - 1, 0.f);

  float oneOverR{0.001f * 0.3f * std::abs(mBz) / trkParam.TrackletMinPt};
  for (unsigned int iLayer{0}; iLayer < mClusters.size(); ++iLayer) {
    mMSangles[iLayer] = MSangle(0.14f, trkParam.TrackletMinPt, trkParam.LayerxX0[iLayer]);
    mPositionResolution[iLayer] = std::hypot(trkParam.LayerMisalignment[iLayer], trkParam.LayerResolution[iLayer]);

    if (iLayer < mClusters.size() - 1) {
      const float& r1 = trkParam.LayerRadii[iLayer];
      const float& r2 = trkParam.LayerRadii[iLayer + 1];
      const float res1 = std::hypot(trkParam.PVres, mPositionResolution[iLayer]);
      const float res2 = std::hypot(trkParam.PVres, mPositionResolution[iLayer + 1]);
      const float cosTheta1half = std::sqrt(1.f - Sq(0.5f * r1 * oneOverR));
      const float cosTheta2half = std::sqrt(1.f - Sq(0.5f * r2 * oneOverR));
      float x = r2 * cosTheta1half - r1 * cosTheta2half;
      float delta = std::sqrt(1. / (1.f - 0.25f * Sq(x * oneOverR)) * (Sq(0.25f * r1 * r2 * Sq(oneOverR) / cosTheta2half + cosTheta1half) * Sq(res1) + Sq(0.25f * r1 * r2 * Sq(oneOverR) / cosTheta1half + cosTheta2half) * Sq(res2)));
      mPhiCuts[iLayer] = std::min(std::asin(0.5f * x * oneOverR) + 2.f * mMSangles[iLayer] + delta, constants::math::Pi * 0.5f);
    }
  }

  for (unsigned int iLayer{0}; iLayer < std::min((int)mTracklets.size(), maxLayers); ++iLayer) {
    mTracklets[iLayer].clear();
    mTrackletLabels[iLayer].clear();
    if (iLayer < mCells.size()) {
      mCells[iLayer].clear();
      mTrackletsLookupTable[iLayer].clear();
      mTrackletsLookupTable[iLayer].resize(mClusters[iLayer + 1].size(), 0);
      mCellLabels[iLayer].clear();
    }

    if (iLayer < mCells.size() - 1) {
      mCellsLookupTable[iLayer].clear();
      mCellsNeighbours[iLayer].clear();
    }
  }
}

unsigned long TimeFrame::getArtefactsMemory()
{
  unsigned long size{0};
  for (auto& trkl : mTracklets) {
    size += sizeof(Tracklet) * trkl.size();
  }
  for (auto& cells : mCells) {
    size += sizeof(Cell) * cells.size();
  }
  for (auto& cellsN : mCellsNeighbours) {
    for (auto& vec : cellsN) {
      size += sizeof(int) * vec.size();
    }
  }
  return size + sizeof(Road) * mRoads.size();
}

void TimeFrame::checkTrackletLUTs()
{
  for (uint32_t iLayer{0}; iLayer < getTracklets().size(); ++iLayer) {
    int prev{-1};
    int count{0};
    for (uint32_t iTracklet{0}; iTracklet < getTracklets()[iLayer].size(); ++iTracklet) {
      auto& trk = getTracklets()[iLayer][iTracklet];
      int currentId{trk.firstClusterIndex};
      if (currentId < prev) {
        std::cout << "First Cluster Index not increasing monotonically on L:T:ID:Prev " << iLayer << "\t" << iTracklet << "\t" << currentId << "\t" << prev << std::endl;
      } else if (currentId == prev) {
        count++;
      } else {
        if (iLayer > 0) {
          auto& lut{getTrackletsLookupTable()[iLayer - 1]};
          if (count != lut[prev + 1] - lut[prev]) {
            std::cout << "LUT count broken " << iLayer - 1 << "\t" << prev << "\t" << count << "\t" << lut[prev + 1] << "\t" << lut[prev] << std::endl;
          }
        }
        count = 1;
      }
      prev = currentId;
      if (iLayer > 0) {
        auto& lut{getTrackletsLookupTable()[iLayer - 1]};
        if (iTracklet >= lut[currentId + 1] || iTracklet < lut[currentId]) {
          std::cout << "LUT broken: " << iLayer - 1 << "\t" << currentId << "\t" << iTracklet << std::endl;
        }
      }
    }
  }
}

void TimeFrame::printTrackletLUTonLayer(int i)
{
  std::cout << "--------" << std::endl
            << "Tracklet LUT " << i << std::endl;
  for (int j : mTrackletsLookupTable[i]) {
    std::cout << j << "\t";
  }
  std::cout << "\n--------" << std::endl
            << std::endl;
}

void TimeFrame::printCellLUTonLayer(int i)
{
  std::cout << "--------" << std::endl
            << "Cell LUT " << i << std::endl;
  for (int j : mCellsLookupTable[i]) {
    std::cout << j << "\t";
  }
  std::cout << "\n--------" << std::endl
            << std::endl;
}

void TimeFrame::printTrackletLUTs()
{
  for (unsigned int i{0}; i < mTrackletsLookupTable.size(); ++i) {
    printTrackletLUTonLayer(i);
  }
}

void TimeFrame::printCellLUTs()
{
  for (unsigned int i{0}; i < mCellsLookupTable.size(); ++i) {
    printCellLUTonLayer(i);
  }
}

void TimeFrame::printVertices()
{
  std::cout << "Vertices in ROF (nROF = " << mNrof << ", lut size = " << mROframesPV.size() << ")" << std::endl;
  for (unsigned int iR{0}; iR < mROframesPV.size(); ++iR) {
    std::cout << mROframesPV[iR] << "\t";
  }
  std::cout << "\n\n Vertices:" << std::endl;
  for (unsigned int iV{0}; iV < mPrimaryVertices.size(); ++iV) {
    std::cout << mPrimaryVertices[iV].getX() << "\t" << mPrimaryVertices[iV].getY() << "\t" << mPrimaryVertices[iV].getZ() << std::endl;
  }
  std::cout << "--------" << std::endl;
}

void TimeFrame::printROFoffsets()
{
  std::cout << "--------" << std::endl;
  for (unsigned int iLayer{0}; iLayer < mROframesClusters.size(); ++iLayer) {
    std::cout << "Layer " << iLayer << std::endl;
    for (auto value : mROframesClusters[iLayer]) {
      std::cout << value << "\t";
    }
    std::cout << std::endl;
  }
}

void TimeFrame::computeTrackletsScans()
{
  std::exclusive_scan(mNTrackletsPerROf[0].begin(), mNTrackletsPerROf[0].end(), mNTrackletsPerROf[0].begin(), 0);
  std::exclusive_scan(mNTrackletsPerROf[1].begin(), mNTrackletsPerROf[1].end(), mNTrackletsPerROf[1].begin(), 0);
}

} // namespace its
} // namespace o2
