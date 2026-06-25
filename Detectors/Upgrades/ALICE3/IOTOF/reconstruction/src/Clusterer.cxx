// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.cxx
/// \brief Implementation of the IOTOF cluster finder

#include "IOTOFReconstruction/Clusterer.h"
#include "IOTOFBase/GeometryTGeo.h"
#include "IOTOFSimulation/Segmentation.h"

#include <algorithm>
#include <numeric>

namespace o2::iotof
{

//__________________________________________________
o2::math_utils::Point3D<float> Clusterer::getClusterGlobalCoordinates(const Cluster& cluster, math_utils::Point3D<float>& coords) noexcept
{
  LOG(info) << "[Clusterer] getClusterGlobalCoordinates() called for cluster at chipID " << cluster.chipID
            << ", row " << cluster.row << ", col " << cluster.col;

  Segmentation::Instance()->detectorToLocal(cluster.row, cluster.col, coords, cluster.subDetID);
  LOG(info) << "[Clusterer] Cluster local coordinates: x=" << coords.x() << ", y=" << coords.y() << ", z=" << coords.z();
  GeometryTGeo::Instance()->getMatrixL2G(cluster.subDetID)(coords);

  LOG(info) << "[Clusterer] Cluster global coordinates: x=" << coords.x() << ", y=" << coords.y() << ", z=" << coords.z();
  return coords;
}

//__________________________________________________
o2::math_utils::Point3D<float> Clusterer::getClusterLocalCoordinates(const Cluster& cluster, math_utils::Point3D<float>& coords) noexcept
{
  LOG(info) << "[Clusterer] getClusterLocalCoordinates() called for cluster at chipID " << cluster.chipID
            << ", row " << cluster.row << ", col " << cluster.col;

  Segmentation::Instance()->detectorToLocal(cluster.row, cluster.col, coords, cluster.subDetID);

  LOG(info) << "[Clusterer] Cluster local coordinates: x=" << coords.x() << ", y=" << coords.y() << ", z=" << coords.z();
  return coords;
}

//__________________________________________________
void Clusterer::process(gsl::span<const Digit> digits,
                        gsl::span<const DigROFRecord> digitROFs,
                        std::vector<o2::iotof::Cluster>& clusters,
                        std::vector<unsigned char>& patterns,
                        std::vector<o2::itsmft::ROFRecord>& clusterROFs,
                        const ConstDigitTruth* digitLabels,
                        ClusterTruth* clusterLabels,
                        gsl::span<const DigMC2ROFRecord> digMC2ROFs,
                        std::vector<o2::itsmft::MC2ROFRecord>* clusterMC2ROFs)
{
  LOG(info) << "[Clusterer] Entered process()";
  if (!mThread) {
    mThread = std::make_unique<ClustererThread>(this);
  }

  auto* geom = o2::iotof::GeometryTGeo::Instance();

  LOG(info) << "[Clusterer] Processing " << digitROFs.size() << " digit ROFs, total digits: " << digits.size();
  for (size_t iROF = 0; iROF < digitROFs.size(); ++iROF) {
    LOG(info) << "[Clusterer] Processing digit ROF " << iROF << "/" << digitROFs.size();
    const auto& inROF = digitROFs[iROF];
    const auto outFirst = static_cast<int>(clusters.size());
    const int first = inROF.getFirstEntry();
    const int nEntries = inROF.getNEntries();

    if (nEntries == 0) {
      LOG(info) << "[Clusterer] Digit ROF " << iROF << " has no entries, skipping";
      clusterROFs.emplace_back(inROF.getBCData(), inROF.getROFrame(), outFirst, 0);
      continue;
    }

    // Sort digit indices within this ROF by (chipID, col, row) 
    // chip by chip, column by column (taken from TRK).
    mSortIdx.resize(nEntries);
    std::iota(mSortIdx.begin(), mSortIdx.end(), first);
    std::sort(mSortIdx.begin(), mSortIdx.end(), [&digits](int a, int b) {
      const auto& da = digits[a];
      const auto& db = digits[b];
      if (da.getChipIndex() != db.getChipIndex()) {
        return da.getChipIndex() < db.getChipIndex();
      }
      if (da.getColumn() != db.getColumn()) {
        return da.getColumn() < db.getColumn();
      }
      return da.getRow() < db.getRow();
    });
    LOG(info) << "[Clusterer] Sorted " << nEntries << " digit indices for ROF " << iROF;

    // Process blocks of chips with the same chipID
    int sliceStart = 0;
    while (sliceStart < nEntries) {
      const int chipFirst = sliceStart;
      const uint16_t chipID = digits[mSortIdx[sliceStart]].getChipIndex();
      while (sliceStart < nEntries && digits[mSortIdx[sliceStart]].getChipIndex() == chipID) {
        ++sliceStart;
      }
      const int chipN = sliceStart - chipFirst;

      LOG(info) << "";
      LOG(info) << "[Clusterer] Processing chip " << chipID << " with " << chipN << " digits, next chip start from index " << sliceStart;
      mThread->processChip(digits, chipFirst, chipN, &clusters, &patterns, digitLabels, clusterLabels, geom);
    }

    LOG(info) << "[Clusterer] Finished processing digit ROF " << iROF << ", produced " << (clusters.size() - outFirst) << " clusters";
    clusterROFs.emplace_back(inROF.getBCData(), inROF.getROFrame(),
                             outFirst, static_cast<int>(clusters.size()) - outFirst);
  }

  LOG(info) << "[Clusterer] Finished processing all digit ROFs, total clusters produced: " << clusters.size();
  if (clusterMC2ROFs && !digMC2ROFs.empty()) {
    clusterMC2ROFs->reserve(clusterMC2ROFs->size() + digMC2ROFs.size());
    for (const auto& in : digMC2ROFs) {
      clusterMC2ROFs->emplace_back(in.eventRecordID, in.rofRecordID, in.minROF, in.maxROF);
    }
  }
}

//__________________________________________________
void Clusterer::ClustererThread::processChip(gsl::span<const Digit> digits,
                                             int chipFirst, int chipN,
                                             std::vector<Cluster>* clustersOut,
                                             std::vector<unsigned char>* patternsOut,
                                             const ConstDigitTruth* labelsDigPtr,
                                             ClusterTruth* labelsClusPtr,
                                             GeometryTGeo* geom)
{
  LOG(info) << "";
  LOG(info) << "[Clusterer] Entered processChip() for chip, will process " << chipN << " digits";
  // chipFirst and chipN are relative to mSortIdx (i.e. mSortIdx[chipFirst..chipFirst+chipN-1]
  // are the global digit indices for this chip, already sorted by col then row).
  // We use parent->mSortIdx to resolve the global index of each pixel.
  const auto& sortIdx = parent->mSortIdx;

  // TRK has per-ROF readout, so multiple hits belonging to the same chip, i.e. chipN > 1,
  // are handled with a preclusterer. TF3 still does not have per-ROF readout, so we 
  // use finishChipSingleHitFast on all hits for now.
  for (auto i = 0; i < chipN; ++i) {
    LOG(info) << "[Clusterer] Processing digit " << sortIdx[chipFirst + i] << " ... ";
    finishChipSingleHitFast(digits, sortIdx[chipFirst + i], labelsDigPtr, labelsClusPtr, geom);
  }

  // // TRK logic for per-ROF readout, not used for TF3 yet.
  // if (chipN == 1) {
  //   LOG(info) << "[Clusterer] Processing single hit chip";
  //   finishChipSingleHitFast(digits, sortIdx[chipFirst], labelsDigPtr, labelsClusPtr, geom);
  // } else {
  //   LOG(info) << "[Clusterer] Processing multi-hit chip with " << chipN << " hits";
  //   // Call to initChip()
  //   // Call to updateChip()
  //   // Call to finishChip()
  //   // Code for preclusters needed
  // }

  // Flush per-thread output into the caller's containers
  if (!clusters.empty()) {
    clustersOut->insert(clustersOut->end(), clusters.begin(), clusters.end());
    clusters.clear();
  }
  if (!patterns.empty()) {
    patternsOut->insert(patternsOut->end(), patterns.begin(), patterns.end());
    patterns.clear();
  }
  if (labelsClusPtr && labels.getNElements()) {
    labelsClusPtr->mergeAtBack(labels);
    labels.clear();
  }
}

//__________________________________________________
void Clusterer::ClustererThread::finishChipSingleHitFast(gsl::span<const Digit> digits, uint32_t digitIdx,
                                                         const ConstDigitTruth* labelsDigPtr,
                                                         ClusterTruth* labelsClusPtr,
                                                         GeometryTGeo* geom)
{
  const auto& d = digits[digitIdx];
  const uint16_t chipID = d.getChipIndex();
  const uint16_t row = d.getRow();
  const uint16_t col = d.getColumn();

  if (labelsClusPtr) {
    int nlab = 0;
    fetchMCLabels(digitIdx, labelsDigPtr, nlab);
    const auto cnt = static_cast<uint32_t>(clusters.size());
    for (int i = nlab; i--;) {
      labels.addElement(cnt, labelsBuff[i]);
    }
  }

  // 1×1 pattern: rowSpan=1, colSpan=1, one byte = 0x80
  patterns.emplace_back(1);
  patterns.emplace_back(1);
  patterns.emplace_back(0x80);

  Cluster cluster;
  cluster.chipID = chipID;
  cluster.row = row;
  cluster.col = col;
  cluster.size = 1;
  if (geom) {
    cluster.subDetID = geom->getIOTOFLayer(chipID);
  }
  math_utils::Point3D<float> localClsCoords{0.f, 0.f, 0.f};
  getClusterLocalCoordinates(cluster, localClsCoords);
  // getClusterGlobalCoordinates(cluster, localClsCoords);
  cluster.xCoord = localClsCoords.x();
  cluster.yCoord = localClsCoords.y();
  cluster.zCoord = localClsCoords.z();

  clusters.emplace_back(cluster);
}

//__________________________________________________
void Clusterer::ClustererThread::fetchMCLabels(uint32_t digID, const ConstDigitTruth* labelsDig, int& nfilled)
{
  if (nfilled >= MaxLabels) {
    return;
  }
  if (!labelsDig || digID >= labelsDig->getIndexedSize()) {
    return;
  }
  const auto& lbls = labelsDig->getLabels(digID);
  for (int i = lbls.size(); i--;) {
    int ic = nfilled;
    for (; ic--;) {
      if (labelsBuff[ic] == lbls[i]) {
        return; // already present
      }
    }
    labelsBuff[nfilled++] = lbls[i];
    if (nfilled >= MaxLabels) {
      break;
    }
  }
}

} // namespace o2::iotof
