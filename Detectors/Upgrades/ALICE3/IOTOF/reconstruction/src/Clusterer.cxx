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

#include "Framework/Logger.h"

#include "IOTOFReconstruction/Clusterer.h"

#include <algorithm>
#include <numeric>

namespace o2::iotof
{

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
  LOG(info) << "Running clusterizer on " << digitROFs.size() << " ROFs, total digits: " << digits.size();

  if (!mThread) {
    mThread = std::make_unique<ClustererThread>(this);
  }

  for (size_t iROF = 0; iROF < digitROFs.size(); ++iROF) {
    LOG(info) << "[Clusterer] Processing digit ROF " << iROF << "/" << digitROFs.size();
    const auto& digitsThisROF = digitROFs[iROF];
    const auto nStoredCls = static_cast<int>(clusters.size());
    const int first = digitsThisROF.getFirstEntry();
    const int nDigits = digitsThisROF.getNEntries();

    if (nDigits == 0) {
      LOG(info) << "[Clusterer] Digit ROF " << iROF << " has no entries, skipping";
      clusterROFs.emplace_back(digitsThisROF.getBCData(), digitsThisROF.getROFrame(), nStoredCls, 0);
      continue;
    }

    // Sort digit indices within this ROF by (chipID, row, col, time)
    // extended with time information from TRK.
    mSortIdx.resize(nDigits);
    std::iota(mSortIdx.begin(), mSortIdx.end(), first);
    std::sort(mSortIdx.begin(), mSortIdx.end(), [&digits](int a, int b) {
      const auto& da = digits[a];
      const auto& db = digits[b];
      if (da.getChipIndex() != db.getChipIndex()) {
        return da.getChipIndex() < db.getChipIndex();
      }
      if (da.getRow() != db.getRow()) {
        return da.getRow() < db.getRow();
      }
      if (da.getColumn() != db.getColumn()) {
        return da.getColumn() < db.getColumn();
      }
      return da.getTime() < db.getTime();
    });
    LOG(debug) << "Found " << nDigits << " digits for ROF " << iROF;

    // Process blocks of digits within the same chip (marked by chipID)
    int iDigit = 0;
    while (iDigit < nDigits) {
      const int firstDigit = iDigit;
      const uint16_t chipID = digits[mSortIdx[iDigit]].getChipIndex();

      // Define the span of digits featuring the same chipID
      while (iDigit < nDigits && digits[mSortIdx[iDigit]].getChipIndex() == chipID) {
        ++iDigit;
      }
      const int nDigitsThisChip = iDigit - firstDigit;

      LOG(debug) << "Processing chip " << chipID << " with " << nDigitsThisChip << " digits, next digit starts from index " << iDigit;
      mThread->processChip(digits, firstDigit, nDigitsThisChip, &clusters, &patterns, digitLabels, clusterLabels);
    }

    LOG(debug) << "Finished processing digit ROF " << iROF << ", produced " << (clusters.size() - nStoredCls) << " clusters";
    clusterROFs.emplace_back(digitsThisROF.getBCData(), digitsThisROF.getROFrame(),
                             nStoredCls, static_cast<int>(clusters.size()) - nStoredCls);
  }

  LOG(info) << "Finished processing all digit ROFs, total clusters produced: " << clusters.size();
  if (clusterMC2ROFs && !digMC2ROFs.empty()) {
    clusterMC2ROFs->reserve(clusterMC2ROFs->size() + digMC2ROFs.size());
    for (const auto& in : digMC2ROFs) {
      clusterMC2ROFs->emplace_back(in.eventRecordID, in.rofRecordID, in.minROF, in.maxROF);
    }
  }

  LOG(info) << "Writing cluster topology map to file TF3ClusterTopologies.root";
  mThread->writeTopologiesToFile("TF3ClusterTopologies.root");
}

//__________________________________________________
void Clusterer::ClustererThread::processChip(gsl::span<const Digit> digits,
                                             int firstDigitIdx, int nDigits,
                                             std::vector<Cluster>* clustersOut,
                                             std::vector<unsigned char>* patternsOut,
                                             const ConstDigitTruth* labelsDigPtr,
                                             ClusterTruth* labelsClusPtr)
{
  // firstDigitIdx and nDigits are relative to mSortIdx (i.e. mSortIdx[firstDigitIdx..firstDigitIdx+nDigits-1]
  // are the global digit indices for this chip, already sorted by time, col then row).
  // We use parent->mSortIdx to resolve the global index of each pixel.
  const auto& sortIdx = mParent->mSortIdx;
  LOG(info) << "";
  LOG(info) << "----------------- NEW CHIP -----------------";

  if (nDigits == 1) {
    LOG(info) << "[Clusterer] Processing single hit chip";
    findClustersSingleHit(digits, sortIdx[firstDigitIdx], labelsDigPtr, labelsClusPtr);
  } else {
    LOG(info) << "[Clusterer] Processing multi-hit chip with " << nDigits << " hits";
    std::vector<uint32_t> digitIdxs(nDigits);
    std::iota(digitIdxs.begin(), digitIdxs.end(), firstDigitIdx);
    findClustersMultipleHits(digits, gsl::span<const uint32_t>(digitIdxs), labelsDigPtr, labelsClusPtr);
  }

  // Flush per-thread output into the caller's containers
  if (!mClusters.empty()) {
    clustersOut->insert(clustersOut->end(), mClusters.begin(), mClusters.end());
    mClusters.clear();
  }
  if (!mPatterns.empty()) {
    patternsOut->insert(patternsOut->end(), mPatterns.begin(), mPatterns.end());
    mPatterns.clear();
  }
  if (labelsClusPtr && mLabels.getNElements()) {
    labelsClusPtr->mergeAtBack(mLabels);
    mLabels.clear();
  }
}

//__________________________________________________
void Clusterer::ClustererThread::findClustersSingleHit(gsl::span<const Digit> digits,
                                                       uint32_t digitIdx,
                                                       const ConstDigitTruth* labelsDigPtr,
                                                       ClusterTruth* labelsClusPtr)
{
  const auto& digit = digits[digitIdx];
  const uint16_t chipID = digit.getChipIndex();
  const uint16_t row = digit.getRow();
  const uint16_t col = digit.getColumn();
  const time_t time = digit.getTime();

  if (labelsClusPtr) {
    int nMcLabels = 0;
    fetchMCLabels(digitIdx, labelsDigPtr, nMcLabels);
    const auto nStoredCls = static_cast<uint32_t>(mClusters.size());
    for (int i = nMcLabels; i--;) {
      mLabels.addElement(nStoredCls, mLabelsBuff[i]);
    }
  }

  const uint16_t minRow = row;
  const uint16_t minCol = col;
  uint8_t rowSpan{1}, colSpan{1}, clsTopology{0};
  constexpr uint16_t firedDigitsMask = (1U << 0); // 0x0001 (1)
  mClsTopoClassifier.getTopology(firedDigitsMask, minRow, rowSpan, minCol, colSpan, clsTopology);
  // Bit 0 corresponds to (rowOffset=0, colOffset=0) in row-major order
  Cluster cluster(row, col, rowSpan, colSpan, firedDigitsMask, clsTopology, chipID, time);

  LOG(info) << "Pushing back cluster with row: " << row << ", col: " << col << ", rowSpan: " << rowSpan
            << ", colSpan: " << colSpan << ", pattern: " << firedDigitsMask
            << ", topology: " << clsTopology << ", chipID: " << chipID
            << ", time: " << time;

  mClusters.emplace_back(cluster);
}

//__________________________________________________
void Clusterer::ClustererThread::findClustersMultipleHits(gsl::span<const Digit> digits,
                                                          gsl::span<const uint32_t> digitIdxs,
                                                          const ConstDigitTruth* labelsDigPtr,
                                                          ClusterTruth* labelsClusPtr)
{

  // Constraints on time resolution
  const auto& digitizerParams = o2::iotof::DPLDigitizerParam::Instance();
  float timeResolution = digitizerParams.timeResolution; // in ns
  const auto& clustererParams = o2::iotof::ClustererParam::Instance();
  int maxTimeDiffNSigma = clustererParams.maxTimeDiffNSigma; // in nsigma
  int maxFiredDigitsForCls = clustererParams.maxFiredDigitsForCls; // max fired digits in a cluster

  // Digits are ordered by (chipID, row, col, time) within the same chip, 
  // so we can group them into preclusters based on adjacency in row and column.
  std::vector<std::vector<uint32_t>> preclusters;
  int chipID = digits[digitIdxs[0]].getChipIndex();
  for (const auto& idx : digitIdxs) {
    const auto& digit = digits[idx];
    const uint16_t row = digit.getRow();
    const uint16_t col = digit.getColumn();

    bool addedToPrecluster = false;
    for (auto& precluster : preclusters) {
      const auto& lastDigitIdx = precluster.back();
      const auto& lastDigit = digits[lastDigitIdx];
      if (std::abs(static_cast<int>(lastDigit.getRow()) - static_cast<int>(row)) <= 1 &&
          std::abs(static_cast<int>(lastDigit.getColumn()) - static_cast<int>(col)) <= 1 &&
          std::abs(lastDigit.getTime() - digit.getTime()) <= maxTimeDiffNSigma*timeResolution) {
        precluster.push_back(idx);
        addedToPrecluster = true;
        break;
      }
    }
    if (!addedToPrecluster) {
      preclusters.emplace_back(std::vector<uint32_t>{idx});
    }
  }

  // Debug preclusters
  LOG(info) << "[Clusterer] Found " << preclusters.size() << " preclusters in chip " << chipID;
  for (size_t i = 0; i < preclusters.size(); ++i) {
    LOG(info) << "Precluster " << i << " has " << preclusters[i].size() << " digits";
  }
  LOG(info) << "";

  for (const auto& precluster : preclusters) {
    LOG(info) << "[Clusterer] Processing precluster with " << precluster.size() << " digits";

    const auto nStoredCls = static_cast<uint32_t>(mClusters.size());

    // Single-digit cluster in chip with multiple fired digits
    if (precluster.size() == 1) {
      LOG(info) << "[Clusterer] Processing single-digit precluster in multi-hit chip";
      const auto& digit = digits[precluster[0]];
      const uint16_t chipID = digit.getChipIndex();
      const uint16_t row = digit.getRow();
      const uint16_t col = digit.getColumn();
      const time_t time = digit.getTime();

      if (labelsClusPtr) {
        int nMcLabels = 0;
        fetchMCLabels(precluster[0], labelsDigPtr, nMcLabels);
        for (int i = nMcLabels; i--;) {
          mLabels.addElement(nStoredCls, mLabelsBuff[i]);
        }
      }

      const uint16_t minRow = row;
      const uint16_t minCol = col;
      uint8_t rowSpan{1}, colSpan{1}, clsTopology{0};
      // Bit 0 corresponds to (rowOffset=0, colOffset=0) in row-major order
      constexpr uint16_t firedDigitsMask = (1U << 0); // 0x0001 (1)
      mClsTopoClassifier.getTopology(firedDigitsMask, minRow, rowSpan, minCol, colSpan, clsTopology);
      // Bit 0 corresponds to (rowOffset=0, colOffset=0) in row-major order
      Cluster cluster(row, col, rowSpan, colSpan, firedDigitsMask, clsTopology, chipID, time);

      LOG(info) << "Pushing back cluster with row: " << row << ", col: " << col << ", rowSpan: " << rowSpan
                << ", colSpan: " << colSpan << ", pattern: " << firedDigitsMask
                << ", topology: " << clsTopology << ", chipID: " << chipID
                << ", time: " << time;

      mClusters.emplace_back(cluster);
    } else {
      LOG(info) << "[Clusterer] Processing multi-digit precluster with " << precluster.size() << " digits";
      // Retrieve min row, min col of the precluster
      uint16_t minRow = std::numeric_limits<uint16_t>::max();
      uint16_t maxRow = std::numeric_limits<uint16_t>::min();
      uint16_t minCol = std::numeric_limits<uint16_t>::max();
      uint16_t maxCol = std::numeric_limits<uint16_t>::min();

      int nMcLabels = 0;

      // Compute average time for digits in the precluster
      time_t clsTime = 0.0;
      for (const auto& idx : precluster) {
        const auto& digit = digits[idx];
        minRow = std::min(minRow, digit.getRow());
        minCol = std::min(minCol, digit.getColumn());
        maxRow = std::max(maxRow, digit.getRow());
        maxCol = std::max(maxCol, digit.getColumn());
        clsTime += digit.getTime();
        fetchMCLabels(idx, labelsDigPtr, nMcLabels);
      }
      clsTime /= precluster.size();
      const uint8_t rowSpan = maxRow - minRow + 1;
      const uint8_t colSpan = maxCol - minCol + 1;

      // Fired digits bitmask packed into a single 16-bit pattern variable
      uint16_t firedDigitsMask = 0;

      if (rowSpan * colSpan > maxFiredDigitsForCls) {
        LOG(warn) << "Adding huge precluster with rowSpan=" << rowSpan << ", colSpan=" << colSpan;
        // Overflow precluster: pass InvalidPatternID (or 0) and kHuge topology flag
        Cluster cluster(minRow, minCol, rowSpan, colSpan, Cluster::InvalidPatternID, Topologies::kHuge, chipID, clsTime);
        mClusters.emplace_back(cluster);
        continue;
      }

      // Fill firedDigitsMask in Row-Major order (bit 0 = (minRow, minCol))
      for (const auto& idx : precluster) {
        const auto& digit = digits[idx];
        const uint16_t rowOffset = digit.getRow() - minRow;
        const uint16_t colOffset = digit.getColumn() - minCol;
        
        // Single bit position calculation
        const uint16_t bitIndex = rowOffset * colSpan + colOffset;
        
        // Set bit in LSB-to-MSB order
        if (bitIndex < ClusterInfo::NBitsPattern) {
          firedDigitsMask |= (1U << bitIndex);
        }
      }

      uint8_t clsTopology{0};
      mClsTopoClassifier.getTopology(firedDigitsMask, minRow, rowSpan, minCol, colSpan, clsTopology);

      // Construct and add cluster using scalar pattern mask
      // LOG(info) << "Number of MC labels for this cluster: " << nMcLabels;
      for (int i = nMcLabels; i--;) {
        // LOG(info) << "[Clusterer::findClustersMultipleHits] Adding MC label " << mLabelsBuff[i] << " to cluster at index " << nStoredCls;
        mLabels.addElement(nStoredCls, mLabelsBuff[i]);
      }
      Cluster cluster(minRow, minCol, rowSpan, colSpan, firedDigitsMask, clsTopology, chipID, clsTime);
      LOG(info) << "Pushing back cluster with row: " << minRow << ", col: " << minCol << ", rowSpan: " << rowSpan
          << ", colSpan: " << colSpan << ", pattern: " << firedDigitsMask
          << ", topology: " << Topologies::kSingleDigit << ", chipID: " << chipID
          << ", time: " << clsTime;
      mClusters.emplace_back(cluster);
    }
  }
}

//__________________________________________________
void Clusterer::ClustererThread::fetchMCLabels(uint32_t digID, const ConstDigitTruth* labelsDig, int& nfilled)
{
  // LOG(info) << "[Clusterer::ClustererThread::fetchMCLabels] Fetching MC labels for digit ID: " << digID;
  if (nfilled >= MaxLabels) {
    // LOG(info) << "[Clusterer::ClustererThread::fetchMCLabels] Maximum number of labels (" << MaxLabels << ") already filled, skipping further labels.";
    return;
  }
  if (!labelsDig || digID >= labelsDig->getIndexedSize()) {
    // LOG(info) << "[Clusterer::ClustererThread::fetchMCLabels] No labels found for digit ID: " << digID;
    return;
  }
  const auto& lbls = labelsDig->getLabels(digID);
  // LOG(info) << "[Clusterer::ClustererThread::fetchMCLabels] Digit ID: " << digID << " has " << lbls.size() << " labels";
  for (int i = lbls.size(); i--;) {
    int ic = nfilled;
    for (; ic--;) {
      if (mLabelsBuff[ic] == lbls[i]) {
        // LOG(info) << "[Clusterer::ClustererThread::fetchMCLabels] Label " << lbls[i] << " already present in buffer, skipping.";
        return; // already present
      }
    }
    mLabelsBuff[nfilled++] = lbls[i];
    if (nfilled >= MaxLabels) {
      // LOG(info) << "[Clusterer::ClustererThread::fetchMCLabels] Reached maximum number of labels (" << MaxLabels << "), stopping further label fetching.";
      break;
    }
  }
}

//__________________________________________________
void Clusterer::ClustererThread::writeTopologiesToFile(const char* filename)
{
  mClsTopoClassifier.saveCacheToFile("TF3ClusterTopologies.root");
}


} // namespace o2::iotof
