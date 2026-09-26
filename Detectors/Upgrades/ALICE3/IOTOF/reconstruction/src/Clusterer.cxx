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
  LOG(info) << "RUNNING CLUSTERIZER ON " << digitROFs.size() << " ROFs, TOTAL DIGITS: " << digits.size();

  if (!mThread) {
    mThread = std::make_unique<ClustererThread>(this);
  }

  for (size_t iROF = 0; iROF < digitROFs.size(); ++iROF) {
    LOG(debug) << "Processing ROF " << iROF << "/" << digitROFs.size();
    const auto& digitsThisROF = digitROFs[iROF];
    const auto nStoredCls = static_cast<int>(clusters.size());
    const int first = digitsThisROF.getFirstEntry();
    const int nDigits = digitsThisROF.getNEntries();

    if (nDigits == 0) {
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

  LOG(info) << "FINISHED PROCESSING ALL DIGIT ROFS, TOTAL CLUSTERS PRODUCED: " << clusters.size();
  if (clusterMC2ROFs && !digMC2ROFs.empty()) {
    clusterMC2ROFs->reserve(clusterMC2ROFs->size() + digMC2ROFs.size());
    for (const auto& in : digMC2ROFs) {
      clusterMC2ROFs->emplace_back(in.eventRecordID, in.rofRecordID, in.minROF, in.maxROF);
    }
  }

  LOG(info) << "WRITING CLUSTER TOPOLOGY MAP TO FILE TF3ClusterTopologies.root";
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

  if (nDigits == 1) {
    findClustersSingleHit(digits, sortIdx[firstDigitIdx], labelsDigPtr, labelsClusPtr);
  } else {
    std::vector<uint32_t> digitIdxs(nDigits);

    for (int i = 0; i < nDigits; ++i) {
      digitIdxs[i] = sortIdx[firstDigitIdx + i];
    }

    findClustersMultipleHits(
      digits,
      gsl::span<const uint32_t>(digitIdxs),
      labelsDigPtr,
      labelsClusPtr);
  }

  // Flush per-thread output into the caller's containers

  // Push-back cluster labels, dummy labels for clusters with
  // empty labels, to ensure that the clusterLabels container
  // WWhas the same size as the clustersOut container.
  if (labelsClusPtr) {
    const size_t base = clustersOut->size(); // before inserting this chip's clusters
    // and store labels as you go, or copy from mLabels:
    for (size_t i = 0; i < mClusters.size(); ++i) {
      auto labels = mLabels.getLabels(i); // empty span if none
      if (labels.empty()) {
        labelsClusPtr->addNoLabelIndex(base + i);
      } else {
        for (const auto& l : labels) {
          labelsClusPtr->addElement(base + i, l);
        }
      }
    }
    mLabels.clear();
  }

  if (!mClusters.empty()) {
    clustersOut->insert(clustersOut->end(), mClusters.begin(), mClusters.end());
    mClusters.clear();
  }
  if (!mPatterns.empty()) {
    patternsOut->insert(patternsOut->end(), mPatterns.begin(), mPatterns.end());
    mPatterns.clear();
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
    int nStoredLabels = 0;
    fetchMCLabels(digitIdx, labelsDigPtr, nStoredLabels);
    const auto nCls = static_cast<uint32_t>(mClusters.size());
    for (int i = 0; i < nStoredLabels; i++) {
      mLabels.addElement(nCls, mLabelsBuff[i]);
    }
  }

  const uint16_t minRow = row;
  const uint16_t minCol = col;
  uint8_t rowSpan{1}, colSpan{1};
  uint32_t clsTopology{0};
  constexpr uint16_t firedDigitsMask = (1U << 0); // 0x0001 (1)
  mClsTopoClassifier.getTopology(firedDigitsMask, minRow, rowSpan, minCol, colSpan, clsTopology);
  // Bit 0 corresponds to (rowOffset=0, colOffset=0) in row-major order
  Cluster cluster(row, col, rowSpan, colSpan, firedDigitsMask, clsTopology, chipID, time);

  LOG(debug) << "Pushing back cluster with row: " << row << ", col: " << col
             << ", rowSpan: " << static_cast<int>(rowSpan) << ", colSpan: " << static_cast<int>(colSpan)
             << ", pattern: " << firedDigitsMask << ", topology: " << static_cast<int>(clsTopology)
             << ", chipID: " << chipID << ", time: " << time;

  mClusters.emplace_back(cluster);
  mPatterns.emplace_back(static_cast<unsigned char>(firedDigitsMask));
}

std::vector<std::vector<uint32_t>> Clusterer::ClustererThread::buildPreclusters(gsl::span<const Digit> digits, gsl::span<const uint32_t> digitIdxs, int maxTimeDiffNSigma, float timeResolution)
{
  std::vector<std::vector<uint32_t>> preclusters;
  std::vector<bool> used(digitIdxs.size(), false);

  auto areNeighbours = [&](const Digit& a, const Digit& b) {
    return std::abs(static_cast<int>(a.getRow()) - static_cast<int>(b.getRow())) <= 1 &&
           std::abs(static_cast<int>(a.getColumn()) - static_cast<int>(b.getColumn())) <= 1 &&
           std::abs(a.getTime() - b.getTime()) <= maxTimeDiffNSigma * timeResolution;
  };

  for (size_t i = 0; i < digitIdxs.size(); ++i) {
    if (used[i]) {
      continue; // already part of an earlier precluster
    }

    std::vector<uint32_t> precluster;
    std::vector<size_t> toVisit{i};
    used[i] = true;

    while (!toVisit.empty()) {
      const size_t cur = toVisit.back();
      toVisit.pop_back();
      precluster.push_back(digitIdxs[cur]);

      // add every not-yet-used digit that touches the current one
      for (size_t j = 0; j < digitIdxs.size(); ++j) {
        if (!used[j] && areNeighbours(digits[digitIdxs[cur]], digits[digitIdxs[j]])) {
          used[j] = true;
          toVisit.push_back(j);
        }
      }
    }

    preclusters.push_back(std::move(precluster));
  }
  return preclusters;
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
  int maxTimeDiffNSigma = clustererParams.maxTimeDiffNSigma;       // in nsigma
  int maxFiredDigitsForCls = clustererParams.maxFiredDigitsForCls; // max fired digits in a cluster

  // Digits are ordered by (chipID, row, col, time) within the same chip,
  // so we can group them into preclusters based on adjacency in row and column.
  std::vector<std::vector<uint32_t>> preclusters = buildPreclusters(digits, digitIdxs, maxTimeDiffNSigma, timeResolution);
  uint16_t chipID = digits[digitIdxs[0]].getChipIndex();

  for (const auto& precluster : preclusters) {

    const auto nStoredCls = static_cast<uint32_t>(mClusters.size());

    // Single-digit cluster in chip with multiple fired digits
    if (precluster.size() == 1) {
      const auto& digit = digits[precluster[0]];
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
      uint8_t rowSpan{1}, colSpan{1};
      uint32_t clsTopology{0};
      // Bit 0 corresponds to (rowOffset=0, colOffset=0) in row-major order
      constexpr uint16_t firedDigitsMask = (1U << 0); // 0x0001 (1)
      mClsTopoClassifier.getTopology(firedDigitsMask, minRow, rowSpan, minCol, colSpan, clsTopology);
      // Bit 0 corresponds to (rowOffset=0, colOffset=0) in row-major order
      Cluster cluster(minRow, minCol, rowSpan, colSpan, firedDigitsMask, clsTopology, chipID, time);

      LOG(debug) << "Pushing back cluster with row: " << row << ", col: " << col
                 << ", rowSpan: " << static_cast<int>(rowSpan) << ", colSpan: " << static_cast<int>(colSpan)
                 << ", pattern: " << firedDigitsMask << ", topology: " << static_cast<int>(clsTopology)
                 << ", chipID: " << chipID << ", time: " << time;

      mClusters.emplace_back(cluster);
      mPatterns.emplace_back(static_cast<unsigned char>(firedDigitsMask));
    } else {
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
        // Overflow precluster: pass InvalidPatternID (or 0) and kHuge topology flag
        Cluster cluster(minRow, minCol, rowSpan, colSpan, Cluster::InvalidPatternID, Topologies::kHuge, chipID, clsTime);
        mClusters.emplace_back(cluster);
        mPatterns.emplace_back(Cluster::InvalidPatternID);
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

      uint32_t clsTopology{0};
      mClsTopoClassifier.getTopology(firedDigitsMask, minRow, rowSpan, minCol, colSpan, clsTopology);

      // Construct and add cluster using scalar pattern mask
      for (int i = 0; i < nMcLabels; i++) {
        mLabels.addElement(nStoredCls, mLabelsBuff[i]);
      }
      Cluster cluster(minRow, minCol, rowSpan, colSpan, firedDigitsMask, clsTopology, chipID, clsTime);
      LOG(debug) << "Pushing back cluster with row: " << minRow << ", col: " << minCol
                 << ", rowSpan: " << static_cast<int>(rowSpan) << ", colSpan: " << static_cast<int>(colSpan)
                 << ", pattern: " << firedDigitsMask << ", topology: " << static_cast<int>(clsTopology)
                 << ", chipID: " << chipID << ", time: " << clsTime;
      mClusters.emplace_back(cluster);
      mPatterns.emplace_back(static_cast<unsigned char>(firedDigitsMask));
    }
  }
}

//__________________________________________________
void Clusterer::ClustererThread::fetchMCLabels(uint32_t digID, const ConstDigitTruth* labelsDig, int& nFilled)
{
  if (!labelsDig || digID >= labelsDig->getIndexedSize()) {
    return;
  }
  auto sortBuffer = [this]() { std::sort(this->mLabelsBuff.begin(), this->mLabelsBuff.end(), [](Label const& a, Label const& b) { return a.getTrackID() < b.getTrackID(); }); };
  for (const auto& label : labelsDig->getLabels(digID)) {
    bool skip = false;
    for (int ic = 0; ic < nFilled; ic++) {
      if (mLabelsBuff[ic] == label) {
        skip = true;
        break;
      }
    }
    if (!skip) {
      if (nFilled < MaxLabels) {
        mLabelsBuff[nFilled++] = label;
        if (nFilled == MaxLabels) {
          sortBuffer();
        }
      } else if (mLabelsBuff.back().getTrackID() > label.getTrackID()) {
        mLabelsBuff.back() = label;
        sortBuffer();
      }
    }
  }
}

//__________________________________________________
void Clusterer::ClustererThread::writeTopologiesToFile(const char* filename)
{
  mClsTopoClassifier.saveCacheToFile("TF3ClusterTopologies.root");
}

} // namespace o2::iotof
