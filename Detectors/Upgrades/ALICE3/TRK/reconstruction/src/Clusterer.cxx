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
/// \brief Implementation of the TRK cluster finder

#include "TRKReconstruction/Clusterer.h"
#include "TRKBase/GeometryTGeo.h"

#include <algorithm>
#include <numeric>

namespace o2::trk
{

//__________________________________________________
void Clusterer::process(gsl::span<const Digit> digits,
                        gsl::span<const DigROFRecord> digitROFs,
                        std::vector<o2::trk::Cluster>& clusters,
                        std::vector<unsigned char>& patterns,
                        std::vector<o2::trk::ROFRecord>& clusterROFs,
                        const ConstDigitTruth* digitLabels,
                        ClusterTruth* clusterLabels)
{
  if (!mThread) {
    mThread = std::make_unique<ClustererThread>(this);
  }

  auto* geom = o2::trk::GeometryTGeo::Instance();

  for (size_t iROF = 0; iROF < digitROFs.size(); ++iROF) {
    const auto& inROF = digitROFs[iROF];
    const auto outFirst = static_cast<int>(clusters.size());
    const int first = inROF.getFirstEntry();
    const int nEntries = inROF.getNEntries();

    if (nEntries == 0) {
      clusterROFs.emplace_back(inROF.getBCData(), inROF.getROFrame(), outFirst, 0);
      continue;
    }

    // Sort digit indices within this ROF by (chipID, col, row) so we can process
    // chip by chip, column by column -- the same ordering the ALPIDE scanner expects.
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

    // Process one chip at a time
    int sliceStart = 0;
    while (sliceStart < nEntries) {
      const int chipFirst = sliceStart;
      const uint16_t chipID = digits[mSortIdx[sliceStart]].getChipIndex();
      while (sliceStart < nEntries && digits[mSortIdx[sliceStart]].getChipIndex() == chipID) {
        ++sliceStart;
      }
      const int chipN = sliceStart - chipFirst;

      mThread->processChip(digits, chipFirst, chipN, &clusters, &patterns, digitLabels, clusterLabels, geom);
    }

    clusterROFs.emplace_back(inROF.getBCData(), inROF.getROFrame(),
                             outFirst, static_cast<int>(clusters.size()) - outFirst);
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
  // chipFirst and chipN are relative to mSortIdx (i.e. mSortIdx[chipFirst..chipFirst+chipN-1]
  // are the global digit indices for this chip, already sorted by col then row).
  // We use parent->mSortIdx to resolve the global index of each pixel.
  const auto& sortIdx = parent->mSortIdx;

  if (chipN == 1) {
    finishChipSingleHitFast(digits, sortIdx[chipFirst], labelsDigPtr, labelsClusPtr, geom);
  } else {
    initChip(digits, sortIdx[chipFirst], geom);
    for (int i = chipFirst + 1; i < chipFirst + chipN; ++i) {
      updateChip(digits, sortIdx[i]);
    }
    finishChip(digits, labelsDigPtr, labelsClusPtr, geom);
  }

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
void Clusterer::ClustererThread::initChip(gsl::span<const Digit> digits, uint32_t first, GeometryTGeo* geom)
{
  const uint16_t chipID = digits[first].getChipIndex();

  // Determine the number of rows for this chip's sensor type
  size = constants::moduleMLOT::chip::nRows + 2; // default for ML/OT
  if (geom) {
    if (geom->getSubDetID(chipID) == 0) { // VD
      const int layer = geom->getLayer(chipID);
      size = constants::VD::petal::layer::nRows[layer] + 2;
    }
  }

  delete[] column1;
  delete[] column2;
  column1 = new int[size];
  column2 = new int[size];
  column1[0] = column1[size - 1] = -1;
  column2[0] = column2[size - 1] = -1;
  prev = column1 + 1;
  curr = column2 + 1;
  resetColumn(curr);

  pixels.clear();
  preClusterHeads.clear();
  preClusterIndices.clear();

  const auto& pix = digits[first];
  currCol = pix.getColumn();
  curr[pix.getRow()] = 0;
  preClusterHeads.push_back(0);
  preClusterIndices.push_back(0);
  pixels.emplace_back(-1, first);
  noLeftCol = true;
}

//__________________________________________________
void Clusterer::ClustererThread::updateChip(gsl::span<const Digit> digits, uint32_t ip)
{
  const auto& pix = digits[ip];
  uint16_t row = pix.getRow();

  if (currCol != pix.getColumn()) {
    swapColumnBuffers();
    resetColumn(curr);
    noLeftCol = false;
    if (pix.getColumn() > currCol + 1) {
      // gap: no connection with previous column
      currCol = pix.getColumn();
      addNewPreCluster(ip, row);
      noLeftCol = true;
      return;
    }
    currCol = pix.getColumn();
  }

  bool orphan = true;

  if (noLeftCol) {
    if (curr[row - 1] >= 0) {
      expandPreCluster(ip, row, curr[row - 1]);
      return;
    }
  } else {
#ifdef _ALLOW_DIAGONAL_TRK_CLUSTERS_
    int neighbours[]{curr[row - 1], prev[row], prev[row + 1], prev[row - 1]};
#else
    int neighbours[]{curr[row - 1], prev[row]};
#endif
    for (auto pci : neighbours) {
      if (pci < 0) {
        continue;
      }
      if (orphan) {
        expandPreCluster(ip, row, pci);
        orphan = false;
        continue;
      }
      // merge two pre-clusters: assign the smaller index to both
      if (preClusterIndices[pci] < preClusterIndices[curr[row]]) {
        preClusterIndices[curr[row]] = preClusterIndices[pci];
      } else {
        preClusterIndices[pci] = preClusterIndices[curr[row]];
      }
    }
  }
  if (orphan) {
    addNewPreCluster(ip, row);
  }
}

//__________________________________________________
void Clusterer::ClustererThread::finishChip(gsl::span<const Digit> digits,
                                            const ConstDigitTruth* labelsDigPtr,
                                            ClusterTruth* labelsClusPtr,
                                            GeometryTGeo* geom)
{
  const uint16_t chipID = digits[pixels[0].second].getChipIndex();

  for (size_t i1 = 0; i1 < preClusterHeads.size(); ++i1) {
    auto ci = preClusterIndices[i1];
    if (ci < 0) {
      continue;
    }
    BBox bbox(chipID);
    int nlab = 0;
    uint32_t totalCharge = 0;
    pixArrBuff.clear();

    // Walk the linked list for this pre-cluster head
    auto collectPixels = [&](int head) {
      int next = head;
      while (next >= 0) {
        const auto& pixEntry = pixels[next];
        const auto& d = digits[pixEntry.second];
        uint16_t r = d.getRow(), c = d.getColumn();
        pixArrBuff.emplace_back(r, c);
        bbox.adjust(r, c);
        totalCharge += d.getCharge();
        if (labelsClusPtr) {
          fetchMCLabels(pixEntry.second, labelsDigPtr, nlab);
        }
        next = pixEntry.first;
      }
    };

    collectPixels(preClusterHeads[i1]);
    preClusterIndices[i1] = -1;

    for (size_t i2 = i1 + 1; i2 < preClusterHeads.size(); ++i2) {
      if (preClusterIndices[i2] != ci) {
        continue;
      }
      collectPixels(preClusterHeads[i2]);
      preClusterIndices[i2] = -1;
    }

    // Determine geometry info
    int subDetID = -1, layer = -1, disk = -1;
    if (geom) {
      subDetID = geom->getSubDetID(chipID);
      layer = geom->getLayer(chipID);
      disk = geom->getDisk(chipID);
    }

    const bool doLabels = (labelsClusPtr != nullptr);
    if (bbox.isAcceptableSize()) {
      streamCluster(bbox, pixArrBuff, totalCharge, doLabels, nlab, chipID, subDetID, layer, disk);
    } else {
      // Huge cluster: split into MaxRowSpan x MaxColSpan tiles (same as ITS3)
      auto warnLeft = MaxHugeClusWarn - parent->mNHugeClus;
      if (warnLeft > 0) {
        LOGP(warn, "Splitting huge TRK cluster: chipID {}, rows {}:{} cols {}:{}{}",
             chipID, bbox.rowMin, bbox.rowMax, bbox.colMin, bbox.colMax,
             warnLeft == 1 ? " (further warnings muted)" : "");
        parent->mNHugeClus++;
      }
      BBox bboxT(chipID);
      bboxT.colMin = bbox.colMin;
      do {
        bboxT.rowMin = bbox.rowMin;
        bboxT.colMax = std::min(bbox.colMax, uint16_t(bboxT.colMin + o2::itsmft::ClusterPattern::MaxColSpan - 1));
        do {
          bboxT.rowMax = std::min(bbox.rowMax, uint16_t(bboxT.rowMin + o2::itsmft::ClusterPattern::MaxRowSpan - 1));
          std::vector<std::pair<uint16_t, uint16_t>> subPix;
          uint32_t subCharge = 0;
          for (const auto& [r, c] : pixArrBuff) {
            if (bboxT.isInside(r, c)) {
              subPix.emplace_back(r, c);
              subCharge += 1;
            }
          }
          if (!subPix.empty()) {
            streamCluster(bboxT, subPix, subCharge, doLabels, nlab, chipID, subDetID, layer, disk);
          }
          bboxT.rowMin = bboxT.rowMax + 1;
        } while (bboxT.rowMin <= bbox.rowMax);
        bboxT.colMin = bboxT.colMax + 1;
      } while (bboxT.colMin <= bbox.colMax);
    }
  }
  // flush per-thread output to the caller via processChip
}

//__________________________________________________
void Clusterer::ClustererThread::finishChipSingleHitFast(gsl::span<const Digit> digits, uint32_t hit,
                                                         const ConstDigitTruth* labelsDigPtr,
                                                         ClusterTruth* labelsClusPtr,
                                                         GeometryTGeo* geom)
{
  const auto& d = digits[hit];
  const uint16_t chipID = d.getChipIndex();
  const uint16_t row = d.getRow();
  const uint16_t col = d.getColumn();

  if (labelsClusPtr) {
    int nlab = 0;
    fetchMCLabels(hit, labelsDigPtr, nlab);
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
    cluster.subDetID = geom->getSubDetID(chipID);
    cluster.layer = geom->getLayer(chipID);
    cluster.disk = geom->getDisk(chipID);
  }
  clusters.emplace_back(cluster);
}

//__________________________________________________
void Clusterer::ClustererThread::streamCluster(const BBox& bbox,
                                               const std::vector<std::pair<uint16_t, uint16_t>>& pixbuf,
                                               uint32_t totalCharge,
                                               bool doLabels, int nlab,
                                               uint16_t chipID, int subDetID, int layer, int disk)
{
  if (doLabels) {
    const auto cnt = static_cast<uint32_t>(clusters.size());
    for (int i = nlab; i--;) {
      labels.addElement(cnt, labelsBuff[i]); // accumulate in thread-local buffer
    }
  }

  const uint16_t rowSpanW = bbox.rowSpan();
  const uint16_t colSpanW = bbox.colSpan();

  // Encode the pixel pattern bitmap (rowSpan, colSpan, bytes...)
  std::array<unsigned char, o2::itsmft::ClusterPattern::MaxPatternBytes> patt{};
  for (const auto& [r, c] : pixbuf) {
    uint32_t ir = r - bbox.rowMin, ic = c - bbox.colMin;
    int nbit = ir * colSpanW + ic;
    patt[nbit >> 3] |= (0x1 << (7 - (nbit % 8)));
  }
  patterns.emplace_back(static_cast<unsigned char>(rowSpanW));
  patterns.emplace_back(static_cast<unsigned char>(colSpanW));
  int nBytes = (rowSpanW * colSpanW + 7) / 8;
  patterns.insert(patterns.end(), patt.begin(), patt.begin() + nBytes);

  Cluster cluster;
  cluster.chipID = chipID;
  cluster.row = bbox.rowMin;
  cluster.col = bbox.colMin;
  cluster.size = static_cast<uint16_t>(pixbuf.size());
  cluster.subDetID = static_cast<int16_t>(subDetID);
  cluster.layer = static_cast<int16_t>(layer);
  cluster.disk = static_cast<int16_t>(disk);
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

} // namespace o2::trk
