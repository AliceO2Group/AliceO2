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

/// \file TopologyClassifier.cxx
/// \brief Implementation of the TopologyClassifier class.

#include "IOTOFReconstruction/TopologyClassifier.h"
#include "DataFormatsIOTOF/Cluster.h"

// Include for bitset
#include <bitset>

ClassImp(o2::iotof::TopologyClassifier);

using std::array;

namespace o2
{
namespace iotof
{

void TopologyClassifier::getTopology(uint16_t bitmask, uint16_t minRow, uint8_t spanRow, uint16_t minCol, uint8_t spanCol, uint8_t& topology)
{

  // 1. Guard against spans exceeding 8-bit representation for 
  // row, col span and 16-bit bitmasks
  if (spanRow > MaxRowSpan || spanCol > MaxColSpan || bitmask > MaxBitmask) {
    topology = Topologies::kHuge;
    return;
  }

  const uint32_t clsTopoKey = packKey(spanRow, spanCol, bitmask);
  // Print the 16 bits of the bitmask for debugging
  LOG(info) << "[TopologyClassifier::getTopology] Bitmask: " << std::bitset<16>(bitmask) << ", minRow: " << static_cast<int>(minRow) << ", spanRow: " << static_cast<int>(spanRow)
            << ", minCol: " << static_cast<int>(minCol) << ", spanCol: " << static_cast<int>(spanCol);
  LOG(info) << "[TopologyClassifier::getTopology] Packed key: " << clsTopoKey;

  // Check if the topology is already cached
  auto it = mTopologyCache.find(clsTopoKey);
  if (it != mTopologyCache.end()) {
    topology = it->second.mTopology;
    it->second.mFrequency++;
    LOG(info) << "[TopologyClassifier::getTopology] Found cached topology: " << static_cast<int>(topology);
    return;
  }

  // Classify the new topology and cache the result
  accountTopology(bitmask, minRow, spanRow, minCol, spanCol, topology);
}


TopologyInfo TopologyClassifier::getTopologyFeatures(uint32_t key)
{
  auto it = mTopologyCache.find(key);
  if (it != mTopologyCache.end()) {
    return it->second;
  } else {
    LOG(info) << "[TopologyClassifier::getTopologyFeatures] No cached features found for key: " << key;
    return TopologyInfo(); // Return default-constructed TopologyInfo if not found
  }
}

void TopologyClassifier::accountTopology(uint16_t bitmask, uint16_t minRow, uint8_t spanRow, uint16_t minCol, uint8_t spanCol, uint8_t& topology)
{
  LOG(info) << "[TopologyClassifier::accountTopology] Classifying topology for bitmask: " << std::bitset<16>(bitmask) << ", minRow: " << static_cast<int>(minRow) << ", spanRow: " << static_cast<int>(spanRow)
            << ", minCol: " << static_cast<int>(minCol) << ", spanCol: " << static_cast<int>(spanCol);

  // New cluster topology features
  TopologyInfo newTopo;
  newTopo.mFrequency = 1;
  newTopo.mPattern = bitmask;
  newTopo.mSizeX = spanRow;
  newTopo.mSizeZ = spanCol;
  float xCOG{0.f}, zCOG{0.f}, mXMean{0.f}, mZMean{0.f}, mXSigma2{0.f}, mZSigma2{0.f};
  computeCOG(bitmask, minRow, spanRow, minCol, spanCol, newTopo);

  const int maxRow = minRow + spanRow - 1;
  const int maxCol = minCol + spanCol - 1;

  const auto hasDigit = [bitmask, minRow, minCol, spanCol](int row, int col) -> bool {
    const int bitIndex = (row - minRow) * spanCol + (col - minCol);
    return (bitmask & (1U << bitIndex)) != 0;
  };

  // Basic shapes
  if (spanRow == 1 && spanCol == 1) {
    newTopo.mTopology = Topologies::kSingleDigit;
    mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
    return;
  }
  if (spanCol == 1) {
    newTopo.mTopology = Topologies::kLineOnRow;
    mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
    return;
  }
  if (spanRow == 1) {
    newTopo.mTopology = Topologies::kLineOnCol;
    mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
    return;
  }

  // Corner occupancy
  const bool hasTopLeft = hasDigit(minRow, minCol);
  const bool hasTopRight = hasDigit(minRow, maxCol);
  const bool hasBottomLeft = hasDigit(maxRow, minCol);
  const bool hasBottomRight = hasDigit(maxRow, maxCol);

  // Diagonal and square
  if (spanRow == spanCol) {

    if ((hasTopLeft && hasBottomRight && !hasTopRight && !hasBottomLeft) ||
        (!hasTopLeft && !hasBottomRight && hasTopRight && hasBottomLeft)) {
      newTopo.mTopology = Topologies::kDiagonal;
      mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
      return;
    }

    if (hasTopLeft && hasTopRight && hasBottomLeft && hasBottomRight) {
      newTopo.mTopology = Topologies::kSquare;
      mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
      return;
    }
  
    // Triangles (exactly one missing corner)
    const int nCorners = hasTopLeft + hasTopRight + hasBottomLeft + hasBottomRight;
    if (nCorners == 3) {
      const int missing = !hasTopLeft ? 0 : !hasTopRight ? 1 : !hasBottomLeft ? 2 : 3;

      switch (missing) {
        case 0: newTopo.mTopology = Topologies::kLowerTriangleLeft;  break;
        case 1: newTopo.mTopology = Topologies::kLowerTriangleRight; break;
        case 2: newTopo.mTopology = Topologies::kUpperTriangleLeft;  break;
        case 3: newTopo.mTopology = Topologies::kUpperTriangleRight; break;
      }
      mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
      return;
    }
  }

  // Snake: 3 x 2
  if (spanRow == 3 && spanCol == 2) {
    const bool hasMiddleMin = hasDigit(minRow + 1, minCol);
    const bool hasMiddleMax = hasDigit(minRow + 1, maxCol);

    if (hasMiddleMin && hasMiddleMax) {
      if (hasTopLeft && hasBottomRight) {
        newTopo.mTopology = Topologies::kSnake;
        mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
        return;
      }

      if (!hasTopLeft && !hasBottomRight) {
        newTopo.mTopology = Topologies::kSnakeRefl;
        mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
        return;
      }
    }
  }

  // Snake rotated by 90 degrees: 2 x 3
  if (spanRow == 2 && spanCol == 3) {
    const bool hasMiddleLeft = hasDigit(minRow, minCol + 1);
    const bool hasMiddleRight = hasDigit(maxRow, minCol + 1);

    if (hasMiddleLeft && hasMiddleRight) {
      if (hasTopLeft && hasBottomRight) {
        newTopo.mTopology = Topologies::kSnakeRot90;
        mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
        return;
      }

      if (!hasTopLeft && !hasBottomRight) {
        newTopo.mTopology = Topologies::kSnakeRot90Refl;
        mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
        return;
      }
    }
  }

  if (newTopo.mTopology == Topologies::kNTopologies) {
    newTopo.mTopology = Topologies::kOther;
    mTopologyCache[packKey(spanRow, spanCol, bitmask)] = newTopo;
    return;
  }
  // Insert in map
}


void TopologyClassifier::computeCOG(uint16_t bitmask, uint16_t minRow, uint8_t spanRow, uint16_t minCol, uint8_t spanCol, TopologyInfo& topoInfo)
{
  LOG(info) << "\n\nComputing COG";
  int xOffsetCOG = 0;
  int zOffsetCOG = 0;
  int firedPixels = 0;

  // Ensure nBits does not exceed the bitmask capacity (16 bits)
  const int nBits = std::min(static_cast<int>(spanRow * spanCol), 16);

  for (int iBit = 0; iBit < nBits; ++iBit) {
    // Check if the pixel bit is set
    if (bitmask & (1U << iBit)) {
      int iRow = iBit / spanCol;
      int iCol = iBit % spanCol;

      xOffsetCOG += minRow + iRow;
      zOffsetCOG += minCol + iCol;
      LOG(info) << "Fired pixel at (row, col): (" << (minRow + iRow) << ", " << (minCol + iCol) << ")";
      LOG(info) << "Current offsets: xOffsetCOG = " << xOffsetCOG << ", zOffsetCOG = " << zOffsetCOG;
      ++firedPixels;
    }
  }

  topoInfo.mOffsetXToCOG = static_cast<int>((static_cast<float>(xOffsetCOG) / firedPixels) - static_cast<float>(minRow));
  topoInfo.mOffsetZToCOG = static_cast<int>((static_cast<float>(zOffsetCOG) / firedPixels) - static_cast<float>(minCol));
  LOG(info) << "Computed COG offsets: (" << topoInfo.mOffsetXToCOG << ", " << topoInfo.mOffsetZToCOG << ")";
  topoInfo.mNPixels = firedPixels;

  LOG(info) << "COG: (" << topoInfo.mOffsetXToCOG << ", " << topoInfo.mOffsetZToCOG << "), Fired Pixels: " << firedPixels;

  // TO BE IMPLEMENTED
  topoInfo.mXMean = 0.f;
  topoInfo.mZMean = 0.f;
  topoInfo.mXSigma2 = 0.f;
  topoInfo.mZSigma2 = 0.f;

  // const auto& chipSpecs = ChipSpecificsParam::Instance();
  // if (useDf) {
  //   topoInfo.mXmean = dX;
  //   topoInfo.mZmean = dZ;
  // } else { // assign expected sigmas from the pixel X, Z sizes
  //   topoInfo.mXsigma2 = chipSpecs.PitchRow * chipSpecs.PitchRow / 12. / std::min(10, topoInfo.mSizeX);
  //   topoInfo.mZsigma2 = chipSpecs.PitchCol * chipSpecs.PitchCol / 12. / std::min(10, topoInfo.mSizeZ);
  // }

}


void TopologyClassifier::saveCacheToFile(const char* filename) {
  TFile file(filename, "RECREATE");
  // Write directly using TObject::Write syntax with explicit class name handling
  file.WriteObject(&mTopologyCache, "TF3ClusterTopologies");
  file.Close();
}


void TopologyClassifier::print() {
  LOG(info) << "Topology Cache Contents:";
  for (const auto& entry : mTopologyCache) {
    const uint32_t key = entry.first;
    const TopologyInfo& topoInfo = entry.second;

    uint8_t spanRow = (key >> 24) & 0xFF;
    uint8_t spanCol = (key >> 16) & 0xFF;
    uint16_t bitmask = key & 0xFFFF;

    LOG(info) << "Key: " << key
              << ", SpanRow: " << static_cast<int>(spanRow)
              << ", SpanCol: " << static_cast<int>(spanCol)
              << ", Bitmask: " << std::bitset<16>(bitmask)
              << ", Topology: " << static_cast<int>(topoInfo.mTopology)
              << ", COGx: " << topoInfo.mOffsetXToCOG
              << ", COGz: " << topoInfo.mOffsetZToCOG
              << ", NPixels: " << topoInfo.mNPixels
              << ", Frequency: " << topoInfo.mFrequency;
  }
}


} // namespace o2::iotof
} // namespace o2
