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

/// \file TopologyClassifier.h
/// \brief Definition of the TopologyClassifier class.
///
/// Short TopologyClassifier descritpion
///
/// This class is for the association of the cluster 
/// topology with the corresponding entry in the dictionary
///

#ifndef ALICEO2_IOTOF_TOPOLOGYCLASSIFIER_H
#define ALICEO2_IOTOF_TOPOLOGYCLASSIFIER_H

#include <array>
#include <cstdint>
#include <unordered_map>

#include <TFile.h>

// TO BE REMOVED BEFORE PUSH
#include "Framework/Logger.h"

namespace o2
{
namespace iotof
{

enum Topologies : uint8_t {
  kSingleDigit,
  kLineOnRow,
  kLineOnCol,
  kSquare,
  kDiagonal,
  kLowerTriangleLeft,
  kLowerTriangleRight,
  kUpperTriangleLeft,
  kUpperTriangleRight,
  kSnake,
  kSnakeRefl,
  kSnakeRot90,
  kSnakeRot90Refl,
  kHuge,
  kOther,
  kNTopologies
};

struct TopologyInfo {
  int mSizeX = 0;
  int mSizeZ = 0;
  int mOffsetXToCOG = 0;
  int mOffsetZToCOG = 0;
  float mXMean = 0.f;
  float mZMean = 0.f;
  float mXSigma2 = 0.f;
  float mZSigma2 = 0.f;
  int mNPixels = 0;
  int mFrequency = 0;
  Topologies mTopology = Topologies::kNTopologies;
  uint16_t mPattern; ///< Bitmask of fired pixels
};

class TopologyClassifier {
 public:
  // Define limits for domain validation
  static constexpr uint8_t MaxRowSpan = 255;
  static constexpr uint8_t MaxColSpan = 255;
  static constexpr uint16_t MaxBitmask = 65535;

  TopologyClassifier() = default;
  TopologyClassifier(std::unordered_map<uint32_t, TopologyInfo> map) : mTopologyCache(std::move(map)) {}

  const std::unordered_map<uint32_t, TopologyInfo>& getTopologyMap() const { return mTopologyCache; };
  void getTopology(uint16_t bitmask, uint16_t minRow, uint8_t spanRow, uint16_t minCol, uint8_t spanCol, uint8_t& topology);
  TopologyInfo getTopologyFeatures(uint32_t key);
  void accountTopology(uint16_t bitmask, uint16_t minRow, uint8_t spanRow, uint16_t minCol, uint8_t spanCol, uint8_t& topology);
  void computeCOG(uint16_t bitmask, uint16_t minRow, uint8_t spanRow, uint16_t minCol, uint8_t spanCol, TopologyInfo& topoInfo);

  void saveCacheToFile(const char* filename);
  void print();

 private:
  /// Packs: [ spanRow (8b) ][ spanCol (8b) ][ bitmask (16b) ] -> 32 bits total
  [[nodiscard]] static constexpr uint32_t packKey(uint8_t spanRow, uint8_t spanCol, uint16_t bitmask) noexcept {
    return (static_cast<uint32_t>(spanRow) << 24) |
           (static_cast<uint32_t>(spanCol) << 16) |
            static_cast<uint32_t>(bitmask);
  }

  std::unordered_map<uint32_t, TopologyInfo> mTopologyCache;
};

} // namespace iotof
} // namespace o2

#endif // ALICEO2_IOTOF_TOPOLOGYCLASSIFIER_H
