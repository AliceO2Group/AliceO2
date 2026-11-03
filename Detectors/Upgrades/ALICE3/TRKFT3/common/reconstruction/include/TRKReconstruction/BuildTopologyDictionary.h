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

/// \file BuildTopologyDictionary.h
/// \brief Builds the TRK and FT3 topology dictionaries from MC clusters and their particles' crossings
///
/// Each cluster is accounted with its particle's crossing of the chip: the
/// ends of its path through the silicon. Per layer, build first scans the
/// depth: the local y, on a grid of depthStep, at which the residuals'
/// widths across and along, added in quadrature, are smallest. It then cuts
/// every crossing at that depth: a pattern with at least minCount residuals
/// gets its own entry; a group takes the widths of its rare patterns, else of
/// the rare patterns with the same span class on the axis, else of the layer.

#ifndef ALICEO2_TRKFT3_BUILDTOPOLOGYDICTIONARY_H
#define ALICEO2_TRKFT3_BUILDTOPOLOGYDICTIONARY_H

#include "TRKReconstruction/TopologyDictionary.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace o2::trkft3
{

template <int DetID>
class BuildTopologyDictionary
{
 public:
  using Dictionary = TopologyDictionary<DetID>;
  using ClusterType = typename Dictionary::ClusterType;

  struct Settings {
    uint32_t minCount = 137;   ///< residuals for an entry of its own: the width to 10 %
    float depthStep = 0.5e-4f; ///< [cm]
    int nThreads = 1;          ///< for the depth scan
  };

  BuildTopologyDictionary(int nLayers, const Settings& settings);

  /// A cluster and its particle's crossing of the chip, in O2's local frame [cm].
  void accountCluster(const ClusterType& cluster, const unsigned char* pattern, const math_utils::Point3D<float>& start, const math_utils::Point3D<float>& end);

  /// Builds every layer; throws std::runtime_error for a layer with too few residuals.
  void build();

  void setApplyOffsets(bool apply) noexcept { mDictionary.setApplyOffsets(apply); }
  [[nodiscard]] const Dictionary& getDictionary() const noexcept { return mDictionary; }
  /// Writes the dictionary as "ccdb_object".
  void saveDictionaryRoot(const std::string& fileName) const;

 private:
  /// A cluster's crossing relative to its centre of gravity in the layer's flat frame [cm].
  struct Crossing {
    uint32_t pattern = 0; ///< index among the layer's patterns
    float x0 = 0.f;       ///< start - centre of gravity, across
    float y0 = 0.f;       ///< start, depth
    float z0 = 0.f;       ///< start - centre of gravity, along
    float dx = 0.f;       ///< end - start
    float dy = 0.f;
    float dz = 0.f;
  };
  struct LayerInput {
    std::unordered_map<std::string, uint32_t> patternIndex; ///< the pattern's bytes -> index
    std::vector<std::string> patterns;
    std::vector<Crossing> crossings;
  };

  [[nodiscard]] float measureDepth(int layer, const LayerInput& input) const;
  void makeEntries(int layer, const LayerInput& input, LayerDictionary& data) const;

  Settings mSettings;             //!
  std::vector<LayerInput> mInput; //!
  Dictionary mDictionary;         //!
};

using TRKBuildTopologyDictionary = BuildTopologyDictionary<o2::detectors::DetID::TRK>;
using FT3BuildTopologyDictionary = BuildTopologyDictionary<o2::detectors::DetID::FT3>;

} // namespace o2::trkft3

#endif
