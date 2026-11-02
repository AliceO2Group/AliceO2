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

/// \file TopologyDictionary.h
/// \brief Positions and errors of TRK and FT3 clusters per layer and pixel pattern
///
/// As the ITS's topology dictionary, measured from MC, but per layer: the
/// layer's depth (the local y positions are taken at) and, per common pattern
/// and per group of rare ones, the width (normalised median absolute
/// deviation) and median of the residuals hit - centre of gravity across
/// (local x) and along (local z). Every pattern finds an entry: its own, else
/// its group's. On the VD the dictionary works in the flat frame of
/// SegmentationChip::curvedToFlat; getClusterCoordinates returns O2's curved
/// local coordinates.

#ifndef ALICEO2_TRKFT3_TOPOLOGYDICTIONARY_H
#define ALICEO2_TRKFT3_TOPOLOGYDICTIONARY_H

#include "DataFormatsTRKFT3/Cluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "MathUtils/Cartesian.h"
#include <Rtypes.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace o2::trkft3
{

template <int DetID>
class BuildTopologyDictionary;

/// A common pattern, or the group of the rare patterns with the same span
/// classes (spans 1 to 5, 6 or more) on both axes.
struct DictionaryEntry {
  std::vector<unsigned char> pattern; ///< row span, column span, pixel bits as the clusteriser streams them; empty for a group
  float xCOG = 0.f;                   ///< centre of gravity from the reference pixel across [pitches]
  float zCOG = 0.f;                   ///< along [pitches]
  float xWidth = 0.f;                 ///< the residuals' width across [cm]
  float zWidth = 0.f;                 ///< along [cm]
  float xOffset = 0.f;                ///< the residuals' median across [cm]
  float zOffset = 0.f;                ///< along [cm]
  bool isGroup = false;

  ClassDefNV(DictionaryEntry, 1);
};

struct LayerDictionary {
  float depth = 0.f;                                   ///< the local y positions are taken at [cm]
  std::vector<DictionaryEntry> entries;                ///< the common patterns, most frequent first, then the groups
  std::unordered_multimap<unsigned long, int> hashMap; ///< itsmft::ClusterTopology hash -> common pattern's entry

  ClassDefNV(LayerDictionary, 1);
};

template <int DetID>
class TopologyDictionary
{
  static_assert(DetID == o2::detectors::DetID::TRK || DetID == o2::detectors::DetID::FT3, "only TRK and FT3 dictionaries are supported");

 public:
  using ClusterType = Cluster<DetID>;

  static constexpr int NSpanClasses = 6;                      ///< spans 1 to 5, 6 or more
  static constexpr int NGroups = NSpanClasses * NSpanClasses; ///< one group per pair of span classes, row class major

  static constexpr int spanClass(int span) noexcept { return std::min(span, NSpanClasses) - 1; }
  static constexpr int groupOf(int rowSpan, int colSpan) noexcept { return spanClass(rowSpan) * NSpanClasses + spanClass(colSpan); }

  /// A cluster's layer: TRK's VD layers, then ML/OT (GeometryTGeo::getLayerTRK); FT3's discs. -1 for none (a VD disc).
  static int layerOf(const ClusterType& cluster) noexcept;
  /// A point from O2's local frame to the layer's flat frame (the VD's unrolled; elsewhere the same).
  static math_utils::Point3D<float> localToFlat(const ClusterType& cluster, const math_utils::Point3D<float>& local) noexcept;
  /// The cluster's centre of gravity in the flat frame at y = 0. \a pattern: row span, column span, pixel bits.
  static math_utils::Point3D<float> centreOfGravity(const ClusterType& cluster, const unsigned char* pattern) noexcept;

  [[nodiscard]] int getNumberOfLayers() const noexcept { return static_cast<int>(mLayers.size()); }
  [[nodiscard]] const LayerDictionary& getLayer(int layer) const { return mLayers.at(layer); }
  [[nodiscard]] const DictionaryEntry& getEntry(int layer, int id) const { return mLayers.at(layer).entries.at(id); }

  /// The pattern's entry on the layer: its own if it is common, else its group's.
  [[nodiscard]] int findID(int layer, const unsigned char* pattern) const;

  /// The position in the flat frame: the entry's centre of gravity, plus its offsets if applied, at the layer's depth.
  [[nodiscard]] math_utils::Point3D<float> getFlatCoordinates(const ClusterType& cluster, const unsigned char* pattern, int id) const;
  /// The same in O2's local frame: curved on the VD.
  [[nodiscard]] math_utils::Point3D<float> getClusterCoordinates(const ClusterType& cluster, const unsigned char* pattern, int id) const;
  [[nodiscard]] math_utils::Point3D<float> getClusterCoordinates(const ClusterType& cluster, const unsigned char* pattern) const
  {
    return getClusterCoordinates(cluster, pattern, findID(layerOf(cluster), pattern));
  }

  [[nodiscard]] bool getApplyOffsets() const noexcept { return mApplyOffsets; }
  void setApplyOffsets(bool apply) noexcept { mApplyOffsets = apply; }

  void print() const;

  static TopologyDictionary* loadFrom(const std::string& fileName, const std::string& objName = "ccdb_object");

 private:
  friend class BuildTopologyDictionary<DetID>;

  std::vector<LayerDictionary> mLayers;
  bool mApplyOffsets = false;

  ClassDefNV(TopologyDictionary, 1);
};

using TRKTopologyDictionary = TopologyDictionary<o2::detectors::DetID::TRK>;
using FT3TopologyDictionary = TopologyDictionary<o2::detectors::DetID::FT3>;

} // namespace o2::trkft3

#endif
