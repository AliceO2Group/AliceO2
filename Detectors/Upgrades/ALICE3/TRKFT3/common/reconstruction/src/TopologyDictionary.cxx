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

/// \file TopologyDictionary.cxx

#include "TRKReconstruction/TopologyDictionary.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "TRKBase/SegmentationChip.h"
#include <TFile.h>
#include <fairlogger/Logger.h>
#include <fmt/format.h>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace o2::trkft3
{

using o2::trk::SegmentationChip;

template <int DetID>
int TopologyDictionary<DetID>::layerOf(const ClusterType& cluster) noexcept
{
  if (cluster.layer < 0) {
    return -1;
  }
  if constexpr (DetID == o2::detectors::DetID::TRK) {
    if (cluster.subDetID == 0) {
      return cluster.layer;
    }
    if (cluster.subDetID == 1) {
      return o2::trk::constants::VD::petal::nLayers + cluster.layer;
    }
    return -1;
  } else {
    return cluster.layer;
  }
}

template <int DetID>
math_utils::Point3D<float> TopologyDictionary<DetID>::localToFlat(const ClusterType& cluster, const math_utils::Point3D<float>& local) noexcept
{
  if (cluster.subDetID != 0) {
    return local;
  }
  const auto flat = SegmentationChip::curvedToFlat(cluster.layer, local.X(), local.Y());
  return {flat.X(), flat.Y(), local.Z()};
}

template <int DetID>
math_utils::Point3D<float> TopologyDictionary<DetID>::centreOfGravity(const ClusterType& cluster, const unsigned char* pattern) noexcept
{
  float cogRow = 0.f, cogCol = 0.f;
  itsmft::ClusterPattern::getCOG(pattern[0], pattern[1], pattern + 2, cogRow, cogCol);
  float x = 0.f, z = 0.f;
  SegmentationChip::detectorToLocalUnchecked(cluster.row, cluster.col, x, z, cluster.subDetID, cluster.layer, cluster.layer);
  const bool vd = cluster.subDetID == 0;
  x -= cogRow * (vd ? SegmentationChip::PitchRowVD : SegmentationChip::PitchRowMLOT);
  z += cogCol * (vd ? SegmentationChip::PitchColVD : SegmentationChip::PitchColMLOT);
  return {x, 0.f, z};
}

template <int DetID>
int TopologyDictionary<DetID>::findID(int layer, const unsigned char* pattern) const
{
  const auto& data = mLayers.at(layer);
  const int rowSpan = pattern[0], colSpan = pattern[1];
  // The hash leaves out a large pattern's last bytes: compare them all.
  const auto nBytes = static_cast<std::size_t>(2 + ((rowSpan * colSpan + 7) / 8));
  const auto [first, last] = data.hashMap.equal_range(itsmft::ClusterTopology::getCompleteHash(rowSpan, colSpan, pattern + 2));
  for (auto it = first; it != last; ++it) {
    const auto& entryPattern = data.entries[it->second].pattern;
    if (entryPattern.size() == nBytes && std::memcmp(entryPattern.data(), pattern, nBytes) == 0) {
      return it->second;
    }
  }
  return static_cast<int>(data.entries.size()) - NGroups + groupOf(rowSpan, colSpan);
}

template <int DetID>
math_utils::Point3D<float> TopologyDictionary<DetID>::getFlatCoordinates(const ClusterType& cluster, const unsigned char* pattern, int id) const
{
  const int layer = layerOf(cluster);
  const auto& data = mLayers.at(layer);
  const auto& entry = data.entries.at(id);
  math_utils::Point3D<float> flat;
  if (entry.isGroup) {
    flat = centreOfGravity(cluster, pattern);
  } else {
    float x = 0.f, z = 0.f;
    SegmentationChip::detectorToLocalUnchecked(cluster.row, cluster.col, x, z, cluster.subDetID, cluster.layer, cluster.layer);
    const bool vd = cluster.subDetID == 0;
    flat.SetX(x + entry.xCOG * (vd ? SegmentationChip::PitchRowVD : SegmentationChip::PitchRowMLOT));
    flat.SetZ(z + entry.zCOG * (vd ? SegmentationChip::PitchColVD : SegmentationChip::PitchColMLOT));
  }
  if (mApplyOffsets) {
    flat.SetX(flat.X() + entry.xOffset);
    flat.SetZ(flat.Z() + entry.zOffset);
  }
  flat.SetY(data.depth);
  return flat;
}

template <int DetID>
math_utils::Point3D<float> TopologyDictionary<DetID>::getClusterCoordinates(const ClusterType& cluster, const unsigned char* pattern, int id) const
{
  auto local = getFlatCoordinates(cluster, pattern, id);
  if (cluster.subDetID == 0) {
    const auto curved = SegmentationChip::flatToCurved(cluster.layer, local.X(), local.Y());
    local.SetX(curved.X());
    local.SetY(curved.Y());
  }
  return local;
}

template <int DetID>
void TopologyDictionary<DetID>::print() const
{
  LOGP(info, "{} TopologyDictionary: {} layers, offsets {}", o2::detectors::DetID(DetID).getName(), mLayers.size(), mApplyOffsets ? "applied" : "not applied");
  for (std::size_t layer = 0; layer < mLayers.size(); ++layer) {
    LOGP(info, "  layer {:2}: depth {:6.2f} um, {} common patterns", layer, 1e4 * mLayers[layer].depth, mLayers[layer].entries.size() - NGroups);
  }
}

template <int DetID>
TopologyDictionary<DetID>* TopologyDictionary<DetID>::loadFrom(const std::string& fileName, const std::string& objName)
{
  LOGP(info, "Loading {} TopologyDictionary from {} with name {}", o2::detectors::DetID(DetID).getName(), fileName, objName);
  std::unique_ptr<TFile> file{TFile::Open(fileName.c_str(), "READ")};
  if (!file || file->IsZombie()) {
    throw std::runtime_error(fmt::format("Failed to open {}", fileName));
  }
  auto* dictionary = file->Get<TopologyDictionary<DetID>>(objName.c_str());
  if (dictionary == nullptr) {
    throw std::runtime_error(fmt::format("Failed to load {} from {}", objName, fileName));
  }
  return dictionary;
}

template class TopologyDictionary<o2::detectors::DetID::TRK>;
template class TopologyDictionary<o2::detectors::DetID::FT3>;

} // namespace o2::trkft3
