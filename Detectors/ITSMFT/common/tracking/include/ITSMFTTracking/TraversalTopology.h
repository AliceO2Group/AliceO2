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

#ifndef ALICEO2_ITSMFT_TRACKING_TRAVERSALTOPOLOGY_H_
#define ALICEO2_ITSMFT_TRACKING_TRAVERSALTOPOLOGY_H_

#include <cstdint>
#include <type_traits>

#ifndef GPUCA_GPUCODE
#include <optional>
#include <vector>
#include "ITSMFTTracking/DetectorLayout.h"
#endif

#include "ITSMFTTracking/IdTypes.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/LayerMask.h"

namespace o2::itsmft
{
struct IterationParameters;
}

namespace o2::itsmft::tracking
{

struct Edge {
  LayerId from{};
  LayerId to{};
};

struct CellPath {
  EdgeId first{};
  EdgeId second{};
};

struct TopologyRange {
  uint32_t firstEntry{0};
  uint32_t entries{0};

  uint32_t getFirstEntry() const noexcept { return firstEntry; }
  uint32_t getEntries() const noexcept { return entries; }
  uint32_t getEntriesBound() const noexcept { return firstEntry + entries; }
};

struct TraversalTopologyView {
  SurfaceCatalogView catalog{};
  uint32_t nLayers{0};
  const LayerId* activeSurfaceList{nullptr};
  uint32_t nActiveSurfaces{0};
  LayerMask activeLayers{};
  const Edge* edges{nullptr};
  uint32_t nEdges{0};
  const CellPath* paths{nullptr};
  uint32_t nPaths{0};
  const uint32_t* pathsByFirstEdgeOffsets{nullptr};
  const CellPathId* pathsByFirstEdge{nullptr};
  const CellPathId* scheduledPaths{nullptr};
  uint32_t nScheduledPaths{0};
  const CellPathId* roadStartPaths{nullptr};
  uint32_t nRoadStartPaths{0};
  const uint32_t* roadStartComponentOffsets{nullptr};
  uint32_t nRoadStartComponentOffsets{0};
  LayerMask seedingLayers{};

  const SurfaceDescriptor& getSurface(LayerId id) const { return catalog.getSurface(id); }
  SurfaceCatalogView getSurfaceCatalogView() const noexcept { return catalog; }
  const Edge& getEdge(EdgeId id) const { return edges[id.value()]; }
  const CellPath& getPath(CellPathId id) const { return paths[id.value()]; }
  TopologyRange getPathsStartingWithEdge(EdgeId edge) const
  {
    const auto index = edge.value();
    return {pathsByFirstEdgeOffsets[index], pathsByFirstEdgeOffsets[index + 1] - pathsByFirstEdgeOffsets[index]};
  }
};

#ifndef GPUCA_GPUCODE
struct TraversalTopology {
  uint16_t nLayers{0};
  std::vector<LayerId> activeSurfaceList;
  LayerMask activeLayers{};
  LayerMask seedingLayers{};
  std::vector<Edge> edges;
  std::vector<CellPath> paths;
  std::vector<uint32_t> pathsByFirstEdgeOffsets;
  std::vector<CellPathId> pathsByFirstEdge;
  std::vector<CellPathId> scheduledPaths;
  std::vector<CellPathId> roadStartPaths;
  std::vector<uint32_t> roadStartComponentOffsets;

  TraversalTopologyView getView(SurfaceCatalogView catalog) const noexcept
  {
    return {catalog,
            nLayers,
            activeSurfaceList.data(), static_cast<uint32_t>(activeSurfaceList.size()),
            activeLayers,
            edges.data(), static_cast<uint32_t>(edges.size()),
            paths.data(), static_cast<uint32_t>(paths.size()),
            pathsByFirstEdgeOffsets.data(), pathsByFirstEdge.data(),
            scheduledPaths.data(), static_cast<uint32_t>(scheduledPaths.size()),
            roadStartPaths.data(), static_cast<uint32_t>(roadStartPaths.size()),
            roadStartComponentOffsets.data(), static_cast<uint32_t>(roadStartComponentOffsets.size()),
            seedingLayers};
  }
};

enum class TraversalTopologyError : uint8_t {
  None,
  InvalidLayout,
  LayerCountMismatch,
  NegativeMaxHoles,
  NoActiveSurfaces,
  TooManyEdges,
  TooManyPaths
};

struct TraversalTopologyBuildResult {
  std::optional<TraversalTopology> topology;
  TraversalTopologyError error{TraversalTopologyError::None};

  bool ok() const noexcept { return topology.has_value(); }
};

// Derive one iteration's topology from the invariant detector layout and the
// Tracker-owned iteration parameters.
TraversalTopologyBuildResult deriveTraversalTopology(const DetectorLayout& layout,
                                                     const o2::itsmft::IterationParameters& parameters);

#endif // GPUCA_GPUCODE

static_assert(sizeof(CellPath) == 4);
static_assert(std::is_standard_layout_v<TraversalTopologyView> && std::is_trivially_copyable_v<TraversalTopologyView>);

} // namespace o2::itsmft::tracking

#endif
