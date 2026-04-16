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

#ifndef O2_ITS_TRACKING_LINE_VERTEXER_HELPERS_H_
#define O2_ITS_TRACKING_LINE_VERTEXER_HELPERS_H_

#include <memory>
#include <span>

#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/ClusterLines.h"

namespace o2::its::line_vertexer
{

struct Settings {
  float beamX = 0.f;
  float beamY = 0.f;
  float pairCut = 0.f;
  float pairCut2 = 0.f;
  float clusterCut = 0.f;
  float coarseZWindow = 0.f;
  float seedDedupZCut = 0.f;
  float refitDedupZCut = 0.f;
  float duplicateZCut = 0.f;
  float duplicateDistance2Cut = 0.f;
  float finalSelectionZCut = 0.f;
  float maxZ = 0.f;
  int seedMemberRadiusTime = 1;
  int seedMemberRadiusZ = 2;
  std::shared_ptr<BoundedMemoryResource> memoryPool;
};

bounded_vector<ClusterLines> buildClusters(std::span<const Line> lines, const Settings& settings);

} // namespace o2::its::line_vertexer

#endif
