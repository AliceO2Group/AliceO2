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

/// \file VertexUtils.h
/// \brief Utility functions for vertex handling

#ifndef O2_ITS_TRACKING_VERTEXUTILS_H_
#define O2_ITS_TRACKING_VERTEXUTILS_H_

#include "DataFormatsITS/Vertex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ITStracking/Configuration.h"

#include <limits>
#include <unordered_map>
#include <utility>

namespace o2::its
{

/// Majority-vote MC label of a vertex: the most frequent (source, event) among its contributors, flagged fake when no label reaches more than half of them.
/// Templated on the container so that the bounded_vector (CPU traits) and std::vector (GPU traits, host side) callers share one implementation.
template <typename Container>
VertexLabel computeMainVertexLabel(const Container& elements)
{
  // we only care about the source&event of the tracks, not the trackId
  auto composeVtxLabel = [](const o2::MCCompLabel& lbl) -> o2::MCCompLabel {
    return {o2::MCCompLabel::maxTrackID(), lbl.getEventID(), lbl.getSourceID(), lbl.isFake()};
  };
  std::unordered_map<o2::MCCompLabel, size_t> frequency;
  for (const auto& element : elements) {
    ++frequency[composeVtxLabel(element)];
  }
  o2::MCCompLabel elem{};
  size_t maxCount = 0;
  for (const auto& [key, count] : frequency) {
    if (count > maxCount) {
      maxCount = count;
      elem = key;
    }
  }
  if (maxCount <= 1) { // need >50%
    elem.setFakeFlag();
  }
  return std::make_pair(elem, static_cast<float>(maxCount) / static_cast<float>(elements.size()));
}

inline Vertex makeDiamondVertex(const TrackingParameters& trkParam)
{
  Vertex diamond(trkParam.Diamond, trkParam.DiamondCov, 1, 1.f);
  diamond.setTimeStamp({0u, std::numeric_limits<TimeStampErrorType>::max()});
  return diamond;
}

} // namespace o2::its

#endif /* O2_ITS_TRACKING_VERTEXUTILS_H_ */
