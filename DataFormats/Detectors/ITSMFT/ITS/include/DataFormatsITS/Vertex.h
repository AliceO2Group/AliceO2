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

#ifndef O2_TRACKINGITS_VERTEX_H_
#define O2_TRACKINGITS_VERTEX_H_

#include "GPUCommonDef.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <type_traits>
#include <unordered_map>
#include <utility>
#endif
#include "ReconstructionDataFormats/Vertex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsITS/TimeEstBC.h"

namespace o2::its
{
// NOTE: this uses the internal asymmetrical time reprenstation!
using Vertex = o2::dataformats::Vertex<o2::its::TimeEstBC>;
using VertexLabel = std::pair<o2::MCCompLabel, float>;

#ifndef GPUCA_GPUCODE_DEVICE
/// Majority-vote MC label of a vertex: the most frequent (source, event) among its
/// contributors, flagged fake when no label reaches more than half of them. Templated
/// on the container so both bounded_vector and std::vector callers share one copy
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
#endif
} // namespace o2::its

#ifndef GPUCA_GPUCODE_DEVICE
/// Defining ITS Vertex explicitly as messageable
namespace o2::framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::dataformats::Vertex<o2::its::TimeEstBC>> : std::true_type {
};
} // namespace o2::framework
#endif

#endif
