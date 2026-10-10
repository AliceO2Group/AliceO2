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

#include "Framework/Logger.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

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

/// Good lines a further vertex needs to not count as debris in a ROF that already has one: goodSig * sqrt(ROF load), clamped to
/// [constants::VtxMinGoodThreshold, suppressLowMultDebris] unless the debris cut is off (UPC pass).
inline float getDebrisThreshold(const float goodSig, const double rofLoad, const int suppressLowMultDebris)
{
  const float threshold = goodSig * std::sqrt(static_cast<float>(std::max(rofLoad, 1.)));
  if (suppressLowMultDebris < constants::VtxMinGoodThreshold) {
    return threshold;
  }
  return std::clamp(threshold, constants::VtxMinGoodThreshold, static_cast<float>(suppressLowMultDebris));
}

/// Caps ROFs whose vertex count is an outlier of the TF
template <typename VtxVec, typename LabVec>
int pruneOverpopulatedRofs(std::vector<VtxVec>& rofVertices, std::vector<LabVec>& rofLabels, const float nSigma, const int minContributors, const float trimFraction)
{
  const int nRofs = static_cast<int>(rofVertices.size());
  if (nSigma <= 0.f || nRofs == 0) {
    return 0;
  }
  std::vector<int> counts(nRofs);
  for (int r = 0; r < nRofs; ++r) {
    counts[r] = static_cast<int>(rofVertices[r].size());
  }
  std::vector<int> sorted(counts);
  std::sort(sorted.begin(), sorted.end());
  const int nUsed = std::max(1, nRofs - std::max(1, static_cast<int>(trimFraction * nRofs)));
  double sum = 0.;
  for (int i = 0; i < nUsed; ++i) {
    sum += sorted[i];
  }
  const double mean = sum / nUsed;
  const double threshold = mean + nSigma * std::sqrt(mean + 1.);
  int removed = 0;
  for (int r = 0; r < nRofs; ++r) {
    if (counts[r] <= threshold) {
      continue;
    }
    auto& vtx = rofVertices[r];
    const bool withLabels = static_cast<int>(rofLabels.size()) == nRofs && rofLabels[r].size() == vtx.size();
    size_t out = 1; // the largest vertex always stays
    for (size_t i = 1; i < vtx.size(); ++i) {
      if (vtx[i].getNContributors() >= minContributors) {
        vtx[out] = vtx[i];
        if (withLabels) {
          rofLabels[r][out] = rofLabels[r][i];
        }
        ++out;
      }
    }
    LOGP(info, "Seeding vertexer: overpopulated ROF {} pruned {} -> {} vertices (threshold {:.1f}, mean {:.2f} per ROF)", r, vtx.size(), out, threshold, mean);
    removed += static_cast<int>(vtx.size() - out);
    vtx.erase(vtx.begin() + out, vtx.end());
    if (withLabels) {
      rofLabels[r].erase(rofLabels[r].begin() + out, rofLabels[r].end());
    }
  }
  return removed;
}

} // namespace o2::its

#endif /* O2_ITS_TRACKING_VERTEXUTILS_H_ */
