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

/// \file BuildTopologyDictionary.cxx

#include "TRKReconstruction/BuildTopologyDictionary.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include <TFile.h>
#include <fairlogger/Logger.h>
#include <fmt/format.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace o2::trkft3
{

namespace
{

/// The median of \a values, which it reorders.
float medianOf(std::span<float> values)
{
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  if (values.size() % 2 == 1) {
    return *middle;
  }
  return 0.5f * (*std::max_element(values.begin(), middle) + *middle);
}

/// The median of \a values and their normalised median absolute deviation
/// (a Gaussian's sigma); \a deviations a buffer at least their size.
std::pair<float, float> medianAndWidth(std::span<float> values, std::span<float> deviations)
{
  const float median = medianOf(values);
  for (std::size_t i = 0; i < values.size(); ++i) {
    deviations[i] = std::abs(values[i] - median);
  }
  return {median, 1.4826f * medianOf(deviations.first(values.size()))};
}

std::pair<float, float> medianAndWidth(std::vector<float>& values)
{
  std::vector<float> deviations(values.size());
  return medianAndWidth(values, deviations);
}

} // namespace

template <int DetID>
BuildTopologyDictionary<DetID>::BuildTopologyDictionary(int nLayers, const Settings& settings) : mSettings(settings), mInput(nLayers)
{
  mDictionary.mLayers.resize(nLayers);
}

template <int DetID>
void BuildTopologyDictionary<DetID>::accountCluster(const ClusterType& cluster, const unsigned char* pattern, const math_utils::Point3D<float>& start,
                                                    const math_utils::Point3D<float>& end)
{
  const int layer = Dictionary::layerOf(cluster);
  if (layer < 0 || layer >= static_cast<int>(mInput.size())) {
    throw std::runtime_error(fmt::format("{}: on no layer of the dictionary's {}", cluster.asString(), mInput.size()));
  }
  auto& input = mInput[layer];
  const auto nBytes = static_cast<std::size_t>(2 + ((pattern[0] * pattern[1] + 7) / 8));
  std::string bytes(reinterpret_cast<const char*>(pattern), nBytes);
  const auto [index, inserted] = input.patternIndex.try_emplace(bytes, static_cast<uint32_t>(input.patterns.size()));
  if (inserted) {
    input.patterns.push_back(std::move(bytes));
  }
  const auto cog = Dictionary::centreOfGravity(cluster, pattern);
  const auto a = Dictionary::localToFlat(cluster, start);
  const auto b = Dictionary::localToFlat(cluster, end);
  input.crossings.push_back({.pattern = index->second,
                             .x0 = a.X() - cog.X(),
                             .y0 = a.Y(),
                             .z0 = a.Z() - cog.Z(),
                             .dx = b.X() - a.X(),
                             .dy = b.Y() - a.Y(),
                             .dz = b.Z() - a.Z()});
}

template <int DetID>
float BuildTopologyDictionary<DetID>::measureDepth(int layer, const LayerInput& input) const
{
  const auto& crossings = input.crossings;
  float low = std::numeric_limits<float>::max(), high = std::numeric_limits<float>::lowest();
  for (const auto& crossing : crossings) {
    low = std::min({low, crossing.y0, crossing.y0 + crossing.dy});
    high = std::max({high, crossing.y0, crossing.y0 + crossing.dy});
  }
  const float step = mSettings.depthStep;
  const int nSteps = static_cast<int>(std::floor((high - low) / step));
  if (nSteps < 1) {
    throw std::runtime_error(fmt::format("layer {}: its hits span {} um of silicon, less than the depth step", layer, 1e4 * (high - low)));
  }
  // The grid's points sit half a step inside the cells; the crossings that reach all of them are scanned.
  const auto depthAt = [low, step](int k) { return low + ((static_cast<float>(k) + 0.5f) * step); };
  const float first = depthAt(0), last = depthAt(nSteps - 1);
  std::vector<uint32_t> scanned;
  for (std::size_t i = 0; i < crossings.size(); ++i) {
    const auto& crossing = crossings[i];
    if (crossing.dy != 0.f && std::min(crossing.y0, crossing.y0 + crossing.dy) <= first && std::max(crossing.y0, crossing.y0 + crossing.dy) >= last) {
      scanned.push_back(static_cast<uint32_t>(i));
    }
  }
  if (scanned.size() < mSettings.minCount) {
    throw std::runtime_error(fmt::format("layer {}: {} crossings through the silicon, fewer than {}", layer, scanned.size(), mSettings.minCount));
  }

  std::vector<float> objective(nSteps);
  const auto scan = [&](int thread, int nThreads) {
    std::vector<float> residualsX(scanned.size()), residualsZ(scanned.size()), deviations(scanned.size());
    for (int k = thread; k < nSteps; k += nThreads) {
      const float depth = depthAt(k);
      for (std::size_t i = 0; i < scanned.size(); ++i) {
        const auto& crossing = crossings[scanned[i]];
        const float t = (depth - crossing.y0) / crossing.dy;
        residualsX[i] = crossing.x0 + (t * crossing.dx);
        residualsZ[i] = crossing.z0 + (t * crossing.dz);
      }
      objective[k] = std::hypot(medianAndWidth(residualsX, deviations).second, medianAndWidth(residualsZ, deviations).second);
    }
  };
  const int nThreads = std::clamp(mSettings.nThreads, 1, nSteps);
  std::vector<std::thread> threads;
  for (int thread = 1; thread < nThreads; ++thread) {
    threads.emplace_back(scan, thread, nThreads);
  }
  scan(0, nThreads);
  for (auto& thread : threads) {
    thread.join();
  }
  return depthAt(static_cast<int>(std::ranges::min_element(objective) - objective.begin()));
}

template <int DetID>
void BuildTopologyDictionary<DetID>::makeEntries(int layer, const LayerInput& input, LayerDictionary& data) const
{
  constexpr int NSpanClasses = Dictionary::NSpanClasses;
  const auto nPatterns = input.patterns.size();
  const auto spansOf = [&input](uint32_t pattern) { return std::pair<int, int>{static_cast<unsigned char>(input.patterns[pattern][0]), static_cast<unsigned char>(input.patterns[pattern][1])}; };

  // The residuals at the depth, by pattern.
  std::vector<std::vector<float>> residualsX(nPatterns), residualsZ(nPatterns);
  std::vector<float> layerX, layerZ;
  for (const auto& crossing : input.crossings) {
    const float t = crossing.dy != 0.f ? (data.depth - crossing.y0) / crossing.dy : -1.f;
    if (!(t >= 0.f && t <= 1.f)) {
      continue;
    }
    const float x = crossing.x0 + (t * crossing.dx), z = crossing.z0 + (t * crossing.dz);
    residualsX[crossing.pattern].push_back(x);
    residualsZ[crossing.pattern].push_back(z);
    layerX.push_back(x);
    layerZ.push_back(z);
  }
  if (layerX.size() < mSettings.minCount) {
    throw std::runtime_error(fmt::format("layer {}: {} residuals at its depth, fewer than {}", layer, layerX.size(), mSettings.minCount));
  }

  // The common patterns, most frequent first (equal counts by their bytes).
  std::vector<uint32_t> common;
  for (uint32_t pattern = 0; pattern < nPatterns; ++pattern) {
    if (residualsX[pattern].size() >= mSettings.minCount) {
      common.push_back(pattern);
    }
  }
  std::ranges::sort(common, [&](uint32_t a, uint32_t b) {
    return residualsX[a].size() != residualsX[b].size() ? residualsX[a].size() > residualsX[b].size() : input.patterns[a] < input.patterns[b];
  });
  std::vector<bool> isCommon(nPatterns, false);
  for (const auto pattern : common) {
    isCommon[pattern] = true;
    const auto* bytes = reinterpret_cast<const unsigned char*>(input.patterns[pattern].data());
    const auto [rowSpan, colSpan] = spansOf(pattern);
    DictionaryEntry entry;
    entry.pattern.assign(bytes, bytes + input.patterns[pattern].size());
    float cogRow = 0.f, cogCol = 0.f;
    itsmft::ClusterPattern::getCOG(rowSpan, colSpan, bytes + 2, cogRow, cogCol);
    entry.xCOG = -cogRow;
    entry.zCOG = cogCol;
    std::tie(entry.xOffset, entry.xWidth) = medianAndWidth(residualsX[pattern]);
    std::tie(entry.zOffset, entry.zWidth) = medianAndWidth(residualsZ[pattern]);
    data.hashMap.emplace(itsmft::ClusterTopology::getCompleteHash(rowSpan, colSpan, bytes + 2), static_cast<int>(data.entries.size()));
    data.entries.push_back(std::move(entry));
  }

  // The rare patterns pooled by group and by span class on each axis.
  std::vector<std::vector<float>> groupX(Dictionary::NGroups), groupZ(Dictionary::NGroups), classX(NSpanClasses), classZ(NSpanClasses);
  for (uint32_t pattern = 0; pattern < nPatterns; ++pattern) {
    if (isCommon[pattern]) {
      continue;
    }
    const auto [rowSpan, colSpan] = spansOf(pattern);
    const int group = Dictionary::groupOf(rowSpan, colSpan);
    groupX[group].insert(groupX[group].end(), residualsX[pattern].begin(), residualsX[pattern].end());
    groupZ[group].insert(groupZ[group].end(), residualsZ[pattern].begin(), residualsZ[pattern].end());
    auto& cx = classX[Dictionary::spanClass(rowSpan)];
    auto& cz = classZ[Dictionary::spanClass(colSpan)];
    cx.insert(cx.end(), residualsX[pattern].begin(), residualsX[pattern].end());
    cz.insert(cz.end(), residualsZ[pattern].begin(), residualsZ[pattern].end());
  }
  const auto wholeX = medianAndWidth(layerX), wholeZ = medianAndWidth(layerZ);
  const auto pooled = [this](std::vector<float>& group, std::vector<float>& spanClass, const std::pair<float, float>& whole) {
    if (group.size() >= mSettings.minCount) {
      return medianAndWidth(group);
    }
    if (spanClass.size() >= mSettings.minCount) {
      return medianAndWidth(spanClass);
    }
    return whole;
  };
  for (int group = 0; group < Dictionary::NGroups; ++group) {
    DictionaryEntry entry;
    entry.isGroup = true;
    std::tie(entry.xOffset, entry.xWidth) = pooled(groupX[group], classX[group / NSpanClasses], wholeX);
    std::tie(entry.zOffset, entry.zWidth) = pooled(groupZ[group], classZ[group % NSpanClasses], wholeZ);
    data.entries.push_back(std::move(entry));
  }

  for (std::size_t id = 0; id < data.entries.size(); ++id) {
    const auto& entry = data.entries[id];
    if (!(entry.xWidth > 0.f && entry.zWidth > 0.f && std::isfinite(entry.xWidth) && std::isfinite(entry.zWidth))) {
      throw std::runtime_error(fmt::format("layer {}: entry {} has widths {} and {} cm", layer, id, entry.xWidth, entry.zWidth));
    }
  }
}

template <int DetID>
void BuildTopologyDictionary<DetID>::build()
{
  for (int layer = 0; layer < static_cast<int>(mInput.size()); ++layer) {
    auto& input = mInput[layer];
    if (input.crossings.empty()) {
      throw std::runtime_error(fmt::format("layer {} has no clusters", layer));
    }
    auto& data = mDictionary.mLayers[layer];
    data = LayerDictionary{};
    data.depth = measureDepth(layer, input);
    makeEntries(layer, input, data);
    input = LayerInput{};
    LOGP(info, "{} dictionary layer {}: depth {:.2f} um, {} common patterns", o2::detectors::DetID(DetID).getName(), layer, 1e4 * data.depth,
         data.entries.size() - Dictionary::NGroups);
  }
}

template <int DetID>
void BuildTopologyDictionary<DetID>::saveDictionaryRoot(const std::string& fileName) const
{
  TFile output(fileName.c_str(), "recreate");
  output.WriteObjectAny(&mDictionary, Dictionary::Class(), "ccdb_object");
  output.Close();
}

template class BuildTopologyDictionary<o2::detectors::DetID::TRK>;
template class BuildTopologyDictionary<o2::detectors::DetID::FT3>;

} // namespace o2::trkft3
