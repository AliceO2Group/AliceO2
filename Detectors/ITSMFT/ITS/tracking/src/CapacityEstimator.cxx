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

#include "ITStracking/CapacityEstimator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Framework/Logger.h"

namespace o2::its
{

struct CapacityEstimator::Impl {
  struct Entry {
    float ratio{0.f};
    float margin{0.f};
    size_t maxEmitted{0};
    uint32_t nSamples{0};
    uint32_t nLowStreak{0};
    uint32_t nOverflows{0};
  };

  explicit Impl(Config config) : cfg{config} {}

  Config cfg;
  mutable std::mutex mutex;
  std::unordered_map<KeyType, Entry> entries;
};

CapacityEstimator::CapacityEstimator() : CapacityEstimator{Config{}} {}

CapacityEstimator::CapacityEstimator(Config cfg) : mImpl{std::make_unique<Impl>(cfg)} {}

CapacityEstimator::~CapacityEstimator() = default;

void CapacityEstimator::reset()
{
  std::lock_guard lock{mImpl->mutex};
  mImpl->entries.clear();
}

size_t CapacityEstimator::capacity(uint64_t key, double scale) const
{
  if (!(scale > 0.)) {
    return 0;
  }
  std::lock_guard lock{mImpl->mutex};
  const auto it = mImpl->entries.find(key);
  if (it == mImpl->entries.end() || it->second.nSamples == 0) {
    return mImpl->cfg.floorSlots;
  }
  const auto& e = it->second;
  const double raw = double(e.ratio) * scale * double(e.margin);
  if (!std::isfinite(raw) || raw < 0.) {
    return mImpl->cfg.floorSlots;
  }
  // A ratio is only meaningful at the scale it was measured at. Learned on a handful of inputs it
  // can be arbitrarily large, and applying it to a scale orders of magnitude bigger asks for a slab
  // nobody can allocate. Bound the request by what this site has ever actually emitted: overshooting
  // burns memory that a bump allocator cannot give back, undershooting only costs one retry.
  const size_t ceiling = std::max(mImpl->cfg.floorSlots, static_cast<size_t>(double(e.maxEmitted) * double(mImpl->cfg.marginMax)));
  if (raw >= static_cast<double>(ceiling)) {
    return ceiling;
  }
  return std::max(mImpl->cfg.floorSlots, static_cast<size_t>(std::ceil(raw)));
}

size_t CapacityEstimator::peakCapacity(uint64_t key) const
{
  std::lock_guard lock{mImpl->mutex};
  const auto it = mImpl->entries.find(key);
  if (it == mImpl->entries.end() || it->second.maxEmitted == 0) {
    return mImpl->cfg.floorSlots;
  }
  const auto& e = it->second;
  const double raw = double(e.maxEmitted) * double(e.margin);
  if (!std::isfinite(raw) || raw >= static_cast<double>(std::numeric_limits<size_t>::max())) {
    return std::numeric_limits<size_t>::max();
  }
  return std::max(mImpl->cfg.floorSlots, static_cast<size_t>(std::ceil(raw)));
}

double CapacityEstimator::expected(uint64_t key, double scale) const
{
  if (!(scale > 0.)) {
    return 0.;
  }
  std::lock_guard lock{mImpl->mutex};
  const auto it = mImpl->entries.find(key);
  if (it == mImpl->entries.end() || it->second.nSamples == 0) {
    return 0.;
  }
  const double raw = double(it->second.ratio) * scale;
  return std::isfinite(raw) && raw > 0. ? raw : 0.;
}

void CapacityEstimator::update(uint64_t key, double scale, size_t emitted, size_t capacityUsed, bool overflowed, bool memoryLimited)
{
  if (!(scale > 0.)) {
    return;
  }
  std::lock_guard lock{mImpl->mutex};
  auto& e = mImpl->entries[key];
  const auto& cfg = mImpl->cfg;

  const bool firstSample = e.nSamples == 0;
  if (firstSample) {
    e.margin = cfg.marginInit;
  }
  const auto sample = static_cast<float>(double(emitted) / scale);
  e.ratio = firstSample ? sample : (cfg.alpha * sample) + ((1.f - cfg.alpha) * e.ratio);
  e.maxEmitted = std::max(e.maxEmitted, emitted);
  ++e.nSamples;

  if (memoryLimited) {
    e.nLowStreak = 0;
    e.margin = std::max(cfg.marginMin, e.margin * cfg.marginDown);
    return;
  }
  if (overflowed) {
    ++e.nOverflows;
    e.nLowStreak = 0;
    if (!firstSample) {
      const float shortfall = capacityUsed ? static_cast<float>(double(emitted) / double(capacityUsed)) : cfg.marginUp;
      e.margin = std::min(cfg.marginMax, e.margin * std::clamp(shortfall * cfg.marginOverflowSlack, 1.02f, cfg.marginUp));
    }
    return;
  }
  const float util = capacityUsed ? float(double(emitted) / double(capacityUsed)) : 1.f;
  if (util < cfg.lowWatermark) {
    if (++e.nLowStreak >= cfg.decayAfter) {
      e.margin = std::max(cfg.marginMin, e.margin * cfg.marginDown);
      e.nLowStreak = 0;
    }
  } else if (e.nLowStreak > 0) {
    --e.nLowStreak;
  }
}

void CapacityEstimator::print() const
{
  std::lock_guard lock{mImpl->mutex};
  std::vector<KeyType> keys;
  keys.reserve(mImpl->entries.size());
  for (const auto& [key, _] : mImpl->entries) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end(), [](KeyType a, KeyType b) {
    const auto da = decodeKey(a);
    const auto db = decodeKey(b);
    return std::tie(da.site, da.iteration, da.variant, da.slot) <
           std::tie(db.site, db.iteration, db.variant, db.slot);
  });
  if (keys.empty()) {
    return;
  }
  LOGP(info, "Printing CapacityEstimators:");
  for (const auto key : keys) {
    const auto& value = mImpl->entries.at(key);
    const auto decoded = decodeKey(key);
    LOGP(info, "\tSite:{} | iter:{} | var:({},{}) | slot:{} | ratio:{} | margin:{} | maxEmitted:{} | sam:{} | low:{} | overflows:{}", SlabSiteNames[decoded.site], decoded.iteration, getVariantHigh(decoded.variant), getVariantLow(decoded.variant), decoded.slot, value.ratio, value.margin, value.maxEmitted, value.nSamples, value.nLowStreak, value.nOverflows);
  }
}

} // namespace o2::its
