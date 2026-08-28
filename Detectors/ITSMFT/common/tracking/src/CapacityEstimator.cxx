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

#include "ITSMFTTracking/CapacityEstimator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Framework/Logger.h"
#include "ITSMFTTracking/SlabBumpAllocator.h"

namespace o2::itsmft::tracking
{

struct CapacityEstimator::Impl {
  struct Entry {
    float ratio{0.f};
    float margin{0.f};
    Statistics statistics{};
  };

  struct UndoRecord {
    bool existed{false};
    Entry previous{};
  };

  explicit Impl(Config config) : cfg{config} {}

  Config cfg;
  mutable std::mutex mutex;
  std::unordered_map<KeyType, Entry> entries;
  std::unordered_map<KeyType, UndoRecord> undo;
  bool transactionActive{false};

  void checkpointBeforeUpdate(KeyType key)
  {
    if (!transactionActive || undo.find(key) != undo.end()) {
      return;
    }
    const auto current = entries.find(key);
    if (current == entries.end()) {
      undo.emplace(key, UndoRecord{});
    } else {
      undo.emplace(key, UndoRecord{.existed = true, .previous = current->second});
    }
  }

  void observe(KeyType key, double scale, size_t requested, size_t granted, size_t emitted,
               size_t spilled, bool overflowed, bool memoryLimited)
  {
    // Record the first-touch undo state before entries[key] can insert or the
    // existing live entry can be modified. If undo insertion throws, the live
    // estimator remains unchanged and Tracker's failure path can roll back the
    // transaction without observing a partial update.
    checkpointBeforeUpdate(key);
    auto& e = entries[key];
    auto& statistics = e.statistics;
    statistics.requested += requested;
    statistics.granted += granted;
    statistics.emitted += emitted;
    statistics.spilled += spilled;

    const bool firstSample = statistics.samples == 0;
    if (firstSample) {
      e.margin = cfg.marginInit;
    }
    const auto sample = static_cast<float>(double(emitted) / scale);
    e.ratio = firstSample ? sample : (cfg.alpha * sample) + ((1.f - cfg.alpha) * e.ratio);
    statistics.maxEmitted = std::max(statistics.maxEmitted, emitted);
    ++statistics.samples;

    if (memoryLimited) {
      statistics.nLowStreak = 0;
      e.margin = std::max(cfg.marginMin, e.margin * cfg.marginDown);
      return;
    }
    if (overflowed) {
      ++statistics.overflowEvents;
      statistics.nLowStreak = 0;
      if (!firstSample) {
        const float shortfall = granted ? static_cast<float>(double(emitted) / double(granted)) : cfg.marginUp;
        e.margin = std::min(cfg.marginMax, e.margin * std::clamp(shortfall * cfg.marginOverflowSlack, 1.02f, cfg.marginUp));
      }
      return;
    }
    const float util = granted ? float(double(emitted) / double(granted)) : 1.f;
    if (util < cfg.lowWatermark) {
      if (++statistics.nLowStreak >= cfg.decayAfter) {
        e.margin = std::max(cfg.marginMin, e.margin * cfg.marginDown);
        statistics.nLowStreak = 0;
      }
    } else if (statistics.nLowStreak > 0) {
      --statistics.nLowStreak;
    }
  }
};

CapacityEstimator::CapacityEstimator() : CapacityEstimator{Config{}} {}

CapacityEstimator::CapacityEstimator(Config cfg) : mImpl{std::make_unique<Impl>(cfg)} {}

CapacityEstimator::~CapacityEstimator() = default;

void CapacityEstimator::reset()
{
  std::lock_guard lock{mImpl->mutex};
  mImpl->entries.clear();
  mImpl->undo.clear();
  mImpl->transactionActive = false;
}

void CapacityEstimator::beginTransaction()
{
  std::lock_guard lock{mImpl->mutex};
  if (mImpl->transactionActive) {
    throw std::logic_error{"CapacityEstimator transaction already active"};
  }
  assert(mImpl->undo.empty());
  mImpl->transactionActive = true;
}

void CapacityEstimator::commitTransaction() noexcept
{
  std::lock_guard lock{mImpl->mutex};
  mImpl->undo.clear();
  mImpl->transactionActive = false;
}

void CapacityEstimator::rollbackTransaction() noexcept
{
  std::lock_guard lock{mImpl->mutex};
  if (!mImpl->transactionActive) {
    return;
  }
  for (const auto& [key, record] : mImpl->undo) {
    if (record.existed) {
      const auto current = mImpl->entries.find(key);
      assert(current != mImpl->entries.end());
      current->second = record.previous;
    } else {
      mImpl->entries.erase(key);
    }
  }
  mImpl->undo.clear();
  mImpl->transactionActive = false;
}

size_t CapacityEstimator::capacity(uint64_t key, double scale) const
{
  if (!(scale > 0.)) {
    return 0;
  }
  std::lock_guard lock{mImpl->mutex};
  const auto it = mImpl->entries.find(key);
  if (it == mImpl->entries.end() || it->second.statistics.samples == 0) {
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
  const size_t ceiling = std::max(mImpl->cfg.floorSlots, static_cast<size_t>(double(e.statistics.maxEmitted) * double(mImpl->cfg.marginMax)));
  if (raw >= static_cast<double>(ceiling)) {
    return ceiling;
  }
  return std::max(mImpl->cfg.floorSlots, static_cast<size_t>(std::ceil(raw)));
}

size_t CapacityEstimator::peakCapacity(uint64_t key) const
{
  std::lock_guard lock{mImpl->mutex};
  const auto it = mImpl->entries.find(key);
  if (it == mImpl->entries.end() || it->second.statistics.maxEmitted == 0) {
    return mImpl->cfg.floorSlots;
  }
  const auto& e = it->second;
  const double raw = double(e.statistics.maxEmitted) * double(e.margin);
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
  if (it == mImpl->entries.end() || it->second.statistics.samples == 0) {
    return 0.;
  }
  const double raw = double(it->second.ratio) * scale;
  return std::isfinite(raw) && raw > 0. ? raw : 0.;
}

CapacityEstimator::Statistics CapacityEstimator::statistics(uint64_t key) const
{
  std::lock_guard lock{mImpl->mutex};
  const auto it = mImpl->entries.find(key);
  if (it == mImpl->entries.end()) {
    return {};
  }
  return it->second.statistics;
}

void CapacityEstimator::update(uint64_t key, double scale, size_t emitted, size_t capacityUsed, bool overflowed, bool memoryLimited)
{
  if (!(scale > 0.)) {
    return;
  }
  std::lock_guard lock{mImpl->mutex};
  mImpl->observe(key, scale, capacityUsed, capacityUsed, emitted,
                 overflowed && emitted > capacityUsed ? emitted - capacityUsed : 0,
                 overflowed, memoryLimited);
}

void CapacityEstimator::update(uint64_t key, double scale, size_t requested, size_t granted,
                               size_t emitted, size_t spilled, bool overflowed, bool memoryLimited)
{
  if (!(scale > 0.)) {
    return;
  }
  std::lock_guard lock{mImpl->mutex};
  mImpl->observe(key, scale, requested, granted, emitted, spilled, overflowed, memoryLimited);
}

void CapacityEstimator::update(uint64_t key, double scale, const SlabSinkStats& stats)
{
  update(key, scale, stats.requested, stats.capacity, stats.emitted, stats.spilled,
         stats.overflowed, stats.memoryLimited);
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
    const auto& statistics = value.statistics;
    const auto decoded = decodeKey(key);
    LOGP(info, "\tSite:{} | iter:{} | var:({},{}) | slot:{} | ratio:{} | margin:{} | maxEmitted:{} | samples:{} | low:{} | requested:{} | granted:{} | emitted:{} | spilled:{} | overflows:{}", SlabSiteNames[decoded.site], decoded.iteration, getVariantHigh(decoded.variant), getVariantLow(decoded.variant), decoded.slot, value.ratio, value.margin, statistics.maxEmitted, statistics.samples, statistics.nLowStreak, statistics.requested, statistics.granted, statistics.emitted, statistics.spilled, statistics.overflowEvents);
  }
}

} // namespace o2::itsmft::tracking
