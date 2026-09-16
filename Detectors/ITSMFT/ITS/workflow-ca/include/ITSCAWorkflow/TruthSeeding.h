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

#ifndef O2_ITS_CA_TRUTH_SEEDING_H_
#define O2_ITS_CA_TRUTH_SEEDING_H_

#include <algorithm>
#include <limits>
#include <optional>
#include <gsl/span>

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsITS/TimeEstBC.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTTracking/ROFLookupTables.h"

namespace o2::its::ca
{
// Keep the readout window anchored to the actual input ROF records, including
// sparse/triggered input, while using the legacy interval and timing types.
inline std::optional<o2::its::TimeEstBC> truthSeedingWindow(
  gsl::span<const o2::itsmft::ROFRecord> rofs, const o2::InteractionRecord& origin,
  const o2::its::LayerTiming& timing) noexcept
{
  if (rofs.empty() || timing.mROFLength == 0) {
    return std::nullopt;
  }
  const int64_t offset = static_cast<int64_t>(timing.mROFDelay) + timing.mROFBias;
  const int64_t begin = std::max(int64_t{0}, rofs.front().getBCData().differenceInBC(origin) + offset - timing.mROFAddTimeErr);
  const int64_t end = rofs.back().getBCData().differenceInBC(origin) + offset + timing.mROFLength + timing.mROFAddTimeErr;
  if (begin >= end || end > std::numeric_limits<o2::its::TimeStampType>::max()) {
    return std::nullopt;
  }
  return o2::its::TimeEstBC{static_cast<uint32_t>(begin), static_cast<uint32_t>(end - begin)};
}

// Use the same origin as cluster loading. ROF delay/bias belong to the
// readout window, not to the collision timestamp. Preserve the existing
// forward uncertainty interval and select only collisions overlapping this TF.
inline std::optional<o2::its::TimeEstBC> truthSeedingTime(
  const o2::InteractionRecord& collision, const o2::InteractionRecord& origin,
  const o2::its::TimeEstBC& window, uint32_t duration) noexcept
{
  if (collision.isDummy() || window.lower() >= window.upper() || duration == 0) {
    return std::nullopt;
  }
  const auto begin = collision.differenceInBC(origin);
  const auto end = begin + duration;
  if (end <= window.lower() || begin >= window.upper() || end <= 0) {
    return std::nullopt;
  }
  // TimeEstBC has unsigned bounds; clip only the part preceding this origin.
  const auto clippedBegin = std::max(int64_t{0}, begin);
  if (end > std::numeric_limits<o2::its::TimeStampType>::max()) {
    return std::nullopt;
  }
  return o2::its::TimeEstBC{static_cast<uint32_t>(clippedBegin), static_cast<uint32_t>(end - clippedBegin)};
}
} // namespace o2::its::ca

#endif
