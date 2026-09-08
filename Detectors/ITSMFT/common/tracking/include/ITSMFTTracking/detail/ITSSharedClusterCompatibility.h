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

#ifndef ALICEO2_ITSMFT_TRACKING_ITSSHAREDCLUSTERCOMPATIBILITY_H_
#define ALICEO2_ITSMFT_TRACKING_ITSSHAREDCLUSTERCOMPATIBILITY_H_

#ifndef GPUCA_GPUCODE

#include <cstddef>
#include <cstdint>
#include <gsl/span>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace o2::itsmft::tracking
{

// ITS output-compatibility sidecar. The pre-sort sequence maps accepted-track
// slots to global GenericTrack indices; entries() exposes the sealed sparse
// sequence after the final serial markTracks() pass.
struct ITSSharedClusterCompatibilityEntry {
  uint32_t genericTrackIndex{};
  bool hasSharedClusters{};
};

enum class ITSSharedClusterCompatibilitySealStep : uint8_t {
  BeforeReserve
};

class ITSSharedClusterCompatibility
{
 public:
  const std::vector<ITSSharedClusterCompatibilityEntry>& entries() const noexcept { return mEntries; }
  bool isSealed() const noexcept { return mSealed; }
  size_t pendingSize() const noexcept { return mPendingIndices.size(); }

  void clear() noexcept
  {
    mPendingIndices.clear();
    mEntries.clear();
    mSealed = false;
  }

  bool replaceFromAcceptedTrackIndices(gsl::span<const uint32_t> indices,
                                       gsl::span<const uint8_t> sharedFlags)
  {
    std::vector<ITSSharedClusterCompatibilityEntry> staged;
    staged.reserve(indices.size());
    uint32_t previous = 0;
    bool havePrevious = false;
    for (const auto index : indices) {
      if ((havePrevious && previous >= index) || index >= sharedFlags.size()) {
        return false;
      }
      staged.push_back({index, sharedFlags[index] != 0});
      previous = index;
      havePrevious = true;
    }
    mPendingIndices.clear();
    mEntries.swap(staged);
    mSealed = true;
    return true;
  }

  // Called after the final serial markTracks() pass, while indices still align
  // with the unreordered accepted slots.
  template <typename Tracks, typename Hook>
  bool sealFromMarkedTracks(const Tracks& tracks, Hook&& hook)
  {
    if (mSealed || tracks.size() != mPendingIndices.size()) {
      return false;
    }
    std::vector<ITSSharedClusterCompatibilityEntry> staged;
    hook(ITSSharedClusterCompatibilitySealStep::BeforeReserve);
    staged.reserve(mPendingIndices.size());
    uint32_t previous = 0;
    bool havePrevious = false;
    for (size_t i = 0; i < mPendingIndices.size(); ++i) {
      const auto index = mPendingIndices[i];
      if ((havePrevious && previous >= index)) {
        return false;
      }
      staged.push_back({index, tracks[i].hasSharedClusters()});
      previous = index;
      havePrevious = true;
    }
    mEntries.swap(staged);
    mSealed = true;
    return true;
  }

  template <typename Tracks>
  bool sealFromMarkedTracks(const Tracks& tracks)
  {
    return sealFromMarkedTracks(tracks, [](ITSSharedClusterCompatibilitySealStep) {});
  }

 private:
  friend class ITSSharedClusterCompatibilityTransaction;
  std::vector<uint32_t> mPendingIndices;
  std::vector<ITSSharedClusterCompatibilityEntry> mEntries;
  bool mSealed = false;
};

// Transactional association between a pre-sort accepted slot and GenericTrack.
class ITSSharedClusterCompatibilityTransaction
{
 public:
  explicit ITSSharedClusterCompatibilityTransaction(ITSSharedClusterCompatibility& sidecar)
    : mSidecar{sidecar}, mOldSize{sidecar.mPendingIndices.size()}
  {
  }

  bool validate(uint32_t genericTrackIndex) const noexcept
  {
    return !mSidecar.mSealed && (mSidecar.mPendingIndices.empty() || mSidecar.mPendingIndices.back() < genericTrackIndex);
  }
  void reserve() { mSidecar.mPendingIndices.reserve(mOldSize + 1); }
  void append(uint32_t genericTrackIndex)
  {
    if (!validate(genericTrackIndex)) {
      throw std::logic_error{"invalid ITS shared-cluster compatibility index"};
    }
    mSidecar.mPendingIndices.push_back(genericTrackIndex);
  }
  void rollback() noexcept { mSidecar.mPendingIndices.resize(mOldSize); }

 private:
  ITSSharedClusterCompatibility& mSidecar;
  size_t mOldSize;
};

} // namespace o2::itsmft::tracking

#endif // !GPUCA_GPUCODE

#endif // ALICEO2_ITSMFT_TRACKING_ITSSHAREDCLUSTERCOMPATIBILITY_H_
