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

#ifndef ALICEO2_ITS_CA_PUBLICATIONADAPTER_H_
#define ALICEO2_ITS_CA_PUBLICATIONADAPTER_H_

#ifndef GPUCA_GPUCODE

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "DetectorsCommonDataFormats/DetID.h"
#include "GPUCommonMath.h"
#include "ITSMFTTracking/detail/ITSSharedClusterCompatibility.h"
#include "ITSMFTTracking/GenericTrack.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/MathUtils.h"

namespace o2::its::ca
{

// Workflow-owned ITS compatibility for generic tracking results.
class PublicationAdapter
{
 public:
  void adoptITSSharedClusterCompatibility(o2::itsmft::tracking::ITSSharedClusterCompatibility* sidecar) noexcept { mSidecar = sidecar; }
  o2::itsmft::tracking::ITSSharedClusterCompatibility* getITSSharedClusterCompatibility() const noexcept { return mSidecar; }

  bool completeAccepted(gsl::span<const uint32_t> trackIndices,
                        const o2::itsmft::IterationParameters& params,
                        const o2::itsmft::tracking::TimeFrame& frame,
                        bool final)
  {
    if (mSidecar == nullptr) {
      return true;
    }
    if (!stageSharedClusterFlags(trackIndices, params, frame)) {
      return false;
    }
    return !final || mSidecar->replaceFromAcceptedTrackIndices(mAcceptedTrackIndices, mSharedClusterFlags);
  }

  void reset() noexcept
  {
    mSharedClusterFlags.clear();
    mAcceptedTrackIndices.clear();
    if (mSidecar != nullptr) {
      mSidecar->clear();
    }
  }

  class Cleanup
  {
   public:
    explicit Cleanup(PublicationAdapter& adapter) : mAdapter(adapter) { mAdapter.reset(); }
    Cleanup(const Cleanup&) = delete;
    Cleanup& operator=(const Cleanup&) = delete;
    ~Cleanup() noexcept { mAdapter.reset(); }

   private:
    PublicationAdapter& mAdapter;
  };
  Cleanup cleanupOnExit() { return Cleanup{*this}; }

 private:
  struct SharedClusterTrackInfo {
    int layer{-1};
    uint32_t clusterId{std::numeric_limits<uint32_t>::max()};
    int rof{-1};
    float phi{0.f};
    float eta{0.f};
    int charge{0};
  };

  static std::optional<SharedClusterTrackInfo> makeSharedClusterTrackInfo(const o2::itsmft::tracking::GenericTrack& track,
                                                                          const o2::itsmft::tracking::TimeFrame& frame)
  {
    const int layer = track.hitLayers.first();
    const auto& references = frame.getTrackClusterIndices();
    if (layer < 0 || !isValidTrackRange(track, static_cast<uint32_t>(references.size())) ||
        track.firstClusterRef == track.clusterRefEnd ||
        static_cast<std::size_t>(layer) >= frame.getLayout().size()) {
      return std::nullopt;
    }
    const auto& reference = references[track.firstClusterRef];
    if (reference.layer != o2::itsmft::tracking::LayerId{static_cast<uint16_t>(layer)} || !reference.isValid()) {
      return std::nullopt;
    }
    const auto& state = track.innerState;
    if (!state.hasRecognizedKind() || !o2::gpu::GPUCommonMath::Finite(state.parameters[3]) ||
        !o2::gpu::GPUCommonMath::Finite(state.parameters[4])) {
      return std::nullopt;
    }
    const float phi = state.kind == o2::itsmft::tracking::SurfaceKind::Cylinder ? std::asin(state.parameters[2]) + state.alpha : state.parameters[2];
    const float eta = std::asinh(state.parameters[3]);
    if (!o2::gpu::GPUCommonMath::Finite(phi) || !o2::gpu::GPUCommonMath::Finite(eta)) {
      return std::nullopt;
    }
    return SharedClusterTrackInfo{layer, reference.clusterId, frame.getClusterROF(layer, static_cast<int>(reference.clusterId)),
                                  phi, eta, state.parameters[4] < 0.f ? -1 : 1};
  }

  bool stageSharedClusterFlags(gsl::span<const uint32_t> trackIndices,
                               const o2::itsmft::IterationParameters& params,
                               const o2::itsmft::tracking::TimeFrame& frame)
  {
    mAcceptedTrackIndices.reserve(mAcceptedTrackIndices.size() + trackIndices.size());
    for (const auto index : trackIndices) {
      if (index >= frame.getGenericTracks().size() ||
          (!mAcceptedTrackIndices.empty() && mAcceptedTrackIndices.back() >= index)) {
        return false;
      }
      mAcceptedTrackIndices.push_back(index);
    }
    if (!trackIndices.empty() && mSharedClusterFlags.size() <= trackIndices.back()) {
      mSharedClusterFlags.resize(static_cast<std::size_t>(trackIndices.back()) + 1, 0);
    }
    if (!params.AllowSharingFirstCluster) {
      return true;
    }
    std::vector<SharedClusterTrackInfo> trackInfo;
    trackInfo.reserve(trackIndices.size());
    for (const auto index : trackIndices) {
      const auto info = makeSharedClusterTrackInfo(frame.getGenericTracks()[index], frame);
      if (!info) {
        return false;
      }
      trackInfo.push_back(*info);
    }
    for (size_t first = 0; first < trackInfo.size(); ++first) {
      for (size_t second = first + 1; second < trackInfo.size(); ++second) {
        if (trackInfo[second].layer != trackInfo[first].layer || trackInfo[second].clusterId != trackInfo[first].clusterId) {
          continue;
        }
        if (trackInfo[first].rof != trackInfo[second].rof) {
          continue;
        }
        if (!o2::its::math_utils::isPhiDifferenceBelow(trackInfo[first].phi, trackInfo[second].phi, params.SharedClusterMaxDeltaPhi)) {
          continue;
        }
        if (std::abs(trackInfo[first].eta - trackInfo[second].eta) > params.SharedClusterMaxDeltaEta) {
          continue;
        }
        if (params.SharedClusterOppositeSign && trackInfo[first].charge == trackInfo[second].charge) {
          continue;
        }
        mSharedClusterFlags[trackIndices[first]] = 1;
        mSharedClusterFlags[trackIndices[second]] = 1;
      }
    }
    return true;
  }

  o2::itsmft::tracking::ITSSharedClusterCompatibility* mSidecar = nullptr;
  std::vector<uint8_t> mSharedClusterFlags;
  std::vector<uint32_t> mAcceptedTrackIndices;
};

} // namespace o2::its::ca

#endif // !GPUCA_GPUCODE

#endif // ALICEO2_ITS_CA_PUBLICATIONADAPTER_H_
