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

#ifndef ALICEO2_ITSMFT_TRACKING_TEST_COMBINEDTRACKINGTESTSUPPORT_H_
#define ALICEO2_ITSMFT_TRACKING_TEST_COMBINEDTRACKINGTESTSUPPORT_H_

#include "TrackingParameterTestSupport.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ITSMFTTracking/Configuration.h"
#include "ITSCAWorkflow/PublicationAdapter.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/Tracker.h"
#include "ITSMFTTracking/TrackerTraits.h"
#include "ITSMFTTracking/ROFLookupTables.h"

namespace o2::itsmft::tracking::test
{

inline constexpr auto CombinedSurfaceCatalog = [] {
  std::array<SurfaceDescriptor, ITSNLayers + MFTNLayers> surfaces{};
  auto output = std::copy(kITSSurfaces.begin(), kITSSurfaces.end(), surfaces.begin());
  std::copy(kMFTSurfaces.begin(), kMFTSurfaces.end(), output);
  return surfaces;
}();

inline SurfaceCatalogView combinedCatalogView()
{
  return {CombinedSurfaceCatalog.data(), static_cast<uint32_t>(CombinedSurfaceCatalog.size())};
}

inline std::vector<LayerId> orderedSurfaceRange(uint16_t first, uint16_t count)
{
  std::vector<LayerId> result;
  result.reserve(count);
  for (uint16_t i = 0; i < count; ++i) {
    result.push_back(LayerId{static_cast<uint16_t>(first + i)});
  }
  return result;
}

inline TrackerInitialization makeCombinedConfiguration(const TrackingParameters& itsParams,
                                                       const TrackingParameters& mftParams)
{
  const std::vector<uint16_t> componentOffsets = {0, ITSNLayers};
  const auto combine = [&] {
    auto parameters = itsParams;
    parameters.NLayers = ITSNLayers + MFTNLayers;
    const auto concatenate = [](auto& output, const auto& prefix, const auto& suffix) {
      output = prefix;
      output.insert(output.end(), suffix.begin(), suffix.end());
    };
    concatenate(parameters.AddTimeError, itsParams.AddTimeError, mftParams.AddTimeError);
    concatenate(parameters.LayerResolution, itsParams.LayerResolution, mftParams.LayerResolution);
    concatenate(parameters.SystError2Row, itsParams.SystError2Row, mftParams.SystError2Row);
    concatenate(parameters.SystError2Col, itsParams.SystError2Col, mftParams.SystError2Col);
    const auto configuredSeedingLayers = [](const auto& input) {
      return input.SeedingLayers.empty() ? LayerMask::span(0, input.NLayers - 1) : input.SeedingLayers;
    };
    parameters.InactiveLayerMask = itsParams.InactiveLayerMask.value() |
                                   (mftParams.InactiveLayerMask.value() << itsParams.NLayers);
    parameters.SeedingLayers = configuredSeedingLayers(itsParams).value() |
                               (configuredSeedingLayers(mftParams).value() << itsParams.NLayers);
    parameters.StartLayerMask = LayerMask{(uint32_t{1} << (ITSNLayers + MFTNLayers)) - 1u};
    return parameters;
  };
  return {combinedCatalogView(), componentOffsets, {}, makeTrackingPlan(combine()), std::make_shared<BoundedMemoryResource>()};
}

class CombinedTrackingPlan
{
 public:
  CombinedTrackingPlan(std::vector<TrackingParameters> itsParams, std::vector<TrackingParameters> mftParams)
  {
    if (itsParams.size() != 1 || mftParams.size() != 1) {
      throw std::invalid_argument{"combined test application plan requires one iteration per detector"};
    }

    mConfiguration = makeCombinedConfiguration(itsParams[0], mftParams[0]);
    mTracker = std::make_unique<Tracker>();
    mTraits = std::make_unique<TrackerTraits>();
  }

  CombinedTrackingPlan(const CombinedTrackingPlan&) = delete;
  CombinedTrackingPlan& operator=(const CombinedTrackingPlan&) = delete;

  void adoptFrame(TimeFrame& frame)
  {
    mFrame = &frame;
    if (!mTracker->initialize(frame, mConfiguration)) {
      throw std::runtime_error{"combined test application plan failed to configure the TimeFrame"};
    }
  }
  void setBz(float bz)
  {
    mFrame->setBz(bz);
  }
  void setNThreads(int n)
  {
    mTraits->setNThreads(n, mArena);
  }

  Tracker& itsTracker() noexcept { return *mTracker; }
  Tracker& mftTracker() noexcept { return *mTracker; }
  bool runITS()
  {
    auto result = mTracker->run(*mFrame, *mTraits);
    mLastResult = result;
    if (result) {
      const auto& statistics = mTracker->getRunStatistics();
      const auto configurations = mTracker->getIterationConfigurations();
      std::size_t firstTrack = 0;
      for (std::size_t i = 0; i < configurations.size(); ++i) {
        if (i >= statistics.acceptedTrackCounts.size() ||
            statistics.acceptedTrackCounts[i] > mFrame->getGenericTracks().size() - firstTrack) {
          throw std::runtime_error{"failed to prepare ITS shared-cluster flags"};
        }
        std::vector<uint32_t> selected;
        for (std::size_t index = 0; index < statistics.acceptedTrackCounts[i]; ++index) {
          const auto globalIndex = firstTrack + index;
          if (mFrame->getGenericTracks()[globalIndex].innerState.kind == SurfaceKind::Cylinder) {
            selected.push_back(static_cast<uint32_t>(globalIndex));
          }
        }
        if (!mITSPublicationAdapter.completeAccepted(
              selected, configurations[i].parameters, *mFrame, i + 1 == configurations.size())) {
          throw std::runtime_error{"failed to prepare ITS shared-cluster flags"};
        }
        firstTrack += statistics.acceptedTrackCounts[i];
      }
    } else {
      mITSPublicationAdapter.reset();
    }
    return result;
  }
  bool runMFT()
  {
    if (!mLastResult) {
      runITS();
    }
    auto result = *mLastResult;
    return result;
  }
  RuntimeROFViews getITSROFViews() const noexcept { return {mITSROFOverlapTable.getView(), mITSROFVertexLookupTable.getView(), mITSMultiplicityMask.getView(), mITSUPCMask.getView()}; }
  RuntimeROFViews getMFTROFViews() const noexcept { return {mMFTROFOverlapTable.getView(), mMFTROFVertexLookupTable.getView(), mMFTMultiplicityMask.getView(), mMFTUPCMask.getView()}; }
  void clearPublicationSidecars() noexcept
  {
    mITSPublicationAdapter.reset();
    mLastResult.reset();
  }

  void validateSources(const ClusterSourceInput& itsSource,
                       const ClusterSourceInput& mftSource) const
  {
    if (itsSource.id != ClusterSourceId{0} || itsSource.detector != o2::detectors::DetID::ITS) {
      throw std::runtime_error("Invalid ITS source");
    }
    if (mftSource.id != ClusterSourceId{1} || mftSource.detector != o2::detectors::DetID::MFT) {
      throw std::runtime_error("Invalid MFT source");
    }
  }

  SurfaceCatalogView catalogView() const noexcept { return combinedCatalogView(); }
  void configureRofTables(const TestClusterSourceInput& itsSource, const TestClusterSourceInput& mftSource)
  {
    auto configure = [](auto& overlap, auto& vertex, auto& mask, const auto& timing, uint32_t nROFs, int layers) {
      o2::its::LayerTiming layerTiming{};
      layerTiming.mNROFsTF = nROFs;
      layerTiming.mROFLength = timing.rofLength;
      layerTiming.mROFDelay = timing.rofDelay;
      layerTiming.mROFBias = timing.rofBias;
      layerTiming.mROFAddTimeErr = timing.rofAddTimeErr;
      for (int layer = 0; layer < layers; ++layer) {
        overlap.defineLayer(layer, layerTiming);
        vertex.defineLayer(layer, layerTiming);
      }
      overlap.init();
      vertex.init();
      mask = std::remove_cvref_t<decltype(mask)>{overlap};
      mask.resetMask();
      for (int layer = 0; layer < layers; ++layer) {
        mask.setROFsEnabled(layer, 0, static_cast<int>(nROFs), 1);
      }
    };
    configure(mITSROFOverlapTable, mITSROFVertexLookupTable, mITSMultiplicityMask, itsSource.timing, static_cast<uint32_t>(itsSource.rofs.size()), ITSNLayers);
    configure(mMFTROFOverlapTable, mMFTROFVertexLookupTable, mMFTMultiplicityMask, mftSource.timing, static_cast<uint32_t>(mftSource.rofs.size()), MFTNLayers);
  }

  const TimeFrameScratch& getITSScratch() const noexcept { return mFrame->getScratch(); }
  const TimeFrameScratch& getMFTScratch() const noexcept { return mFrame->getScratch(); }
  gsl::span<const LayerId> getITSLayerMapping() const noexcept { return mITSLayerMapping; }
  gsl::span<const LayerId> getMFTLayerMapping() const noexcept { return mMFTLayerMapping; }
  gsl::span<const uint8_t> getITSSharedClusterFlags() const noexcept
  {
    return mITSPublicationAdapter.sharedClusterFlags();
  }
  TraversalTopologyView getITSLayoutView() const noexcept
  {
    const auto* configuration = mTracker == nullptr ? nullptr : mTracker->getIterationConfiguration(0);
    return mFrame != nullptr && configuration != nullptr && mTracker->isConfiguredFor(*mFrame)
             ? configuration->getTopologyView(mFrame->getDetectorConfiguration().getSurfaceCatalog())
             : TraversalTopologyView{};
  }
  TraversalTopologyView getMFTLayoutView() const noexcept { return getITSLayoutView(); }

 private:
  const std::vector<LayerId> mITSLayerMapping = orderedSurfaceRange(0, ITSNLayers);
  const std::vector<LayerId> mMFTLayerMapping = orderedSurfaceRange(ITSNLayers, MFTNLayers);
  TrackerInitialization mConfiguration;
  TimeFrame* mFrame = nullptr;
  std::unique_ptr<Tracker> mTracker;
  std::unique_ptr<TrackerTraits> mTraits;
  std::optional<bool> mLastResult;
  o2::its::ca::PublicationAdapter mITSPublicationAdapter;
  o2::its::ROFOverlapTable<ITSNLayers> mITSROFOverlapTable;
  o2::its::ROFVertexLookupTable<ITSNLayers> mITSROFVertexLookupTable;
  o2::its::ROFMaskTable<ITSNLayers> mITSMultiplicityMask;
  o2::its::ROFMaskTable<ITSNLayers> mITSUPCMask;
  o2::its::ROFOverlapTable<MFTNLayers> mMFTROFOverlapTable;
  o2::its::ROFVertexLookupTable<MFTNLayers> mMFTROFVertexLookupTable;
  o2::its::ROFMaskTable<MFTNLayers> mMFTMultiplicityMask;
  o2::its::ROFMaskTable<MFTNLayers> mMFTUPCMask;
  std::shared_ptr<tbb::task_arena> mArena;
};

} // namespace o2::itsmft::tracking::test

#endif // ALICEO2_ITSMFT_TRACKING_TEST_COMBINEDTRACKINGTESTSUPPORT_H_
