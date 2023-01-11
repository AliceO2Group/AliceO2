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
#ifndef ALICEO2_FOCAL_RAWDECODERSPEC_H
#define ALICEO2_FOCAL_RAWDECODERSPEC_H

#include <array>
#include <map>
#include <unordered_map>
#include <vector>
#include <gsl/span>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFOCAL/Event.h"
#include "DataFormatsFOCAL/PixelHit.h"
#include "DataFormatsFOCAL/PixelChipRecord.h"
#include "DataFormatsFOCAL/TriggerRecord.h"
#include "FOCALReconstruction/PadDecoder.h"
#include "FOCALReconstruction/PixelDecoder.h"
#include "FOCALReconstruction/PixelMapper.h"

namespace o2::focal
{

class PixelChip;

namespace reco_workflow
{

class RawDecoderSpec : public framework::Task
{
 public:
  struct HBFData {
    std::vector<std::array<PadLayerEvent, constants::PADS_NLAYERS>> mPadEvents;
    std::vector<std::array<PixelLayerEvent, constants::PIXELS_NLAYERS>> mPixelEvent;
    std::vector<o2::InteractionRecord> mPixelTriggers;
  };
  RawDecoderSpec() = default;
  RawDecoderSpec(uint32_t outputSubspec, bool usePadData, bool usePixelData, bool debug) : mDebugMode(debug), mUsePadData(usePadData), mUsePixelData(usePixelData), mOutputSubspec(outputSubspec) {}
  ~RawDecoderSpec() override = default;

  void init(framework::InitContext& ctx) final;

  void run(framework::ProcessingContext& ctx) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  void sendOutput(framework::ProcessingContext& ctx);
  void resetContainers();
  int decodePadData(const gsl::span<const char> padWords, o2::InteractionRecord& interaction);
  void decodePadEvent(const gsl::span<const char> padWords, o2::InteractionRecord& interaction);
  int decodePixelData(const gsl::span<const char> pixelWords, o2::InteractionRecord& interaction, int fecID);
  std::array<PadLayerEvent, constants::PADS_NLAYERS> createPadLayerEvent(const o2::focal::PadData& data) const;
  void fillChipsToLayer(PixelLayerEvent& pixellayer, const gsl::span<const PixelChip>& chipData);
  void fillEventPixeHitContainer(std::vector<PixelHit>& eventHits, std::vector<PixelChipRecord>& eventChips, const PixelLayerEvent& pixelLayer, int layerIndex);
  void buildEvents();
  bool consistencyCheckPixelFEE(const std::unordered_map<int, int>& counters) const;
  int maxCounter(const std::unordered_map<int, int>& counters) const;
  void printCounters(const std::unordered_map<int, int>& counters) const;
  void printEvents(const std::unordered_map<int, std::vector<int>>& counters) const;
  bool checkEventsHBFConsistency(const std::unordered_map<int, std::vector<int>>& counters) const;
  void fillPixelEventHBFCount(const std::unordered_map<int, std::vector<int>>& counters);

  bool mDebugMode = false;
  bool mUsePadData = true;
  bool mUsePixelData = true;
  uint32_t mOutputSubspec = 0;
  PadDecoder mPadDecoder;
  PixelDecoder mPixelDecoder;
  std::map<o2::InteractionRecord, HBFData> mHBFs;
  std::vector<TriggerRecord> mOutputTriggerRecords;
  std::vector<PixelHit> mOutputPixelHits;
  std::vector<PixelChipRecord> mOutputPixelChips;
  std::vector<PadLayerEvent> mOutputPadLayers;

  // Some counters
  int mNumTimeframes = 0;
  int mNumHBFPads = 0;
  int mNumHBFPixels = 0;
  int mNumEventsPads = 0;
  int mNumEventsPixels = 0;
  int mNumInconsistencyPixelHBF = 0;
  int mNumInconsistencyPixelEvent = 0;
  int mNumInconsistencyPixelEventHBF = 0;
  std::map<int, int> mNumEventsHBFPads;
  std::map<int, int> mNumEventsHBFPixels;
  std::map<int, int> mNumHBFperTFPads;
  std::map<int, int> mNumHBFperTFPixels;
};

framework::DataProcessorSpec getRawDecoderSpec(bool askDISTSTF, uint32_t outputSubspec, bool usePadData, bool usePixelData, bool debugMode);

} // namespace reco_workflow

} // namespace o2::focal

#endif // ALICEO2_FOCAL_RAWDECODERSPEC_H
