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

/// \file   MID/Workflow/src/MaskMakerSpec.cxx
/// \brief  Processor to compute the masks
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 may 2021

#include "MIDWorkflow/MaskMakerSpec.h"

#include <array>
#include <vector>
#include <chrono>
#include <gsl/gsl>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#include "MIDFiltering/ChannelScalers.h"
#include "MIDFiltering/FetToDead.h"
#include "MIDFiltering/MaskMaker.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class MaskMakerDeviceDPL
{
 public:
  MaskMakerDeviceDPL(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks)
  {
    mRefMasks = makeDefaultMasksFromCrateConfig(feeIdConfig, crateMasks);
    mFetToDead.setMasks(mRefMasks);
  }

  void init(of::InitContext& ic)
  {
    mThreshold = ic.options().get<double>("mid-mask-threshold");
    mNReset = ic.options().get<int>("mid-mask-reset");

    auto stop = [this]() {
      double scaleFactor = (mCounter == 0) ? 0 : 1.e6 / mCounter;
      LOG(info) << "Processing time / " << mCounter << " events: full: " << mTimer.count() * scaleFactor << " us  mask maker: " << mTimerMaskMaker.count() * scaleFactor << " us";
    };
    ic.services().get<of::CallbackService>().set<of::CallbackService::Id::Stop>(stop);
  }

  void run(of::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();

    gsl::span<const ColumnData> calibData, fetData;
    gsl::span<const ROFRecord> calibDataRof, fetDataRof;

    std::vector<of::InputSpec> filter = {
      {"check_data", of::ConcreteDataTypeMatcher{header::gDataOriginMID, "DATA"}, of::Lifetime::Timeframe},
      {"check_rof", of::ConcreteDataTypeMatcher{header::gDataOriginMID, "DATAROF"}, of::Lifetime::Timeframe},
    };

    for (auto const& inputRef : of::InputRecordWalker(pc.inputs(), filter)) {
      auto const* dh = framework::DataRefUtils::getHeader<o2::header::DataHeader*>(inputRef);
      if (of::DataRefUtils::match(inputRef, "mid_data")) {
        if (dh->subSpecification == 1) {
          calibData = pc.inputs().get<gsl::span<o2::mid::ColumnData>>(inputRef);
        } else if (dh->subSpecification == 2) {
          fetData = pc.inputs().get<gsl::span<o2::mid::ColumnData>>(inputRef);
        }
      }
      if (of::DataRefUtils::match(inputRef, "mid_data_rof")) {
        if (dh->subSpecification == 1) {
          calibDataRof = pc.inputs().get<gsl::span<o2::mid::ROFRecord>>(inputRef);
        } else if (dh->subSpecification == 2) {
          fetDataRof = pc.inputs().get<gsl::span<o2::mid::ROFRecord>>(inputRef);
        }
      }
    }

    unsigned long nEvents = calibDataRof.size();
    if (nEvents == 0) {
      return;
    }

    auto tAlgoStart = std::chrono::high_resolution_clock::now();

    for (auto& col : calibData) {
      mScalers[0].count(col);
    }

    for (auto& rof : fetDataRof) {
      auto subSet = fetData.subspan(rof.firstEntry, rof.nEntries);
      auto deadChannels = mFetToDead.process(subSet);
      for (auto& col : deadChannels) {
        mScalers[1].count(col);
      }
    }

    mCounter += nEvents;
    mCounterSinceReset += nEvents;

    if (mCounterSinceReset >= mNReset) {
      for (size_t itype = 0; itype < 2; ++itype) {
        auto masks = o2::mid::makeMasks(mScalers[itype], mCounterSinceReset, mThreshold, mRefMasks);
        pc.outputs().snapshot(of::Output{header::gDataOriginMID, "MASKS", static_cast<header::DataHeader::SubSpecificationType>(itype + 1), of::Lifetime::Timeframe}, masks);
      }
      mCounterSinceReset = 0;
      for (auto& scaler : mScalers) {
        scaler.reset();
      }
    }

    mTimerMaskMaker += std::chrono::high_resolution_clock::now() - tAlgoStart;

    mTimer += std::chrono::high_resolution_clock::now() - tStart;
  }

 private:
  FetToDead mFetToDead{};                           ///< FET to dead channels converter
  std::vector<ColumnData> mRefMasks{};              ///< Reference masks
  std::array<ChannelScalers, 2> mScalers{};         ///< Array fo channel scalers
  std::chrono::duration<double> mTimer{0};          ///< full timer
  std::chrono::duration<double> mTimerMaskMaker{0}; ///< mask maker timer
  unsigned long mCounter{0};                        ///< Total number of processed events
  unsigned long mCounterSinceReset{0};              ///< Total number of processed events since last reset
  double mThreshold{0.9};                           ///< Occupancy threshold for producing a mask
  int mNReset{1};                                   ///< Number of calibration events to be tested before checking the scalers
};

framework::DataProcessorSpec getMaskMakerSpec(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks)
{
  std::vector<of::InputSpec> inputSpecs;
  inputSpecs.emplace_back("mid_data", of::ConcreteDataTypeMatcher(header::gDataOriginMID, "DATA"), of::Lifetime::Timeframe);
  inputSpecs.emplace_back("mid_data_rof", of::ConcreteDataTypeMatcher(header::gDataOriginMID, "DATAROF"), of::Lifetime::Timeframe);

  std::vector<of::OutputSpec> outputSpecs{
    of::OutputSpec{header::gDataOriginMID, "MASKS", 1},
    of::OutputSpec{header::gDataOriginMID, "MASKS", 2}};

  return of::DataProcessorSpec{
    "MIDMaskMaker",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::MaskMakerDeviceDPL>(feeIdConfig, crateMasks)},
    of::Options{{"mid-mask-threshold", of::VariantType::Double, 0.9, {"Tolerated occupancy before producing a map"}}, {"mid-mask-reset", of::VariantType::Int, 100, {"Number of calibration events to be checked before resetting the scalers"}}}};
}
} // namespace mid
} // namespace o2