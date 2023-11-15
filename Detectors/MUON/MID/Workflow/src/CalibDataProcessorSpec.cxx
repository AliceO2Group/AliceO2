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

/// \file   MID/Workflow/src/CalibDataProcessorSpec.cxx
/// \brief  Device to convert the calibration data into a list of bad channel candidates
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   25 October 2022

#include "MIDWorkflow/CalibDataProcessorSpec.h"

#include <array>
#include <vector>
#include <gsl/gsl>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDBase/ColumnDataHandler.h"
#include "MIDFiltering/FetToDead.h"
#include "MIDFiltering/MaskMaker.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class CalibDataProcessorDPL
{
 public:
  CalibDataProcessorDPL(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks)
  {
    auto refMasks = makeDefaultMasksFromCrateConfig(feeIdConfig, crateMasks);
    mFetToDead.setMasks(refMasks);
  }

  void init(of::InitContext& ic)
  {
    mMinDiff = ic.options().get<int64_t>("mid-merge-fet-bc-diff-min");
    mMaxDiff = ic.options().get<int64_t>("mid-merge-fet-bc-diff-max");
  }

  void run(of::ProcessingContext& pc)
  {
    std::array<gsl::span<const ColumnData>, 3> data;
    std::array<gsl::span<const ROFRecord>, 3> dataRof;

    std::vector<of::InputSpec> filter = {
      {"check_data", of::ConcreteDataTypeMatcher{header::gDataOriginMID, "DATA"}, of::Lifetime::Timeframe},
      {"check_rof", of::ConcreteDataTypeMatcher{header::gDataOriginMID, "DATAROF"}, of::Lifetime::Timeframe},
    };

    for (auto const& inputRef : of::InputRecordWalker(pc.inputs(), filter)) {
      auto const* dh = framework::DataRefUtils::getHeader<o2::header::DataHeader*>(inputRef);
      auto subSpecIdx = static_cast<size_t>(dh->subSpecification);
      if (of::DataRefUtils::match(inputRef, "mid_data")) {
        data[subSpecIdx] = pc.inputs().get<gsl::span<o2::mid::ColumnData>>(inputRef);
      } else if (of::DataRefUtils::match(inputRef, "mid_data_rof")) {
        dataRof[subSpecIdx] = pc.inputs().get<gsl::span<o2::mid::ROFRecord>>(inputRef);
      }
    }

    mNoise.clear();
    mNoiseROF.clear();
    mDead.clear();
    mDeadROF.clear();

    mNoise.insert(mNoise.end(), data[1].begin(), data[1].end());
    mNoiseROF.insert(mNoiseROF.end(), dataRof[1].begin(), dataRof[1].end());

    mergeChannels(data[2], dataRof[2], data[0], dataRof[0]);

    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "NOISE", 0}, mNoise);
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "NOISEROF", 0}, mNoiseROF);
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DEAD", 0}, mDead);
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DEADROF", 0}, mDeadROF);
  }

 private:
  FetToDead mFetToDead{};           ///< FET to dead channels converter
  std::vector<ColumnData> mNoise;   ///< Merged noise
  std::vector<ROFRecord> mNoiseROF; ///< Merged noise ROFs
  std::vector<ColumnData> mDead;    ///< Merged dead
  std::vector<ROFRecord> mDeadROF;  ///< Merged dead ROFs
  int64_t mMinDiff = -1;            /// Maximum BC difference for FET merging
  int64_t mMaxDiff = 1;             /// Minimum BC difference for FET merging

  void mergeChannels(gsl::span<const ColumnData> fetData, gsl::span<const ROFRecord> fetDataRof, gsl::span<const ColumnData> selfTrigData, gsl::span<const ROFRecord> selfTrigDataRof)
  {
    // This method selects the self-triggered events that are close in time with a FET event and merges them with the fet event.
    // This is needed since the detector answer is not perfectly aligned in time.
    // Since calibration runs occur with no beam, all other self-triggers are actually noise.

    ColumnDataHandler handler;
    // The FET data can be split into different BCs.
    // Try to merge the expected FET data with the data in the close BCs
    // which are probably badly tagged FET data
    auto auxRofIt = selfTrigDataRof.begin();
    // Loop on FET ROF
    for (auto& rof : fetDataRof) {
      handler.clear();
      auto eventFetData = fetData.subspan(rof.firstEntry, rof.nEntries);
      handler.merge(eventFetData);
      for (; auxRofIt != selfTrigDataRof.end(); ++auxRofIt) {
        auto bcDiff = auxRofIt->interactionRecord.differenceInBC(rof.interactionRecord);
        if (bcDiff > mMaxDiff) {
          // ROFs are time ordered. If the difference is larger than the maximum difference for merging,
          // it means that the auxRofIt is in the future.
          // We break and compare it to the next rof.
          break;
        } else if (bcDiff >= mMinDiff) {
          // With the previous condition, this implies mMinDiff <= bcDiff <= mMaxDiff
          auto auxFet = selfTrigData.subspan(auxRofIt->firstEntry, auxRofIt->nEntries);
          handler.merge(auxFet);
        } else {
          // If bcDiff is < mMinDiff, it means that the auxRofIt is too much in the past
          // So this was actually noise
          mNoise.insert(mNoise.end(), selfTrigData.begin() + auxRofIt->firstEntry, selfTrigData.begin() + auxRofIt->getEndIndex());
          mNoiseROF.emplace_back(*auxRofIt);
        }
      }
      auto eventDeadChannels = mFetToDead.process(handler.getMerged());
      mDeadROF.emplace_back(rof.interactionRecord, rof.eventType, mDead.size(), eventDeadChannels.size());
      mDead.insert(mDead.end(), eventDeadChannels.begin(), eventDeadChannels.end());
    }
  }
};

of::DataProcessorSpec getCalibDataProcessorSpec(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks)
{
  std::vector<of::InputSpec> inputSpecs;
  inputSpecs.emplace_back("mid_data", of::ConcreteDataTypeMatcher(header::gDataOriginMID, "DATA"), of::Lifetime::Timeframe);
  inputSpecs.emplace_back("mid_data_rof", of::ConcreteDataTypeMatcher(header::gDataOriginMID, "DATAROF"), of::Lifetime::Timeframe);

  std::vector<of::OutputSpec> outputSpecs;
  outputSpecs.emplace_back(header::gDataOriginMID, "NOISE", 0);
  outputSpecs.emplace_back(header::gDataOriginMID, "NOISEROF", 0);
  outputSpecs.emplace_back(header::gDataOriginMID, "DEAD", 0);
  outputSpecs.emplace_back(header::gDataOriginMID, "DEADROF", 0);

  return of::DataProcessorSpec{
    "MIDFetToDead",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::CalibDataProcessorDPL>(feeIdConfig, crateMasks)},
    of::Options{
      {"mid-merge-fet-bc-diff-min", of::VariantType::Int, -1, {"Merge to FET if BC-BC_FET >= this value"}},
      {"mid-merge-fet-bc-diff-max", of::VariantType::Int, 1, {"Merge to FET if BC-BC_FET <= this value"}}}};
}
} // namespace mid
} // namespace o2