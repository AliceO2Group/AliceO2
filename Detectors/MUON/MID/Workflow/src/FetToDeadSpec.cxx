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

/// \file   MID/Workflow/src/FetToDeadSpec.cxx
/// \brief  Device to convert the FEE test event into dead channels
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   21 February 2022

#include "MIDWorkflow/FetToDeadSpec.h"

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

class FetToDeadDeviceDPL
{
 public:
  FetToDeadDeviceDPL(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks)
  {
    auto refMasks = makeDefaultMasksFromCrateConfig(feeIdConfig, crateMasks);
    mFetToDead.setMasks(refMasks);
  }

  void init(of::InitContext& ic)
  {
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

    std::vector<ColumnData> noisyChannels;
    noisyChannels.insert(noisyChannels.end(), data[1].begin(), data[1].end());

    std::vector<ROFRecord> noisyChannelsRof;
    noisyChannelsRof.insert(noisyChannelsRof.end(), dataRof[1].begin(), dataRof[1].end());

    auto deadChannelsPair = getDeadChannels(data[2], dataRof[2], data[0], dataRof[0]);

    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "NOISE", 0}, noisyChannels);
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "NOISEROF", 0}, noisyChannelsRof);
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DEAD", 0}, deadChannelsPair.first);
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DEADROF", 0}, deadChannelsPair.second);
  }

 private:
  FetToDead mFetToDead{}; ///< FET to dead channels converter

  std::pair<std::vector<ColumnData>, std::vector<ROFRecord>> getDeadChannels(gsl::span<const ColumnData> fetData, gsl::span<const ROFRecord> fetDataRof, gsl::span<const ColumnData> selfTrigData, gsl::span<const ROFRecord> selfTrigDataRof)
  {
    int64_t maxDiff = 1;
    int64_t minDiff = -maxDiff;

    ColumnDataHandler handler;
    std::vector<ColumnData> deadChannels;
    std::vector<ROFRecord> deadChannelROFs;
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
        if (bcDiff > maxDiff) {
          // ROFs are time ordered. If the difference is larger than the maximum difference for merging,
          // it means that the auxRofIt is in the future.
          // We break and compare it to the next rof.
          break;
        } else if (bcDiff >= minDiff) {
          // With the previous condition, this implies minDiff <= bcDiff <= maxDiff
          auto auxFet = selfTrigData.subspan(auxRofIt->firstEntry, auxRofIt->nEntries);
          handler.merge(auxFet);
        }
        // If bcDiff is < minDiff, it means that the auxRofIt is too much in the past
        // So we do nothing, i.e. we move to the next auxRofIt in the for loop
      }
      auto eventDeadChannels = mFetToDead.process(handler.getMerged());
      deadChannelROFs.emplace_back(rof.interactionRecord, rof.eventType, deadChannels.size(), eventDeadChannels.size());
      deadChannels.insert(deadChannels.end(), eventDeadChannels.begin(), eventDeadChannels.end());
    }
    return {deadChannels, deadChannelROFs};
  }
};

of::DataProcessorSpec getFetToDeadSpec(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks)
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
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::FetToDeadDeviceDPL>(feeIdConfig, crateMasks)},
    of::Options{}};
}
} // namespace mid
} // namespace o2