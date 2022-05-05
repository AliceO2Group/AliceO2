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

/// \file   MID/Workflow/src/ZeroSuppressionSpec.cxx
/// \brief  MID zero suppression spec
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   23 October 2020

#include "MIDWorkflow/ZeroSuppressionSpec.h"

#include <vector>
#include <gsl/gsl>
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/DecodedDataAggregator.h"
#include "MIDRaw/ROBoardResponse.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class ZeroSuppressionDeviceDPL
{
 public:
  ZeroSuppressionDeviceDPL(bool useMC) : mUseMC(useMC) {}

  void init(o2::framework::InitContext& ic)
  {
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    const auto patterns = pc.inputs().get<gsl::span<ColumnData>>("mid_data_mc");

    const auto inROFRecords = pc.inputs().get<gsl::span<ROFRecord>>("mid_data_mc_rof");

    const auto inMCContainer = mUseMC ? pc.inputs().get<const o2::dataformats::MCTruthContainer<MCLabel>*>("mid_data_mc_labels") : nullptr;

    o2::dataformats::MCTruthContainer<MCLabel> outMCContainer;

    std::vector<ROFRecord> zsROFs, tmpROFs(1);
    std::vector<ColumnData> zsData;
    for (auto& rof : inROFRecords) {
      mConverter.process(patterns.subspan(rof.firstEntry, rof.nEntries));
      if (!mConverter.getData().empty()) {
        std::vector<ROBoard> decodedData = mConverter.getData();
        mResponse.applyZeroSuppression(decodedData);
        tmpROFs.front().interactionRecord = rof.interactionRecord;
        tmpROFs.front().eventType = rof.eventType;
        tmpROFs.front().firstEntry = 0;
        tmpROFs.front().nEntries = decodedData.size();
        mAggregator.process(decodedData, tmpROFs);
        auto& tmpOut = mAggregator.getData();
        zsROFs.emplace_back(rof.interactionRecord, rof.eventType, zsData.size(), tmpOut.size());
        zsData.insert(zsData.end(), tmpOut.begin(), tmpOut.end());

        if (mUseMC) {
          for (auto outColIt = zsData.begin() + zsROFs.back().firstEntry, outEnd = zsData.begin() + zsROFs.back().firstEntry + zsROFs.back().nEntries; outColIt != outEnd; ++outColIt) {
            for (auto inColIt = patterns.begin() + rof.firstEntry, inEnd = patterns.begin() + rof.firstEntry + rof.nEntries; inColIt != inEnd; ++inColIt) {
              if (inColIt->deId == outColIt->deId && inColIt->columnId == outColIt->columnId) {
                auto inIdx = std::distance(patterns.begin(), inColIt);
                auto outIdx = std::distance(zsData.begin(), outColIt);
                outMCContainer.addElements(outIdx, inMCContainer->getLabels(inIdx));
                break;
              }
            }
          }
        }
      }
    }

    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DATA", 0, of::Lifetime::Timeframe}, zsData);
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DATAROF", 0, of::Lifetime::Timeframe}, zsROFs);
    if (mUseMC) {
      pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DATALABELS", 0, of::Lifetime::Timeframe}, outMCContainer);
    }
  }

 private:
  ColumnDataToLocalBoard mConverter{};
  DecodedDataAggregator mAggregator{};
  ROBoardResponse mResponse{};
  bool mUseMC{true};
};

framework::DataProcessorSpec getZeroSuppressionSpec(bool useMC)
{
  std::vector<of::InputSpec> inputSpecs{of::InputSpec{"mid_data_mc", header::gDataOriginMID, "DATAMC"}, of::InputSpec{"mid_data_mc_rof", header::gDataOriginMID, "DATAMCROF"}};

  std::vector<of::OutputSpec> outputSpecs{of::OutputSpec{header::gDataOriginMID, "DATA"}, of::OutputSpec{header::gDataOriginMID, "DATAROF"}};
  if (useMC) {
    inputSpecs.emplace_back(of::InputSpec{"mid_data_mc_labels", header::gDataOriginMID, "DATAMCLABELS"});
    outputSpecs.emplace_back(of::OutputSpec{header::gDataOriginMID, "DATALABELS"});
  }

  return of::DataProcessorSpec{
    "MIDZeroSuppression",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::ZeroSuppressionDeviceDPL>(useMC)}};
}
} // namespace mid
} // namespace o2