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
#include "MIDWorkflow/ColumnDataSpecsUtils.h"

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
    const auto patterns = specs::getData(pc, "mid_zs_in", EventType::Standard);
    const auto inROFRecords = specs::getRofs(pc, "mid_zs_in", EventType::Standard);
    const auto inMCContainer = mUseMC ? specs::getLabels(pc, "mid_zs_in") : nullptr;

    o2::dataformats::MCTruthContainer<MCLabel> outMCContainer;

    auto& zsData = pc.outputs().make<std::vector<ColumnData>>(of::OutputRef{"mid_zs_out_0"});
    auto& zsROFs = pc.outputs().make<std::vector<ROFRecord>>(of::OutputRef{"mid_zs_out_rof_0"});

    zsData.reserve(patterns.size());
    zsROFs.reserve(inROFRecords.size());

    std::vector<ROFRecord> tmpROFs(1);
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

framework::DataProcessorSpec getZeroSuppressionSpec(bool useMC, std::string_view dataDesc)
{
  auto inputSpecs = specs::buildInputSpecs("mid_zs_in", dataDesc, useMC);
  auto outputSpecs = specs::buildStandardOutputSpecs("mid_zs_out", "DATA", useMC);

  return of::DataProcessorSpec{
    "MIDZeroSuppression",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::ZeroSuppressionDeviceDPL>(useMC)}};
}
} // namespace mid
} // namespace o2