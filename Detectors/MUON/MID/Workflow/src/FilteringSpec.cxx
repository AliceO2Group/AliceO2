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

/// \file   MID/Workflow/src/FilteringSpec.cxx
/// \brief  MID filtering spec
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   16 March 2022

#include "MIDWorkflow/FilteringSpec.h"

#include <vector>
#include <gsl/gsl>
#include <fmt/format.h>
#include "Framework/CCDBParamSpec.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#include "MIDWorkflow/ColumnDataSpecsUtils.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class FilteringDeviceDPL
{
 public:
  FilteringDeviceDPL(bool useMC, std::vector<of::OutputSpec> outputSpecs) : mUseMC(useMC)
  {
    mOutputs = specs::buildOutputs(outputSpecs);

    if (useMC) {
      mFillLabels = [](size_t inIdx, size_t outIdx, const o2::dataformats::MCTruthContainer<MCLabel>* inMCContainer, o2::dataformats::MCTruthContainer<MCLabel>& outMCContainer, const ColumnData& col) {
        auto labels = inMCContainer->getLabels(inIdx);
        for (auto& label : labels) {
          if (label.getDEId() == col.deId && label.getColumnId() == col.columnId) {
            bool hasLabel = false;
            for (int istrip = label.getFirstStrip(), last = label.getLastStrip(); istrip <= last; ++istrip) {
              if ((label.getCathode() == 1 && col.isNBPStripFired(istrip)) || (label.getCathode() == 0 && col.isBPStripFired(label.getStripInLine(istrip), label.getLine(istrip)))) {
                hasLabel = true;
                break;
              }
            }
            if (hasLabel) {
              outMCContainer.addElement(outIdx, label);
            }
          }
        }
      };
    }
  }

  void init(of::InitContext& ic)
  {
  }

  void finaliseCCDB(of::ConcreteDataMatcher matcher, void* obj)
  {
    if (matcher == of::ConcreteDataMatcher(header::gDataOriginMID, "BAD_CHANNELS", 0)) {
      LOG(info) << "Update MID_BAD_CHANNELS";
      auto* badChannels = static_cast<std::vector<ColumnData>*>(obj);
      mMasksHandler.switchOffChannels(*badChannels);
    }
  }

  void run(of::ProcessingContext& pc)
  {
    // Triggers finalizeCCDB
    pc.inputs().get<std::vector<ColumnData>*>("mid_bad_channels");

    auto data = specs::getData(pc, "mid_filter_in", EventType::Standard);
    auto inROFRecords = specs::getRofs(pc, "mid_filter_in", EventType::Standard);

    std::unique_ptr<const o2::dataformats::MCTruthContainer<MCLabel>> inMCContainer = mUseMC ? specs::getLabels(pc, "mid_filter_in") : nullptr;

    auto& maskedData = pc.outputs().make<std::vector<ColumnData>>(of::OutputRef{"mid_filter_out_0"});
    auto& maskedRofs = pc.outputs().make<std::vector<ROFRecord>>(of::OutputRef{"mid_filter_out_rof_0"});

    maskedData.reserve(data.size());
    maskedRofs.reserve(inROFRecords.size());

    o2::dataformats::MCTruthContainer<MCLabel> outMCContainer;

    for (auto& rof : inROFRecords) {
      auto firstEntry = maskedData.size();
      for (auto dataIt = data.begin() + rof.firstEntry, end = data.begin() + rof.getEndIndex(); dataIt != end; ++dataIt) {
        auto col = *dataIt;
        if (mMasksHandler.applyMask(col)) {
          // Data are not fully masked
          maskedData.emplace_back(col);
          auto inIdx = std::distance(data.begin(), dataIt);
          mFillLabels(inIdx, maskedData.size() - 1, inMCContainer.get(), outMCContainer, col);
        }
      }
      auto nEntries = maskedData.size() - firstEntry;
      if (nEntries > 0) {
        maskedRofs.emplace_back(rof);
        maskedRofs.back().firstEntry = firstEntry;
        maskedRofs.back().nEntries = nEntries;
      }
    }

    if (mUseMC) {
      pc.outputs().snapshot(mOutputs[2], outMCContainer);
    }
  }

 private:
  ChannelMasksHandler mMasksHandler{};
  bool mUseMC{false};
  std::vector<of::Output> mOutputs;
  std::function<void(size_t, size_t, const o2::dataformats::MCTruthContainer<MCLabel>*, o2::dataformats::MCTruthContainer<MCLabel>&, const ColumnData& col)> mFillLabels{[](size_t, size_t, const o2::dataformats::MCTruthContainer<MCLabel>*, o2::dataformats::MCTruthContainer<MCLabel>&, const ColumnData&) {}};
};

of::DataProcessorSpec getFilteringSpec(bool useMC, std::string_view inDesc, std::string_view outDesc)
{

  auto inputSpecs = specs::buildInputSpecs("mid_filter_in", inDesc, useMC);
  inputSpecs.emplace_back("mid_bad_channels", header::gDataOriginMID, "BAD_CHANNELS", 0, of::Lifetime::Condition, of::ccdbParamSpec("MID/Calib/BadChannels"));

  auto outputSpecs = specs::buildStandardOutputSpecs("mid_filter_out", outDesc, useMC);

  return of::DataProcessorSpec{
    "MIDFiltering",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::FilteringDeviceDPL>(useMC, outputSpecs)}};
}
} // namespace mid
} // namespace o2