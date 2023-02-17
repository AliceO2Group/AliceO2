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

/// \file EventFinderSpec.cxx
/// \brief Implementation of a data processor to group MCH digits based on MID information
///
/// \author Philippe Pillot, Subatech

#include "EventFinderSpec.h"

#include <chrono>
#include <string>
#include <vector>

#include <fmt/format.h>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Logger.h"
#include "CommonUtils/ConfigurableParam.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MCHTriggering/EventFinder.h"

namespace o2
{
namespace mch
{

using namespace o2::framework;
using namespace o2::dataformats;

class EventFinderTask
{
 public:
  EventFinderTask(bool useMC) : mUseMC{useMC} {}

  //_________________________________________________________________________________________________
  /// prepare the event finding
  void init(InitContext& ic)
  {
    LOG(info) << "initializing event finding";

    auto config = ic.options().get<std::string>("mch-config");
    if (!config.empty()) {
      conf::ConfigurableParam::updateFromFile(config, "MCHTriggering", true);
    }

    auto stop = [this]() {
      LOG(info) << "event finder duration = " << mElapsedTime.count() << " s";
    };
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(stop);
  }

  //_________________________________________________________________________________________________
  /// run the track matching
  void run(ProcessingContext& pc)
  {
    auto mchROFs = pc.inputs().get<gsl::span<mch::ROFRecord>>("mchrofs");
    auto mchDigits = pc.inputs().get<gsl::span<mch::Digit>>("mchdigits");
    auto mchLabels = mUseMC ? pc.inputs().get<MCTruthContainer<MCCompLabel>*>("mchlabels") : nullptr;
    auto midROFs = pc.inputs().get<gsl::span<mid::ROFRecord>>("midrofs");

    auto tStart = std::chrono::high_resolution_clock::now();
    mEventFinder.run(mchROFs, mchDigits, mchLabels.get(), midROFs);
    auto tEnd = std::chrono::high_resolution_clock::now();
    mElapsedTime += tEnd - tStart;

    pc.outputs().snapshot(OutputRef{"rofs"}, mEventFinder.getOutputROFs());
    pc.outputs().snapshot(OutputRef{"digits"}, mEventFinder.getOutputDigits());
    if (mUseMC) {
      pc.outputs().snapshot(OutputRef{"labels"}, mEventFinder.getOutputLabels());
    }

    LOGP(info, "produced {} ROFs with {} digits {}",
         mEventFinder.getOutputROFs().size(),
         mEventFinder.getOutputDigits().size(),
         mUseMC ? fmt::format("and {} labels", mEventFinder.getOutputLabels().getNElements()) : "");
  }

 private:
  bool mUseMC = false;                          ///< MC flag
  EventFinder mEventFinder{};                   ///< MID triggered event finder
  std::chrono::duration<double> mElapsedTime{}; ///< timer
};

//_________________________________________________________________________________________________
DataProcessorSpec getEventFinderSpec(bool useMC,
                                     std::string_view specName,
                                     std::string_view inputDigitDataDescription,
                                     std::string_view outputDigitDataDescription,
                                     std::string_view inputDigitRofDataDescription,
                                     std::string_view outputDigitRofDataDescription,
                                     std::string_view inputDigitLabelDataDescription,
                                     std::string_view outputDigitLabelDataDescription)
{
  std::string input =
    fmt::format("mchdigits:MCH/{}/0;mchrofs:MCH/{}/0", inputDigitDataDescription, inputDigitRofDataDescription);
  if (useMC) {
    input += fmt::format(";mchlabels:MCH/{}/0", inputDigitLabelDataDescription);
  }
  input += ";midrofs:MID/TRACKROFS/0";

  std::string output =
    fmt::format("digits:MCH/{}/0;rofs:MCH/{}/0", outputDigitDataDescription, outputDigitRofDataDescription);
  if (useMC) {
    output += fmt::format(";labels:MCH/{}/0", outputDigitLabelDataDescription);
  }

  std::vector<OutputSpec> outputs;
  auto matchers = select(output.c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
  }

  return DataProcessorSpec{
    specName.data(),
    Inputs{select(input.c_str())},
    outputs,
    AlgorithmSpec{adaptFromTask<EventFinderTask>(useMC)},
    Options{{"mch-config", VariantType::String, "", {"JSON or INI file with event finder parameters"}}}};
}

} // namespace mch
} // namespace o2
