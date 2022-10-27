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

#include "CommonUtils/NameConf.h"
#include "EMCALWorkflow/PublisherSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Headers/DataHeader.h"
#include <memory>
#include <utility>

namespace o2
{

namespace emcal
{

o2::framework::DataProcessorSpec createPublisherSpec(PublisherConf const& config, uint32_t subspec, bool propagateMC, workflow_reader::Creator creator)
{
  struct ProcessAttributes {
    std::shared_ptr<o2::framework::RootTreeReader> reader;
    std::string datatype;
    bool terminateOnEod;
    bool finished;
  };

  auto initFunction = [config, subspec, propagateMC, creator](o2::framework::InitContext& ic) {
    // get the option from the init context
    auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                  ic.options().get<std::string>("infile"));
    auto treename = ic.options().get<std::string>("treename");
    auto dtbrName = ic.options().get<std::string>(config.databranch.option.c_str());           // databranch name
    auto trgbrName = ic.options().get<std::string>(config.triggerrecordbranch.option.c_str()); // triggerbranch name
    auto mcbrName = ic.options().get<std::string>(config.mcbranch.option.c_str());             // mcbranch name
    auto nofEvents = ic.options().get<int>("nevents");
    auto publishingMode = nofEvents == -1 ? o2::framework::RootTreeReader::PublishingMode::Single : o2::framework::RootTreeReader::PublishingMode::Loop;

    auto processAttributes = std::make_shared<ProcessAttributes>();
    {
      using Reader = o2::framework::RootTreeReader;
      using TriggerInputType = std::vector<o2::emcal::TriggerRecord>;
      processAttributes->terminateOnEod = ic.options().get<bool>("terminate-on-eod");
      processAttributes->finished = false;
      processAttributes->datatype = config.databranch.defval;
      o2::header::DataHeader::SubSpecificationType subSpec = 0;
      processAttributes->reader = creator(treename.c_str(), // tree name
                                          filename.c_str(), // input file name
                                          nofEvents,        // number of entries to publish
                                          publishingMode,
                                          dtbrName.c_str(),  // databranch name
                                          trgbrName.c_str(), // triggerbranch name
                                          mcbrName.c_str()); // mcbranch name
    }

    auto processFunction = [processAttributes, propagateMC](o2::framework::ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }

      auto publish = [&processAttributes, &pc, propagateMC]() {
        if (processAttributes->reader->next()) {
          (*processAttributes->reader)(pc);
        } else {
          processAttributes->reader.reset();
          return false;
        }
        return true;
      };

      if (!publish()) {
        pc.services().get<o2::framework::ControlService>().endOfStream();
        pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
      }
    };

    return processFunction;
  };

  auto createOutputSpecs = [&config, subspec, propagateMC]() {
    std::vector<o2::framework::OutputSpec> outputSpecs;
    auto dto = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.dataoutput);
    auto tro = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.triggerrecordoutput);
    auto mco = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.mcoutput);
    outputSpecs.emplace_back(o2::framework::OutputSpec{{"output"}, dto.origin, dto.description, subspec, o2::framework::Lifetime::Timeframe});
    outputSpecs.emplace_back(o2::framework::OutputSpec{{"outputTRG"}, tro.origin, tro.description, subspec, o2::framework::Lifetime::Timeframe});
    if (propagateMC) {
      outputSpecs.emplace_back(o2::framework::OutputSpec{{"outputMC"}, mco.origin, mco.description, subspec, o2::framework::Lifetime::Timeframe});
    }
    return std::move(outputSpecs);
  };

  auto& dtb = config.databranch;
  auto& mcb = config.mcbranch;
  auto& trb = config.triggerrecordbranch;
  return o2::framework::DataProcessorSpec{
    config.processName.c_str(),
    o2::framework::Inputs{}, // no inputs
    {createOutputSpecs()},
    o2::framework::AlgorithmSpec(initFunction),
    o2::framework::Options{
      {"infile", o2::framework::VariantType::String, config.defaultFileName, {"Name of the input file"}},
      {"input-dir", o2::framework::VariantType::String, "none", {"Input directory"}},
      {"treename", o2::framework::VariantType::String, config.defaultTreeName.c_str(), {"Name of input tree"}},
      {dtb.option.c_str(), o2::framework::VariantType::String, dtb.defval.c_str(), {dtb.help.c_str()}},
      {trb.option.c_str(), o2::framework::VariantType::String, trb.defval.c_str(), {trb.help.c_str()}},
      {mcb.option.c_str(), o2::framework::VariantType::String, mcb.defval.c_str(), {mcb.help.c_str()}},
      {"nevents", o2::framework::VariantType::Int, -1, {"number of events to run"}},
      {"terminate-on-eod", o2::framework::VariantType::Bool, true, {"terminate on end-of-data"}},
    }};
}

} // namespace emcal

} // namespace o2
