// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "PHOSWorkflow/PublisherSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Headers/DataHeader.h"
#include "DPLUtils/RootTreeReader.h"
#include "Framework/DataSpecUtils.h"
#include <memory>
#include <utility>

namespace o2
{

namespace phos
{

o2::framework::DataProcessorSpec getPublisherSpec(PublisherConf const& config, bool propagateMC, bool createMCMap)
{
  struct ProcessAttributes {
    std::shared_ptr<o2::framework::RootTreeReader> reader;
    std::string datatype;
    bool terminateOnEod;
    bool finished;
  };

  auto initFunction = [config, propagateMC, createMCMap](o2::framework::InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("infile");
    auto treename = ic.options().get<std::string>("treename");
    auto dtbrName = ic.options().get<std::string>(config.databranch.option.c_str());     // databranch name
    auto dttrbrName = ic.options().get<std::string>(config.datatrbranch.option.c_str()); // datatrigrec name
    auto mcbrName = ic.options().get<std::string>(config.mcbranch.option.c_str());       // mcbranch name
    auto mcmapbrName = ic.options().get<std::string>(config.mcmapbranch.option.c_str()); // mc map branch name
    auto nofEvents = ic.options().get<int>("nevents");
    // auto publishingMode = nofEvents == -1 ? o2::framework::RootTreeReader::PublishingMode::Single : o2::framework::RootTreeReader::PublishingMode::Loop;
    auto publishingMode = o2::framework::RootTreeReader::PublishingMode::Single;

    auto processAttributes = std::make_shared<ProcessAttributes>();
    {
      processAttributes->terminateOnEod = ic.options().get<bool>("terminate-on-eod");
      processAttributes->finished = false;
      processAttributes->datatype = config.databranch.defval;
      auto dto = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.dataoutput);
      auto dttro = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.datatroutput);
      auto mco = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.mcoutput);
      auto mcmapo = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.mcmapoutput);
      constexpr auto persistency = o2::framework::Lifetime::Timeframe;
      o2::header::DataHeader::SubSpecificationType subSpec = 0;
      if (propagateMC) {
        if (!createMCMap) {
          processAttributes->reader = std::make_shared<o2::framework::RootTreeReader>(treename.c_str(), // tree name
                                                                                      filename.c_str(), // input file name
                                                                                      nofEvents,        // number of entries to publish
                                                                                      publishingMode,
                                                                                      o2::framework::Output{dto.origin, dto.description, subSpec, persistency},
                                                                                      dtbrName.c_str(), // name of data branch
                                                                                      o2::framework::Output{dttro.origin, dttro.description, subSpec, persistency},
                                                                                      dttrbrName.c_str(), // name of data triggerrecords branch
                                                                                      o2::framework::Output{mco.origin, mco.description, subSpec, persistency},
                                                                                      mcbrName.c_str() // name of mc label branch
          );
        } else {
          processAttributes->reader = std::make_shared<o2::framework::RootTreeReader>(treename.c_str(), // tree name
                                                                                      filename.c_str(), // input file name
                                                                                      nofEvents,        // number of entries to publish
                                                                                      publishingMode,
                                                                                      o2::framework::Output{dto.origin, dto.description, subSpec, persistency},
                                                                                      dtbrName.c_str(), // name of data branch
                                                                                      o2::framework::Output{dttro.origin, dttro.description, subSpec, persistency},
                                                                                      dttrbrName.c_str(), // name of data triggerrecords branch
                                                                                      o2::framework::Output{mco.origin, mco.description, subSpec, persistency},
                                                                                      mcbrName.c_str(), // name of mc label branch
                                                                                      o2::framework::Output{mcmapo.origin, mcmapo.description, subSpec, persistency},
                                                                                      mcmapbrName.c_str() // name of mc label branch
          );
        }
      } else {
        processAttributes->reader = std::make_shared<o2::framework::RootTreeReader>(treename.c_str(), // tree name
                                                                                    filename.c_str(), // input file name
                                                                                    nofEvents,        // number of entries to publish
                                                                                    publishingMode,
                                                                                    o2::framework::Output{dto.origin, dto.description, subSpec, persistency},
                                                                                    dtbrName.c_str(), // name of data branch
                                                                                    o2::framework::Output{dttro.origin, dttro.description, subSpec, persistency},
                                                                                    dttrbrName.c_str() // name of data tr branch
        );
      }
    }

    auto processFunction = [processAttributes, propagateMC, createMCMap](o2::framework::ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }

      auto publish = [&processAttributes, &pc, propagateMC, createMCMap]() {
        o2::phos::PHOSBlockHeader phosheader(true);
        if (processAttributes->reader->next()) {
          (*processAttributes->reader)(pc, phosheader);
        } else {
          processAttributes->reader.reset();
          return false;
        }
        return true;
      };

      bool active(true);
      if (!publish()) {
        active = false;
        // Send dummy header with no payload option
        o2::phos::PHOSBlockHeader dummyheader(false);
        pc.outputs().snapshot(o2::framework::OutputRef{"output", 0, {dummyheader}}, 0);
        pc.outputs().snapshot(o2::framework::OutputRef{"outputTR", 0, {dummyheader}}, 0);
        if (propagateMC) {
          pc.outputs().snapshot(o2::framework::OutputRef{"outputMC", 0, {dummyheader}}, 0);
          if (createMCMap) {
            pc.outputs().snapshot(o2::framework::OutputRef{"outputMCmap", 0, {dummyheader}}, 0);
          }
        }
      }
      if ((processAttributes->finished = (active == false)) && processAttributes->terminateOnEod) {
        pc.services().get<o2::framework::ControlService>().endOfStream();
        pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
      }
    };

    return processFunction;
  };

  auto createOutputSpecs = [&config, propagateMC, createMCMap]() {
    std::vector<o2::framework::OutputSpec> outputSpecs;
    auto dto = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.dataoutput);
    auto dttro = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.datatroutput);
    auto mco = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.mcoutput);
    auto mcmapo = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.mcmapoutput);
    outputSpecs.emplace_back(o2::framework::OutputSpec{{"output"}, dto.origin, dto.description, 0, o2::framework::Lifetime::Timeframe});
    outputSpecs.emplace_back(o2::framework::OutputSpec{{"outputTR"}, dttro.origin, dttro.description, 0, o2::framework::Lifetime::Timeframe});
    if (propagateMC) {
      outputSpecs.emplace_back(o2::framework::OutputSpec{{"outputMC"}, mco.origin, mco.description, 0, o2::framework::Lifetime::Timeframe});
      if (createMCMap) {
        outputSpecs.emplace_back(o2::framework::OutputSpec{{"outputMCmap"}, mcmapo.origin, mcmapo.description, 0, o2::framework::Lifetime::Timeframe});
      }
    }
    return std::move(outputSpecs);
  };

  auto& dtb = config.databranch;
  auto& dttrb = config.datatrbranch;
  auto& mcb = config.mcbranch;
  auto& mcmapb = config.mcmapbranch;
  return o2::framework::DataProcessorSpec{
    config.processName.c_str(),
    o2::framework::Inputs{}, // no inputs
    {createOutputSpecs()},
    o2::framework::AlgorithmSpec(initFunction),
    o2::framework::Options{
      {"infile", o2::framework::VariantType::String, "phosdigits.root", {"Name of the input file"}},
      {"treename", o2::framework::VariantType::String, config.defaultTreeName.c_str(), {"Name of input tree"}},
      {dtb.option.c_str(), o2::framework::VariantType::String, dtb.defval.c_str(), {dtb.help.c_str()}},
      {dttrb.option.c_str(), o2::framework::VariantType::String, dttrb.defval.c_str(), {dttrb.help.c_str()}},
      {mcb.option.c_str(), o2::framework::VariantType::String, mcb.defval.c_str(), {mcb.help.c_str()}},
      {mcmapb.option.c_str(), o2::framework::VariantType::String, mcmapb.defval.c_str(), {mcmapb.help.c_str()}},
      {"nevents", o2::framework::VariantType::Int, -1, {"number of events to run"}},
      {"terminate-on-eod", o2::framework::VariantType::Bool, true, {"terminate on end-of-data"}},
    }};
}

} // namespace phos

} // namespace o2
