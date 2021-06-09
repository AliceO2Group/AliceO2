// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SimReaderSpec.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "Steer/HitProcessingManager.h"
#include "Steer/InteractionSampler.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DetectorsRaw/HBFUtils.h"
#include <FairMQLogger.h>
#include <TMessage.h> // object serialization
#include <memory>     // std::unique_ptr
#include <cstring>    // memcpy
#include <string>     // std::string
#include <cassert>
#include <chrono>
#include <thread>
#include <algorithm>

using namespace o2::framework;
namespace o2lhc = o2::constants::lhc;

using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
namespace o2
{
namespace steer
{
DataProcessorSpec getSimReaderSpec(SubspecRange range, const std::vector<std::string>& simprefixes, const std::vector<int>& tpcsectors)
{
  uint64_t activeSectors = 0;
  for (const auto& tpcsector : tpcsectors) {
    activeSectors |= (uint64_t)0x1 << tpcsector;
  }

  auto doit = [range, tpcsectors, activeSectors](ProcessingContext& pc) {
    auto& mgr = steer::HitProcessingManager::instance();
    auto eventrecords = mgr.getDigitizationContext().getEventRecords();
    const auto& context = mgr.getDigitizationContext();

    for (auto const& sector : tpcsectors) {
      // Note: the TPC sector header was serving the sector to lane mapping before
      // now the only remaining purpose is to propagate the mask of valid sectors
      // in principle even that is not necessary any more because every sector has
      // a dedicated route bound to the sector numbers as subspecification and the
      // merging can be done based on that. However if we in the future go over to
      // a multipart scheme again, we again will need the information, so we keep it
      // For the moment, sector member in the sector header is in sync with
      // subspecification of the route
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      pc.outputs().snapshot(OutputRef{"collisioncontext", static_cast<SubSpecificationType>(sector), {header}},
                            context);
    }

    // the first 36 channel numbers are reserved for the TPC, now publish the remaining
    // channels
    for (int subchannel = range.min; subchannel < range.max; ++subchannel) {
      LOG(INFO) << "SENDING SOMETHING TO OTHERS";
      pc.outputs().snapshot(
        OutputRef{"collisioncontext", static_cast<SubSpecificationType>(subchannel)},
        context);
    }

    // digitizer workflow runs only once
    // send endOfData control event and mark the reader as ready to finish
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  };

  // init function return a lambda taking a ProcessingContext
  auto initIt = [simprefixes, doit](InitContext& ctx) {
    // initialize fundamental objects
    auto& mgr = steer::HitProcessingManager::instance();

    // init gRandom to random start
    // TODO: offer option to set seed
    gRandom->SetSeed(0);

    if (simprefixes.size() == 0) {
      LOG(ERROR) << "No simulation prefix available";
    } else {
      LOG(INFO) << "adding " << simprefixes[0] << "\n";
      mgr.addInputFile(simprefixes[0]);
      for (int part = 1; part < simprefixes.size(); ++part) {
        mgr.addInputSignalFile(simprefixes[part]);
      }
    }

    // do we start from an existing context
    auto incontextstring = ctx.options().get<std::string>("incontext");
    LOG(INFO) << "INCONTEXTSTRING " << incontextstring;
    if (incontextstring.size() > 0) {
      auto success = mgr.setupRunFromExistingContext(incontextstring.c_str());
      if (!success) {
        LOG(FATAL) << "Could not read collision context from " << incontextstring;
      }
    } else {

      auto intRate = ctx.options().get<float>("interactionRate"); // is interaction rate requested?
      if (intRate < 1.f) {
        intRate = 1.f;
      }
      LOG(INFO) << "Imposing hadronic interaction rate " << intRate << "Hz";
      mgr.getInteractionSampler().setInteractionRate(intRate);
      o2::raw::HBFUtils::Instance().print();
      o2::raw::HBFUtils::Instance().checkConsistency();
      mgr.getInteractionSampler().setFirstIR({0, o2::raw::HBFUtils::Instance().orbitFirstSampled});
      mgr.getDigitizationContext().setFirstOrbitForSampling(o2::raw::HBFUtils::Instance().orbitFirstSampled);

      auto bcPatternFile = ctx.options().get<std::string>("bcPatternFile");
      if (!bcPatternFile.empty()) {
        mgr.getInteractionSampler().setBunchFilling(bcPatternFile);
      }

      mgr.getInteractionSampler().init();
      mgr.getInteractionSampler().print();
      // doing a random event selection/subsampling?
      mgr.setRandomEventSequence(ctx.options().get<int>("randomsample") > 0);

      // finalize collisions (with number of collisions asked)
      auto col = ctx.options().get<int>("ncollisions");
      if (col != 0) {
        mgr.setupRun(col);
      } else {
        mgr.setupRun();
      }

      // --- we add QED contributions to the digitization context
      // --- for now in between first and last real collision
      auto qedprefix = ctx.options().get<std::string>("simPrefixQED");
      if (qedprefix.size() > 0) {
        o2::steer::InteractionSampler qedInteractionSampler;
        if (!bcPatternFile.empty()) {
          qedInteractionSampler.setBunchFilling(bcPatternFile);
        }

        // get first and last "hadronic" interaction records and let
        // QED events range from the first bunch crossing to the last bunch crossing
        // in this range
        auto first = mgr.getDigitizationContext().getEventRecords().front();
        auto last = mgr.getDigitizationContext().getEventRecords().back();
        first.bc = 0;
        last.bc = o2::constants::lhc::LHCMaxBunches;

        const float ratio = ctx.options().get<float>("qed-x-section-ratio");
        if (ratio <= 0.) {
          throw std::runtime_error("no meaningful qed-x-section-ratio was provided");
        }
        const float hadronicrate = ctx.options().get<float>("interactionRate");
        const float qedrate = ratio * hadronicrate;
        LOG(INFO) << "QED RATE " << qedrate;
        qedInteractionSampler.setInteractionRate(qedrate);
        qedInteractionSampler.setFirstIR(first);
        qedInteractionSampler.init();
        qedInteractionSampler.print();
        std::vector<o2::InteractionTimeRecord> qedinteractionrecords;
        o2::InteractionTimeRecord t;
        LOG(INFO) << "GENERATING COL TIMES";
        t = qedInteractionSampler.generateCollisionTime();
        while ((t = qedInteractionSampler.generateCollisionTime()) < last) {
          qedinteractionrecords.push_back(t);
        }
        LOG(INFO) << "DONE GENERATING COL TIMES";

        // get digitization context and add QED stuff
        mgr.getDigitizationContext().fillQED(qedprefix, qedinteractionrecords);
        mgr.getDigitizationContext().printCollisionSummary(true);
      }
      // --- end addition of QED contributions

      LOG(INFO) << "Initializing Spec ... have " << mgr.getDigitizationContext().getEventRecords().size() << " times ";
      LOG(INFO) << "Serializing Context for later reuse";
      mgr.writeDigitizationContext(ctx.options().get<std::string>("outcontext").c_str());
    }

    return doit;
  };

  std::vector<OutputSpec> outputs;
  for (auto const& tpcsector : tpcsectors) {
    outputs.emplace_back(
      OutputSpec{{"collisioncontext"}, "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(tpcsector), Lifetime::Timeframe});
  }
  for (int subchannel = range.min; subchannel < range.max; ++subchannel) {
    outputs.emplace_back(
      OutputSpec{{"collisioncontext"}, "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(subchannel), Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    /*ID*/ "SimReader",
    /*INPUT CHANNELS*/ Inputs{}, outputs,
    /* ALGORITHM */
    AlgorithmSpec{initIt},
    /* OPTIONS */
    Options{
      {"interactionRate", VariantType::Float, 50000.0f, {"Total hadronic interaction rate (Hz)"}},
      {"bcPatternFile", VariantType::String, "", {"Interacting BC pattern file (e.g. from CreateBCPattern.C)"}},
      {"simPrefixQED", VariantType::String, "", {"Sim (QED) input prefix (example: path/o2qed). The prefix allows to find files like path/o2qed_Kine.root etc."}},
      {"qed-x-section-ratio", VariantType::Float, -1.f, {"Ratio of cross sections QED/hadronic events. Determines QED interaction rate from hadronic interaction rate."}},
      {"outcontext", VariantType::String, "collisioncontext.root", {"Output file for collision context"}},
      {"incontext", VariantType::String, "", {"Take collision context from this file"}},
      {"ncollisions,n",
       VariantType::Int,
       0,
       {"number of collisions to sample (default is given by number of entries in chain"}},
      {"randomsample", VariantType::Int, 0, {"Draw collisions random instead of linear sequence. (Default no = 0)"}}}};
}
} // namespace steer
} // namespace o2
