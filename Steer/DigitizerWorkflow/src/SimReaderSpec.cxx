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
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "Steer/HitProcessingManager.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
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
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
namespace o2
{
namespace steer
{
DataProcessorSpec getSimReaderSpec(int fanoutsize, const std::vector<int>& tpcsectors,
                                   std::shared_ptr<std::vector<int>> tpcsubchannels)
{
  // this container will contain the TPC sector assignment per subchannel per invocation
  // it will allow that we snapshot/send exactly one sector assignment per algorithm invocation
  // to ensure that they all have different timeslice ids
  auto tpcsectormessages = std::make_shared<std::vector<std::vector<int>>>();
  tpcsectormessages->resize(tpcsubchannels->size());
  int tpcchannelcounter = 0;
  uint64_t activeSectors = 0;
  for (const auto& tpcsector : tpcsectors) {
    activeSectors |= (uint64_t)0x1 << tpcsector;
    auto actualchannel = (*tpcsubchannels.get())[tpcchannelcounter % tpcsubchannels->size()];
    LOG(DEBUG) << " WILL ASSIGN SECTOR " << tpcsector << " to subchannel " << actualchannel;
    tpcsectormessages->operator[](actualchannel).emplace_back(tpcsector);
    tpcchannelcounter++;
  }

  // this is the number of invocations of the algorithm needed for the TPC
  size_t tpcinvocations = 0;
  for (int i = 0; i < tpcsubchannels->size(); ++i) {
    tpcinvocations = std::max(tpcinvocations, tpcsectormessages->operator[](i).size());
  }
  // in principle each channel needs to be invoked exactly the same number of times (sigh)
  // so I am adding some kind of "NOP" sectors in case needed
  for (int i = 0; i < tpcsubchannels->size(); ++i) {
    auto size = tpcsectormessages->operator[](i).size();
    if (size < tpcinvocations) {
      for (int k = 0; k < tpcinvocations - size; ++k) {
        tpcsectormessages->operator[](i).emplace_back(-2); // -2 is NOP
        LOG(INFO) << "ADDING NOP TO CHANNEL " << i << "\n";
      }
      assert(tpcsectormessages->operator[](i).size() == tpcinvocations);
    }
  }
  // at this moment all tpc digitizers should receive the exact same number of messages/invocations

  auto doit = [fanoutsize, tpcsectormessages, tpcinvocations, tpcsubchannels, activeSectors](ProcessingContext& pc) {
    auto& mgr = steer::HitProcessingManager::instance();
    auto eventrecords = mgr.getRunContext().getEventRecords();
    const auto& context = mgr.getRunContext();

    // counter to make sure we are sending the data only once
    static int counter = 0;

    static bool finished = false;
    static bool tpc_end_messagesent = false;
    if (finished) {
      if (!tpc_end_messagesent) {
        // we need to send this in a different time slice
        // send message telling tpc workers that they can terminate
        // do this only one
        for (const auto& channel : *tpcsubchannels.get()) {
          // -1 is marker for end of work
          o2::tpc::TPCSectorHeader header{-1};
          header.activeSectors = activeSectors;
          pc.outputs().snapshot(
            OutputRef{"collisioncontext", static_cast<SubSpecificationType>(channel), {header}},
            context);
        }
        tpc_end_messagesent = true;
      }
      // now mark the reader as ready to finish
      pc.services().get<ControlService>().readyToQuit(false);
      return;
    }

    for (int tpcchannel = 0; tpcchannel < tpcsubchannels->size(); ++tpcchannel) {
      auto& sectors = tpcsectormessages->operator[](tpcchannel);
      if (counter < sectors.size()) {
        auto sector = sectors[counter];
        // send the sectorassign as header with the collision context data
        o2::tpc::TPCSectorHeader header{sector};
        header.activeSectors = activeSectors;
        pc.outputs().snapshot(
          OutputRef{"collisioncontext", static_cast<SubSpecificationType>(tpcchannel), {header}},
          context);
      }
    }

    // everything not done previously treat here (this is to be seen how since other things than TPC will have
    // a different number of invocations)
    for (int subchannel = 0; subchannel < fanoutsize; ++subchannel) {
      // TODO: this is temporary ... we should find a more clever+ faster + scalable mechanism
      if (std::find(tpcsubchannels->begin(), tpcsubchannels->end(), subchannel) != tpcsubchannels->end()) {
        continue;
      }
      LOG(INFO) << "SENDING SOMETHING TO OTHERS";
      pc.outputs().snapshot(
        OutputRef{"collisioncontext", static_cast<SubSpecificationType>(subchannel)},
        context);
    }
    counter++;
    if (tpcinvocations == 0 || counter == tpcinvocations) {
      finished = true;
    }
  };

  // init function return a lambda taking a ProcessingContext
  auto initIt = [doit](InitContext& ctx) {
    // initialize fundamental objects
    auto& mgr = steer::HitProcessingManager::instance();

    mgr.addInputFile(ctx.options().get<std::string>("simFile").c_str());
    if (ctx.options().get<std::string>("simFileS").size() > 0) {
      mgr.addInputSignalFile(ctx.options().get<std::string>("simFileS").c_str());
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
      mgr.getInteractionSampler().init();

      // number of collisions asked?
      auto col = ctx.options().get<int>("ncollisions");
      if (col != 0) {
        mgr.setupRun(col);
      } else {
        mgr.setupRun();
      }
      LOG(INFO) << "Initializing Spec ... have " << mgr.getRunContext().getEventRecords().size() << " times ";
      LOG(INFO) << "Serializing Context for later reuse";
      mgr.writeRunContext(ctx.options().get<std::string>("outcontext").c_str());
    }

    return doit;
  };

  std::vector<OutputSpec> outputs;
  for (int subchannel = 0; subchannel < fanoutsize; ++subchannel) {
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
      {"simFile", VariantType::String, "o2sim.root", {"Sim input filename"}},
      {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}},
      {"simFileQED", VariantType::String, "", {"Sim (QED) input filename"}},
      {"outcontext", VariantType::String, "collisioncontext.root", {"Output file for collision context"}},
      {"incontext", VariantType::String, "", {"Take collision context from this file"}},
      {"ncollisions,n",
       VariantType::Int,
       0,
       {"number of collisions to sample (default is given by number of entries in chain"}}}};
}
} // namespace steer
} // namespace o2
