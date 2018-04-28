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
#include <FairMQLogger.h>
#include <TMessage.h> // object serialization
#include <memory>     // std::unique_ptr
#include <cstring>    // memcpy
#include <string>     // std::string
#include <cassert>
#include <chrono>
#include <thread>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
namespace o2
{
namespace steer
{
DataProcessorSpec getSimReaderSpec(int fanoutsize)
{
  auto doit = [fanoutsize](ProcessingContext& pc) {
    auto& mgr = steer::HitProcessingManager::instance();
    auto eventrecords = mgr.getRunContext().getEventRecords();
    const auto& context = mgr.getRunContext();

    // counter to make sure we are sending the data only once
    static int counter = 0;
    if (counter++ == 0) {
      // sleep(10);
      LOG(INFO) << "SENDING " << eventrecords.size() << " records";

      for (int subchannel = 0; subchannel < fanoutsize; ++subchannel) {
        pc.outputs().snapshot(
          Output{ "SIM", "EVENTTIMES", static_cast<SubSpecificationType>(subchannel), Lifetime::Timeframe }, context);
      }

      // do this only one
      pc.services().get<ControlService>().readyToQuit(false);
    }
  };

  // init function return a lambda taking a ProcessingContext
  auto initIt = [doit](InitContext& ctx) {
    // initialize fundamental objects
    auto& mgr = steer::HitProcessingManager::instance();
    mgr.addInputFile(ctx.options().get<std::string>("simFile").c_str());
    mgr.setupRun();

    LOG(INFO) << "Initializing Spec ... have " << mgr.getRunContext().getEventRecords().size() << " times ";
    return doit;
  };

  std::vector<OutputSpec> outputs;
  for (int subchannel = 0; subchannel < fanoutsize; ++subchannel) {
    outputs.emplace_back(
      OutputSpec{ "SIM", "EVENTTIMES", static_cast<SubSpecificationType>(subchannel), Lifetime::Timeframe });
  }

  return DataProcessorSpec{ /*ID*/ "SimReader",
                            /*INPUT CHANNELS*/ Inputs{}, outputs,
                            /* ALGORITHM */
                            AlgorithmSpec{ initIt },
                            /* OPTIONS */
                            Options{ { "simFile", VariantType::String, "o2sim.root", { "Sim input filename" } } } };
}
}
}
