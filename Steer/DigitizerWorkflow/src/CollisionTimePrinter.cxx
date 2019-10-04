// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CollisionTimePrinter.h"
#include <Steer/InteractionSampler.h>
#include "Headers/DataHeader.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Headers/DataHeader.h"
#include "Framework/Lifetime.h"
#include "Framework/ControlService.h"
#include "Steer/HitProcessingManager.h"
#include <FairMQLogger.h>
#include <TMessage.h> // object serialization
#include <memory>     // std::unique_ptr
#include <cstring>    // memcpy
#include <string>     // std::string

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
namespace o2
{
namespace steer
{

// a very simple DataProcessor consuming data sent
// from the SimReader
// mainly here for debugging/demonstration
DataProcessorSpec getCollisionTimePrinter(int channel)
{
  // set up the processing function

  // init function return a lambda taking a ProcessingContext
  auto doIt = [](ProcessingContext& pc) {
    std::cout << " ######### ";

    // access data
    auto dataref = pc.inputs().get("input");
    auto header = o2::header::get<const o2::header::DataHeader*>(dataref.header);
    LOG(INFO) << "PAYLOAD SIZE " << header->payloadSize;

    const auto context = pc.inputs().get<o2::steer::RunContext*>("input");
    //    auto view = DataRefUtils::as<o2::MCInteractionRecord>(dataref);
    auto view = context->getEventRecords();
    LOG(INFO) << "GOT " << view.size() << "times";
    int counter = 0;
    for (auto& collrecord : view) {
      LOG(INFO) << "TIME " << counter++ << " : " << collrecord.timeNS;
    }
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  };

  return DataProcessorSpec{/*ID*/ "CollTimePrinter",
                           /*INPUT CHANNELS*/ Inputs{InputSpec{"input", "SIM", "EVENTTIMES", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
                           /*OUTPUT CHANNELS*/ Outputs{},
                           /* ALGORITHM */
                           AlgorithmSpec(doIt),
                           /* OPTIONS */
                           Options{}};
}
} // namespace steer
} // namespace o2
