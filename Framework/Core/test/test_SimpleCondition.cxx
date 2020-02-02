// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRefUtils.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include <Monitoring/Monitoring.h>
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/Logger.h"

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

/// This shows how to get a condition for the
/// origin "TES" and the description "STRING".
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    DataProcessorSpec{
      "condition_consumer", //
      {
        InputSpec{"condition", "TES", "STRING", 0, Lifetime::Condition}, //
      },                                                                 //
      Outputs{},                                                         //
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          auto s = ctx.inputs().get<std::string>("condition");

          if (s != "Hello") {
            LOG(ERROR) << "Expecting `Hello', found `" << s << "'";
          } else {
            LOG(INFO) << "Everything OK";
          }
          ctx.services().get<ControlService>().readyToQuit(QuitRequest::All);
        } //
      }   //
    }     //
  };
}
