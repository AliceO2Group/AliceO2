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
#include <boost/algorithm/string.hpp>

#include "Framework/InputSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ParallelContext.h"
#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

#include <chrono>
#include <thread>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    {"source",
     Inputs{},
     {
       OutputSpec{{"test"}, "TST", "A"},
     },
     AlgorithmSpec{
       [](ProcessingContext& ctx) {
         auto& out = ctx.outputs().make<int>(OutputRef{"test", 0});
         ctx.services().get<ControlService>().endOfStream();
         ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
       }}},
    {"dest",
     Inputs{
       {"test", "TST", "A"}},
     Outputs{},
     AlgorithmSpec{
       [](InitContext& ic) {
         auto count = std::make_shared<int>(0);
         auto callback = [count](fair::mq::RegionInfo const&) {
           LOG(info) << "once";
           (*count)++;
         };
         fair::mq::RegionInfo dummy;
         ic.services().get<CallbackService>().set<CallbackService::Id::RegionInfoCallback>(callback);
         return [count](ProcessingContext& ctx) {
           if (*count >= 1) {
             ctx.services().get<ControlService>().readyToQuit(QuitRequest::All);
           }
         };
       }}}};
}
