// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/RootSerializationSupport.h"
#include "Framework/runDataProcessing.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Logger.h"
#include <iostream>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <TObjString.h>

using namespace o2::framework;

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return {{"A1",
           Inputs{},
           Outputs{
             OutputSpec{{"out"}, "TST", "OUT", 0}},
           AlgorithmSpec{
             [](ProcessingContext& ctx) {
               ctx.outputs().make<TObjString>(OutputRef{"out", 0}, "abc");
               ctx.services().get<ControlService>().endOfStream();
               ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
             }}},
          {"A2",
           Inputs{},
           Outputs{
             OutputSpec{{"out"}, "TST", "OUT", 1}},
           AlgorithmSpec{
             [](ProcessingContext& ctx) {
               ctx.outputs().make<TObjString>(OutputRef{"out", 1}, "def");
               ctx.services().get<ControlService>().endOfStream();
               ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
             }}},
          {"B",
           Inputs{InputSpec{"in", {"TST", "OUT"}}},
           Outputs{},
           AlgorithmSpec{adaptStateful([](CallbackService& callbacks) {
             callbacks.set(CallbackService::Id::EndOfStream, [](EndOfStreamContext& context) {
               context.services().get<ControlService>().readyToQuit(QuitRequest::All);
             });
             return adaptStateless([](InputRecord& inputs) {
               auto s = inputs.get<TObjString*>("in");
               LOG(INFO) << "String is " << s->GetString().Data();
             }); })}}};
}
