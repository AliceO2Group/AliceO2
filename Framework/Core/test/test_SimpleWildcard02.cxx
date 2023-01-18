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

#include "Framework/RootSerializationSupport.h"
#include "Framework/runDataProcessing.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/RootMessageContext.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Logger.h"
#include "Framework/DataRefUtils.h"
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
             OutputSpec{{"out"}, {"TST", "OUT"}}},
           AlgorithmSpec{
             [](InitContext& initCtx) {
             auto count = std::make_shared<unsigned int>(0);
             return [count](ProcessingContext& ctx) {
               TObjString s("abc");
               for (int i = 0; i < 2; i++) {
                 ctx.outputs().snapshot(OutputRef{"out", *count}, s);
                 if (*count > 10) {
                   ctx.services().get<ControlService>().endOfStream();
                   ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
                 }
                 (*count)++;
               }
             }; }}},
          {"B",
           Inputs{InputSpec{"in", ConcreteDataTypeMatcher{"TST", "OUT"}}},
           Outputs{},
           AlgorithmSpec{adaptStateful([](CallbackService& callbacks) {
             callbacks.set(CallbackService::Id::EndOfStream, [](EndOfStreamContext& context) {
               context.services().get<ControlService>().readyToQuit(QuitRequest::All);
             });
             return adaptStateless([](InputRecord& inputs) {
               auto s = inputs.get<TObjString*>("in");
               auto n = inputs.getNofParts(0);
               LOG(info) << "Number of parts " << inputs.getNofParts(0);
               if (n != 2) {
                 LOG(error) << "Bad number of parts" << inputs.getNofParts(0);
               }
               for (size_t i = 0; i < n; ++i) {
                 auto ref = inputs.getByPos(0, i);
                 auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
                 LOG(info) << "String is " << s->GetString().Data() << " " << dh->subSpecification;
               }
             }); })}}};
}
