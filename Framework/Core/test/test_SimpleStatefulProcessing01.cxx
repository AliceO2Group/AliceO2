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

// This is a simple consumer / producer workflow where both are
// stateful, i.e. they have context which comes from their initialization.
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    //
    DataProcessorSpec{
      "producer",                                              //
      Inputs{},                                                //
      {OutputSpec{"TES", "STATEFUL", 0, Lifetime::Timeframe}}, //
      // The producer is stateful, we use a static for the state in this
      // particular case, but a Singleton or a captured new object would
      // work as well.
      AlgorithmSpec{
        adaptStateful(
          [](CallbackService& callbacks) {
            static int foo = 0;
            static int step = 0; // incremented in registered callbacks
            auto startcb = []() {
              ++step;
              LOG(INFO) << "start " << step;
            };
            auto stopcb = []() {
              ++step;
              LOG(INFO) << "stop " << step;
            };
            auto resetcb = []() {
              ++step;
              LOG(INFO) << "reset " << step;
            };
            callbacks.set(CallbackService::Id::Start, startcb);
            callbacks.set(CallbackService::Id::Stop, stopcb);
            callbacks.set(CallbackService::Id::Reset, resetcb);
            return adaptStateless([](DataAllocator& outputs) {
              auto& out = outputs.newChunk({"TES", "STATEFUL", 0}, sizeof(int));
              auto outI = reinterpret_cast<int*>(out.data());
              outI[0] = foo++;
            });
          }) //
      }      //
    },       //
    DataProcessorSpec{
      "consumer",                                                     //
      {InputSpec{"test", "TES", "STATEFUL", 0, Lifetime::Timeframe}}, //
      Outputs{},                                                      //
      AlgorithmSpec{
        adaptStateful(
          []() {
            static int expected = 0;
            return adaptStateless([](InputRecord& inputs, ControlService& control) {
              const int* in = reinterpret_cast<const int*>(inputs.get("test").payload);

              if (*in != expected++) {
                LOG(ERROR) << "Expecting " << expected << " found " << *in;
              } else {
                LOG(INFO) << "Everything OK for " << expected << std::endl;
                control.readyToQuit(true);
              }
            });
          }) //
      }      //
    }        //
  };
}
