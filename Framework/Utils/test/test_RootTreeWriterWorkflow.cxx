// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/runDataProcessing.h"
#include "Framework/DataAllocator.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Utils/RootTreeWriter.h"
#include "Utils/MakeRootTreeWriterSpec.h"
#include "Headers/DataHeader.h"
#include "../../Core/test/TestClasses.h"
#include "FairMQLogger.h"
#include <TSystem.h>
#include <TTree.h>
#include <TFile.h>
#include <vector>

using namespace o2::framework;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(ERROR) << R"(Test condition ")" #condition R"(" failed)"; \
  }

constexpr int kTreeSize = 10; // elemants to send and write
DataProcessorSpec getSourceSpec()
{
  auto initFct = [](InitContext& ic) {
    auto counter = std::make_shared<int>();
    *counter = 0;

    auto processingFct = [counter](ProcessingContext& pc) {
      if (*counter >= kTreeSize) {
        // don't publish more
        return;
      }
      o2::test::Polymorphic a(*counter);
      pc.outputs().snapshot(OutputRef{ "output" }, a);
      int& metadata = pc.outputs().make<int>(Output{ "TST", "METADATA", 0, Lifetime::Timeframe });
      metadata = *counter;
      *counter = *counter + 1;
    };

    return processingFct;
  };
  return DataProcessorSpec{ "source", // name of the processor
                            {},
                            { OutputSpec{ { "output" }, "TST", "SOMEOBJECT", 0, Lifetime::Timeframe },
                              OutputSpec{ { "meta" }, "TST", "METADATA", 0, Lifetime::Timeframe } },
                            AlgorithmSpec(initFct) };
}

// this is the direct definition of the processor spec, can be generated from
// MakeRootTreeWriterSpec as shown below
DataProcessorSpec getSinkSpec()
{
  auto initFct = [](InitContext& ic) {
    std::string fileName = gSystem->TempDirectory();
    fileName += "/test_RootTreeWriter.root";
    using WriterType = RootTreeWriter<o2::test::Polymorphic, int>;
    auto writer = std::make_shared<WriterType>(fileName.c_str(),      // output file name
                                               "testtree",            // tree name
                                               "input", "polyobject", // input key and branch name
                                               "meta", "counter"      // input key and branch name
                                               );
    auto counter = std::make_shared<int>();
    *counter = 0;

    // the callback to be set as hook at stop of processing for the framework
    auto finishWriting = [writer]() { writer->close(); };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishWriting);

    auto processingFct = [writer, counter](ProcessingContext& pc) {
      (*writer)(pc);
      *counter = *counter + 1;
      if (*counter >= kTreeSize) {
        pc.services().get<ControlService>().readyToQuit(true);
      }
    };

    return processingFct;
  };

  return DataProcessorSpec{ "sink",
                            { InputSpec{ "input", "TST", "SOMEOBJECT" }, //
                              InputSpec{ "meta", "TST", "METADATA" } },  //
                            Outputs{},
                            AlgorithmSpec(initFct) };
}

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  std::string fileName = gSystem->TempDirectory();
  fileName += "/test_RootTreeWriter.root";

  return WorkflowSpec{
    getSourceSpec(),
    MakeRootTreeWriterSpec<o2::test::Polymorphic, int>         // type setup
    (                                                          //
      "sink",                                                  // process name
      fileName.c_str(),                                        // default file name
      "testtree",                                              // default tree name
      1,                                                       // default number of events
      InputSpec{ "input", "TST", "SOMEOBJECT" }, "polyobject", // input and branch config
      InputSpec{ "meta", "TST", "METADATA" }, "counter"        // input and branch config
      )()                                                      // call the generator
  };
}
