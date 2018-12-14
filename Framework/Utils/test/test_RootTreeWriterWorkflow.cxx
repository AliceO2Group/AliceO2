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
    using Polymorphic = o2::test::Polymorphic;
    using WriterType = RootTreeWriter;
    auto writer = std::make_shared<WriterType>(fileName.c_str(), // output file name
                                               "testtree",       // tree name
                                               WriterType::BranchDef<Polymorphic>{ "input", "polyobject" },
                                               WriterType::BranchDef<int>{ "meta", "counter" });
    auto counter = std::make_shared<int>();
    *counter = 0;

    // the callback to be set as hook at stop of processing for the framework
    auto finishWriting = [writer]() { writer->close(); };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishWriting);

    auto processingFct = [writer, counter](ProcessingContext& pc) {
      (*writer)(pc.inputs());
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

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  std::string fileName = gSystem->TempDirectory();
  fileName += "/test_RootTreeWriter.root";

  using Polymorphic = o2::test::Polymorphic;
  return WorkflowSpec{
    getSourceSpec(),
    MakeRootTreeWriterSpec                                                                      //
    (                                                                                           //
      "sink",                                                                                   // process name
      fileName.c_str(),                                                                         // default file name
      "testtree",                                                                               // default tree name
      1,                                                                                        // default number of events
      MakeRootTreeWriterSpec::TerminationPolicy::Workflow,                                      // terminate the workflow
      BranchDefinition<Polymorphic>{ InputSpec{ "input", "TST", "SOMEOBJECT" }, "polyobject" }, // branch config
      BranchDefinition<int>{ InputSpec{ "meta", "TST", "METADATA" }, "counter" }                // branch config
      )()                                                                                       // call the generator
  };
}
