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
#include "Framework/RootMessageContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataAllocator.h"
#include "Framework/InputRecord.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/CustomWorkflowTerminationHook.h"
#include "Framework/DataRefUtils.h"
#include "DPLUtils/RootTreeWriter.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"
#include "../../Core/test/TestClasses.h"
#include "Framework/Logger.h"
#include <TSystem.h>
#include <TTree.h>
#include <TFile.h>
#include <vector>
#include <stdexcept>
#include <iostream>
// note: std filesystem is first supported in gcc 8
#include <filesystem>

using namespace o2::framework;

// a helper class to do the checking at the end of the program when
// the destructor of the class is called.
class StaticChecker
{
 public:
  StaticChecker() = default;
  // have to throw an exception in the destructor if checking fails
  // this is ok in this cae because no other instances which would require proper
  // cleanup are expected
  ~StaticChecker() noexcept(false)
  {
    // the check in the desctructor makes sure that the workflow has been run at all
    if (mChecks.size() > 0) {
      throw std::runtime_error("Workflow error: Checks have not been executed");
    }
  }

  struct Attributes {
    std::string fileName;
    int nEntries = 0;
    int nBranches = 0;
  };

  void runChecks()
  {
    for (auto const& check : mChecks) {
      TFile* file = TFile::Open(check.fileName.c_str());
      if (file == nullptr) {
        setError(std::string("missing file ") + check.fileName.c_str());
        continue;
      }
      TTree* tree = reinterpret_cast<TTree*>(file->GetObjectChecked("testtree", "TTree"));
      if (tree == nullptr) {
        setError(std::string("can not find tree 'testtree' in file ") + check.fileName.c_str());
      } else if (tree->GetEntries() != check.nEntries) {
        setError(std::string("inconsistent number of entries in 'testtree' of file ") + check.fileName + " expecting " + std::to_string(check.nEntries) + " got " + std::to_string(tree->GetEntries()));
      } else if (tree->GetNbranches() != check.nBranches) {
        setError(std::string("inconsistent number of branches in 'testtree' of file ") + check.fileName + " expecting " + std::to_string(check.nBranches) + " got " + std::to_string(tree->GetNbranches()));
      }
      file->Close();
      std::filesystem::remove(check.fileName.c_str());
    }
    mChecks.clear();
    if (mErrorMessage.empty() == false) {
      throw std::runtime_error(mErrorMessage);
    }
  }

  void addCheck(std::string filename, int entries, int branches = 0)
  {
    mChecks.emplace_back(Attributes{filename, entries, branches});
  }

  template <typename T>
  void setError(T const& message)
  {
    if (mErrorMessage.empty()) {
      mErrorMessage = message;
    }
  }

  void clear()
  {
    mChecks.clear();
    mErrorMessage.clear();
  }

 private:
  std::vector<Attributes> mChecks;
  std::string mErrorMessage;
};
static StaticChecker sChecker;

void customize(o2::framework::OnWorkflowTerminationHook& hook)
{
  hook = [](const char* idstring) {
    // run the checks in the master driver process, all the individual
    // processes have the same checker setup, so this needs to be cleared for child
    // (device) processes.
    if (idstring == nullptr) {
      sChecker.runChecks();
    } else {
      sChecker.clear();
    }
  };
}

#include "Framework/runDataProcessing.h"

static constexpr int sTreeSize = 10; // elements to send and write
DataProcessorSpec getSourceSpec()
{
  auto initFct = [](InitContext& ic) {
    auto counter = std::make_shared<int>();
    *counter = 0;

    auto processingFct = [counter](ProcessingContext& pc) {
      if (*counter >= sTreeSize) {
        // don't publish more
        return;
      }
      o2::test::Polymorphic a(*counter);
      pc.outputs().snapshot(OutputRef{"output", 0}, a);
      pc.outputs().snapshot(OutputRef{"output", 1}, a);
      int& metadata = pc.outputs().make<int>(Output{"TST", "METADATA", 0});
      metadata = *counter;
      *counter = *counter + 1;
      if (*counter >= sTreeSize) {
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      }
    };

    return processingFct;
  };
  return DataProcessorSpec{"source", // name of the processor
                           {},
                           {OutputSpec{{"output"}, "TST", "SOMEOBJECT", 0, Lifetime::Timeframe},
                            OutputSpec{{"output"}, "TST", "SOMEOBJECT", 1, Lifetime::Timeframe},
                            OutputSpec{{"meta"}, "TST", "METADATA", 0, Lifetime::Timeframe}},
                           AlgorithmSpec(initFct)};
}

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  std::string fileName = std::filesystem::temp_directory_path().string();
  fileName += "/test_RootTreeWriter";
  std::string altFileName = fileName + "_alt.root";
  fileName += ".root";

  // the first writer is configured with number of events (1)
  // it receives two routes and saves those to two branches
  // a third route is disabled (and not published by the source)
  sChecker.addCheck(fileName, 1, 2);
  // the second writer uses a check function to determine when its ready
  // the first definition creates two branches, input data comes in over the
  // same route, together with the second definition its three branches
  sChecker.addCheck(altFileName, sTreeSize, 3);

  auto checkReady = [counter = std::make_shared<int>(0)](auto) -> bool {
    *counter = *counter + 1;
    // processing function checks the callback for two inputs -> factor 2
    // the two last calls have to return true to signal on both inputs
    return (*counter + 1) >= (2 * sTreeSize);
  };

  auto getIndex = [](auto const& ref) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    return static_cast<size_t>(dh->subSpecification);
  };
  auto getName = [](std::string base, size_t index) {
    return base + "_" + std::to_string(index);
  };

  auto preprocessor = [](ProcessingContext& ctx) {
    for (auto const& ref : InputRecordWalker(ctx.inputs())) {
      auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      LOGP(info, "got data: {}/{}/{}", dh->dataOrigin, dh->dataDescription, dh->subSpecification);
    }
  };

  using Polymorphic = o2::test::Polymorphic;
  return WorkflowSpec{
    getSourceSpec(),
    MakeRootTreeWriterSpec                                                                  //
    (                                                                                       //
      "sink1",                                                                              // process name
      fileName.c_str(),                                                                     // default file name
      MakeRootTreeWriterSpec::TreeAttributes{"testtree", "what a naive test"},              // default tree name
      1,                                                                                    // default number of events
      BranchDefinition<Polymorphic>{InputSpec{"in", "TST", "SOMEOBJECT", 0}, "polyobject"}, // branch config
      BranchDefinition<int>{InputSpec{"disabl", "TST", "NODATA"}, "dummy", 0},              // disabled branch config
      BranchDefinition<int>{InputSpec{"meta", "TST", "METADATA"}, "counter"}                // branch config
      )(),                                                                                  // call the generator
    MakeRootTreeWriterSpec                                                                  //
    (                                                                                       //
      "sink2",                                                                              // process name
      altFileName.c_str(),                                                                  // default file name
      "testtree",                                                                           // default tree name
      MakeRootTreeWriterSpec::TerminationPolicy::Workflow,                                  // terminate the workflow
      MakeRootTreeWriterSpec::TerminationCondition{checkReady},                             // custom termination condition
      MakeRootTreeWriterSpec::Preprocessor{preprocessor},                                   // custom preprocessor
      BranchDefinition<Polymorphic>{
        InputSpec{"input",                                                   // key
                  ConcreteDataTypeMatcher{"TST", "SOMEOBJECT"}},             // subspec independent
        "polyobject",                                                        // base name of branch
        "",                                                                  // empty option
        2,                                                                   // two branches
        RootTreeWriter::IndexExtractor(getIndex),                            // index retriever
        RootTreeWriter::BranchNameMapper(getName)                            // branch name retriever
      },                                                                     //
      BranchDefinition<int>{InputSpec{"meta", "TST", "METADATA"}, "counter"} // branch config
      )()                                                                    // call the generator
  };
}
