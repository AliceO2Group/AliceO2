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
#include "DPLUtils/RootTreeWriter.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Headers/DataHeader.h"
#include "../../Core/test/TestClasses.h"
#include "FairMQLogger.h"
#include <TTree.h>
#include <TFile.h>
#include <vector>
#include <stdexcept>
// note: std filesystem is first supported in gcc 8
#include <boost/filesystem.hpp>

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
    for (auto const& check : mChecks) {
      TFile* file = TFile::Open(check.first.c_str());
      if (file == nullptr) {
        setError(std::string("missing file ") + check.first.c_str());
        continue;
      }
      TTree* tree = reinterpret_cast<TTree*>(file->GetObjectChecked("testtree", "TTree"));
      if (tree) {
        if (tree->GetEntries() != check.second) {
          setError(std::string("inconsistent number of entries in 'testtree' of file ") + check.first.c_str() + " expecting " + std::to_string(check.second) + " got " + std::to_string(tree->GetEntries()));
        }
      } else {
        setError(std::string("can not find tree 'testtree' in file ") + check.first.c_str());
      }
      file->Close();
      // TODO: keeping this as a note because this is not understood
      // files should get deleted at this point, but if this line is in, the files are already deleted when entering
      // the loop. Not clear if there is some out-of-order execution
      // could not test with std filesystem, because the is first supported in gcc 8
      //boost::filesystem::remove(check.first.c_str());
    }
    if (mErrorMessage.empty() == false) {
      throw std::runtime_error(mErrorMessage);
    }
  }

  void addCheck(std::string filename, int entries)
  {
    mChecks.emplace_back(filename, entries);
  }

  template <typename T>
  void setError(T const& message)
  {
    if (mErrorMessage.empty()) {
      mErrorMessage = message;
    }
  }

 private:
  std::vector<std::pair<std::string, int>> mChecks;
  std::string mErrorMessage;
};
static StaticChecker sChecker;

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

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  std::string fileName = boost::filesystem::temp_directory_path().string();
  fileName += "/test_RootTreeWriter";
  std::string altFileName = fileName + "_alt.root";
  fileName += ".root";

  sChecker.addCheck(fileName, 1);
  sChecker.addCheck(altFileName, sTreeSize);

  auto checkReady = [counter = std::make_shared<int>(0)](auto) -> bool {
    *counter = *counter + 1;
    // processing function checks the callback for two inputs -> factor 2
    // the two last calls have to return true to signal on both inputs
    return (*counter + 1) >= (2 * sTreeSize);
  };

  using Polymorphic = o2::test::Polymorphic;
  return WorkflowSpec{
    getSourceSpec(),
    MakeRootTreeWriterSpec                                                                      //
    (                                                                                           //
      "sink1",                                                                                  // process name
      fileName.c_str(),                                                                         // default file name
      "testtree",                                                                               // default tree name
      1,                                                                                        // default number of events
      BranchDefinition<Polymorphic>{ InputSpec{ "input", "TST", "SOMEOBJECT" }, "polyobject" }, // branch config
      BranchDefinition<int>{ InputSpec{ "meta", "TST", "METADATA" }, "counter" }                // branch config
      )(),                                                                                      // call the generator
    MakeRootTreeWriterSpec                                                                      //
    (                                                                                           //
      "sink2",                                                                                  // process name
      altFileName.c_str(),                                                                      // default file name
      "testtree",                                                                               // default tree name
      MakeRootTreeWriterSpec::TerminationPolicy::Workflow,                                      // terminate the workflow
      MakeRootTreeWriterSpec::TerminationCondition{ checkReady },                               // custom termination condition
      BranchDefinition<Polymorphic>{ InputSpec{ "input", "TST", "SOMEOBJECT" }, "polyobject" }, // branch config
      BranchDefinition<int>{ InputSpec{ "meta", "TST", "METADATA" }, "counter" }                // branch config
      )()                                                                                       // call the generator
  };
}
