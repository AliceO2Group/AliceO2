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
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/runDataProcessing.h"
#include "Framework/DataAllocator.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ControlService.h"
#include "DPLUtils/RootTreeReader.h"
#include "Headers/DataHeader.h"
#include "Headers/NameHeader.h"
#include "../../Core/test/TestClasses.h"
#include "Framework/Logger.h"
#include <TSystem.h>
#include <TTree.h>
#include <TFile.h>
#include <vector>

using namespace o2::framework;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(ERROR) << R"(Test condition ")" #condition R"(" failed)"; \
  }

const int gTreeSize = 10; // elements in the test tree
DataProcessorSpec getSourceSpec()
{
  auto initFct = [](InitContext& ic) {
    // create a test tree in a temporary file
    std::string fileName = gSystem->TempDirectory();
    fileName += "/test_RootTreeReader.root";

    {
      std::unique_ptr<TFile> testFile(TFile::Open(fileName.c_str(), "RECREATE"));
      std::unique_ptr<TTree> testTree = std::make_unique<TTree>("testtree", "testtree");

      std::vector<o2::test::TriviallyCopyable> msgblarray;
      std::vector<o2::test::Polymorphic> valarray;
      auto* branch1 = testTree->Branch("msgblarray", &msgblarray);
      auto* branch2 = testTree->Branch("dataarray", &valarray);

      for (int entry = 0; entry < gTreeSize; entry++) {
        msgblarray.clear();
        valarray.clear();
        for (int idx = 0; idx < entry + 1; ++idx) {
          msgblarray.emplace_back((entry * 10) + idx, 0, 0);
          valarray.emplace_back((entry * 10) + idx);
        }
        testTree->Fill();
      }
      testTree->Write();
      testTree->SetDirectory(nullptr);
      testFile->Close();
    }

    constexpr auto persistency = Lifetime::Transient;
    auto reader = std::make_shared<RootTreeReader>("testtree",       // tree name
                                                   fileName.c_str(), // input file name
                                                   RootTreeReader::BranchDefinition<std::vector<o2::test::TriviallyCopyable>>{Output{"TST", "ARRAYOFMSGBL", 0, persistency}, "msgblarray"},
                                                   Output{"TST", "ARRAYOFDATA", 0, persistency},
                                                   "dataarray",
                                                   RootTreeReader::PublishingMode::Single);

    auto processingFct = [reader](ProcessingContext& pc) {
      if (reader->getCount() >= gTreeSize) {
        return;
      }
      if (reader->getCount() == 0) {
        // add two additional headers on the stack in the first entry
        o2::header::NameHeader<16> auxHeader("extended_info");
        o2::header::DataHeader dummyheader;
        (++(*reader))(pc, auxHeader, dummyheader);
      } else {
        // test signature without headers for the rest of the entries
        (++(*reader))(pc);
      }
      if ((reader->getCount()) >= gTreeSize) {
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      }
    };

    return processingFct;
  };

  return DataProcessorSpec{"source", // name of the processor
                           {},
                           {OutputSpec{"TST", "ARRAYOFDATA"},
                            OutputSpec{"TST", "ARRAYOFMSGBL"}},
                           AlgorithmSpec(initFct)};
}

DataProcessorSpec getSinkSpec()
{
  auto processingFct = [](ProcessingContext& pc) {
    static int counter = 0;
    using DataHeader = o2::header::DataHeader;
    for (auto& input : pc.inputs()) {
      auto dh = o2::header::get<const DataHeader*>(input.header);
      LOG(INFO) << dh->dataOrigin.str << " " << dh->dataDescription.str << " " << dh->payloadSize;
    }
    auto data = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input1");
    if (counter == 0) {
      // the first entry comes together with additional headers on the stack, test those ...
      auto auxHeader = DataRefUtils::getHeader<o2::header::NameHeader<16>*>(pc.inputs().get("input1"));
      ASSERT_ERROR(auxHeader != nullptr);
      if (auxHeader != nullptr) {
        ASSERT_ERROR(std::string("extended_info") == auxHeader->getName());
        o2::header::hexDump("", auxHeader, auxHeader->headerSize);
        auto dummyheader = auxHeader->next();
        ASSERT_ERROR(dummyheader != nullptr && dummyheader->size() == 80);
      }
    }

    LOG(INFO) << "count: " << counter << "  data elements:" << data.size();
    ASSERT_ERROR(counter + 1 == data.size());

    // retrieving the unserialized message as vector
    auto msgblvec = pc.inputs().get<std::vector<o2::test::TriviallyCopyable>*>("input2");
    ASSERT_ERROR(counter + 1 == msgblvec->size());

    // retrieving the unserialized message as span
    auto msgblspan = pc.inputs().get<gsl::span<o2::test::TriviallyCopyable>>("input2");
    ASSERT_ERROR(counter + 1 == msgblspan.size());

    for (unsigned int idx = 0; idx < data.size(); idx++) {
      LOG(INFO) << data[idx].get();
      auto expected = 10 * counter + idx;
      ASSERT_ERROR(data[idx].get() == expected);
      ASSERT_ERROR(((*msgblvec)[idx] == o2::test::TriviallyCopyable{expected, 0, 0}));
      ASSERT_ERROR((msgblspan[idx] == o2::test::TriviallyCopyable{expected, 0, 0}));
    }

    ++counter;
  };

  return DataProcessorSpec{"sink", // name of the processor
                           {InputSpec{"input1", "TST", "ARRAYOFDATA"},
                            InputSpec{"input2", "TST", "ARRAYOFMSGBL"}},
                           Outputs{},
                           AlgorithmSpec(processingFct)};
}

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    getSourceSpec(),
    getSinkSpec()};
}
