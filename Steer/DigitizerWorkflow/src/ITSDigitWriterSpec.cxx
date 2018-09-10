// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Processor spec for a ROOT file writer for ITS digits

#include "ITSDigitWriterSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "ITSMFTBase/Digit.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <TTree.h>
#include <TBranch.h>
#include <TFile.h>
#include <memory> // for make_shared, make_unique, unique_ptr
#include <vector>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace ITS
{

template <typename T>
TBranch* getOrMakeBranch(TTree& tree, std::string brname, T* ptr)
{
  if (auto br = tree.GetBranch(brname.c_str())) {
    br->SetAddress(static_cast<void*>(&ptr));
    return br;
  }
  // otherwise make it
  return tree.Branch(brname.c_str(), ptr);
}

/// create the processor spec
/// describing a processor receiving digits for ITS writing them to file
DataProcessorSpec getITSDigitWriterSpec()
{
  auto initFunction = [](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("its-digit-outfile");
    auto treename = ic.options().get<std::string>("treename");

    auto outputfile = std::make_shared<TFile>(filename.c_str(), "RECREATE");
    auto outputtree = std::make_shared<TTree>(treename.c_str(), treename.c_str());

    // container for incoming digits // RS TODO: why can't we use raw pointer from the input directly?
    auto digits = std::make_shared<std::vector<o2::ITSMFT::Digit>>();
    //    auto labels = std::make_shared<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

    // the callback to be set as hook at stop of processing for the framework
    auto finishWriting = [outputfile, outputtree]() {
      outputtree->SetEntries(1); //RS ???
      outputtree->Write();
      outputfile->Close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishWriting);

    // setup the processing function
    // using by-copy capture of the worker instance shared pointer
    // the shared pointer makes sure to clean up the instance when the processing
    // function gets out of scope
    auto processingFct = [outputfile, outputtree, digits](ProcessingContext& pc) {
      static bool finished = false;
      if (finished) {
        // avoid being executed again when marked as finished;
        return;
      }

      // retrieve the digits from the input
      auto inDigits = pc.inputs().get<std::vector<o2::ITSMFT::Digit>>("itsdigits");
      auto inLabels = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("itsdigitsMCTR");
      LOG(INFO) << "RECEIVED DIGITS SIZE " << inDigits.size();
      *digits.get() = std::move(inDigits);
      auto labelsRaw = inLabels.get();
      // connect this to a particular branch
      auto brDig = getOrMakeBranch(*outputtree.get(), "ITSDigit", digits.get());
      brDig->Fill();
      auto brLbl = getOrMakeBranch(*outputtree.get(), "ITSDigitMCTruth", &labelsRaw);
      brLbl->Fill();

      finished = true;
      pc.services().get<ControlService>().readyToQuit(false);
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  std::vector<InputSpec> inputs;
  inputs.emplace_back(InputSpec{ "itsdigits", "ITS", "DIGITS", 0, Lifetime::Timeframe });
  inputs.emplace_back(InputSpec{ "itsdigitsMCTR", "ITS", "DIGITSMCTR", 0, Lifetime::Timeframe });
  return DataProcessorSpec{
    "ITSDigitWriter",
    inputs,
    {}, // no output
    AlgorithmSpec(initFunction),
    Options{
      { "its-digit-outfile", VariantType::String, "itsdigits.root", { "Name of the input file" } },
      { "treename", VariantType::String, "o2sim", { "Name of top-level TTree" } },
    }
  };
}
} // end namespace ITS
} // end namespace o2
