// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Processor spec for a ROOT file writer for TOF digits

#include "TOFWorkflow/TOFDigitWriterSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "TOFBase/Digit.h"
#include <memory> // for make_shared, make_unique, unique_ptr
#include <vector>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tof
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
DataProcessorSpec getTOFDigitWriterSpec(bool useMC)
{
  auto initFunction = [useMC](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("tof-digit-outfile");
    auto treename = ic.options().get<std::string>("treename");

    auto outputfile = std::make_shared<TFile>(filename.c_str(), "RECREATE");
    auto outputtree = std::make_shared<TTree>(treename.c_str(), treename.c_str());

    // the callback to be set as hook at stop of processing for the framework
    auto finishWriting = [outputfile, outputtree]() {
      outputtree->SetEntries(1);
      outputtree->Write();
      outputfile->Close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishWriting);

    // setup the processing function
    // using by-copy capture of the worker instance shared pointer
    // the shared pointer makes sure to clean up the instance when the processing
    // function gets out of scope
    auto processingFct = [outputfile, outputtree, useMC](ProcessingContext& pc) {
      static bool finished = false;
      if (finished) {
        // avoid being executed again when marked as finished;
        return;
      }

      // retrieve the digits from the input
      auto indata = pc.inputs().get<std::vector<o2::tof::Digit>*>("tofdigits");
      LOG(INFO) << "RECEIVED DIGITS SIZE " << indata->size();
      auto row = pc.inputs().get<std::vector<o2::tof::ReadoutWindowData>*>("readoutwin");
      LOG(INFO) << "RECEIVED READOUT WINDOWS " << row->size();

      auto digVect = indata.get();
      auto rowVect = row.get();

      // connect this to a particular branch
      auto brDig = getOrMakeBranch(*outputtree.get(), "TOFDigit", &digVect);
      brDig->Fill();
      auto brRow = getOrMakeBranch(*outputtree.get(), "TOFReadoutWindow", &rowVect);
      brRow->Fill();

      // retrieve labels from the input
      if (useMC) {
        auto labeldata = pc.inputs().get<std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>*>("tofdigitlabels");
        //        for (int i = 0; i < labeldata->size(); i++) {
        //          LOG(INFO) << "TOF GOT " << labeldata->at(i).getNElements() << " LABELS ";
        //        }
        auto labeldataraw = labeldata.get();
        // connect this to a particular branch
        auto labelbr = getOrMakeBranch(*outputtree.get(), "TOFDigitMCTruth", &labeldataraw);
        labelbr->Fill();
      }

      finished = true;
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofdigits", o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("readoutwin", o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe);
  if (useMC)
    inputs.emplace_back("tofdigitlabels", o2::header::gDataOriginTOF, "DIGITSMCTR", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFDigitWriter",
    inputs,
    {}, // no output
    AlgorithmSpec(initFunction),
    Options{
      {"tof-digit-outfile", VariantType::String, "tofdigits.root", {"Name of the input file"}},
      {"treename", VariantType::String, "o2sim", {"Name of top-level TTree"}},
    }};
}
} // end namespace tof
} // end namespace o2
