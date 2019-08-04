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

#include "TOFWorkflow/TOFMatchedWriterSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/SerializationMethods.h"
#include "Headers/DataHeader.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/Cluster.h"
#include <memory> // for make_shared, make_unique, unique_ptr
#include <vector>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tof
{

using evIdx = o2::dataformats::EvIndex<int, int>;
using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;

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
/// describing a processor receiving matching info for TOF writing them to file
/// TODO: make this processor generic and reusable!!
DataProcessorSpec getTOFMatchedWriterSpec(bool useMC)
{
  auto initFunction = [useMC](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("tof-matched-outfile");
    auto treename = ic.options().get<std::string>("treename");

    auto outputfile = std::make_shared<TFile>(filename.c_str(), "RECREATE");
    auto outputtree = std::make_shared<TTree>(treename.c_str(), treename.c_str());

    // container for incoming data
    auto matched = std::make_shared<MatchOutputType>();
    auto labeltof = std::make_shared<std::vector<o2::MCCompLabel>>();
    auto labeltpc = std::make_shared<std::vector<o2::MCCompLabel>>();
    auto labelits = std::make_shared<std::vector<o2::MCCompLabel>>();

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
    auto processingFct = [outputfile, outputtree, matched, labeltof, labeltpc, labelits, useMC](ProcessingContext& pc) {
      static bool finished = false;
      if (finished) {
        // avoid being executed again when marked as finished;
        return;
      }

      // retrieve the matched info from the input
      //auto indata = pc.inputs().get<MatchOutputType*>("tofmatching");

      // THIS PART HAS TO BE INSERTED (HOW?) TO FILL THE MATCHING INFO IN THE OUTPUT FILE
      auto indata = pc.inputs().get<MatchOutputType>("tofmatching");
      LOG(INFO) << "RECEIVED MATCHED SIZE " << indata.size();
      *matched.get() = std::move(indata);

      // connect this to a particular branch
      auto br = getOrMakeBranch(*outputtree.get(), "TOFMatchInfo", matched.get());
      br->Fill();

      if (useMC) {
        // retrieve labels from the input

        auto indatalabeltof = pc.inputs().get<std::vector<o2::MCCompLabel>>("matchtoflabels");
        LOG(INFO) << "TOF LABELS GOT " << indatalabeltof.size() << " LABELS ";
        *labeltof.get() = std::move(indatalabeltof);
        //connect this to a particular branch
        auto labeltofbr = getOrMakeBranch(*outputtree.get(), "MatchTOFMCTruth", labeltof.get());
        labeltofbr->Fill();

        auto indatalabeltpc = pc.inputs().get<std::vector<o2::MCCompLabel>>("matchtpclabels");
        LOG(INFO) << "TPC LABELS GOT " << indatalabeltpc.size() << " LABELS ";
        *labeltpc.get() = std::move(indatalabeltpc);
        //connect this to a particular branch
        auto labeltpcbr = getOrMakeBranch(*outputtree.get(), "MatchTPCMCTruth", labeltpc.get());
        labeltpcbr->Fill();

        auto indatalabelits = pc.inputs().get<std::vector<o2::MCCompLabel>>("matchitslabels");
        LOG(INFO) << "ITS LABELS GOT " << indatalabelits.size() << " LABELS ";
        *labelits.get() = std::move(indatalabelits);
        //connect this to a particular branch
        auto labelitsbr = getOrMakeBranch(*outputtree.get(), "MatchITSMCTruth", labelits.get());
        labelitsbr->Fill();
      }

      finished = true;
      LOG(INFO) << "TOF Macthing info filled! N matched = " << matched.get()->size();
      pc.services().get<ControlService>().readyToQuit(false);
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofmatching", "TOF", "MATCHINFOS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("matchtoflabels", "TOF", "MATCHTOFINFOSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchtpclabels", "TOF", "MATCHTPCINFOSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchitslabels", "TOF", "MATCHITSINFOSMC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "TOFMatchedWriter",
    inputs,
    {}, // no output
    AlgorithmSpec(initFunction),
    Options{
      {"tof-matched-outfile", VariantType::String, "o2match_tof.root", {"Name of the input file"}},
      {"treename", VariantType::String, "matchTOF", {"Name of top-level TTree"}},
    }};
}
} // end namespace tof
} // end namespace o2
