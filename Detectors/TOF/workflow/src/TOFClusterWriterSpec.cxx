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

#include "TOFWorkflow/TOFClusterWriterSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "DataFormatsTOF/Cluster.h"
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
/// describing a processor receiving clusters for TOF writing them to file
/// TODO: make this processor generic and reusable!!
DataProcessorSpec getTOFClusterWriterSpec(bool useMC)
{
  auto initFunction = [useMC](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("tof-cluster-outfile");
    auto treename = ic.options().get<std::string>("treename");

    auto outputfile = std::make_shared<TFile>(filename.c_str(), "RECREATE");
    auto outputtree = std::make_shared<TTree>(treename.c_str(), treename.c_str());

    // container for incoming data
    auto digits = std::make_shared<std::vector<o2::tof::Cluster>>();

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
    auto processingFct = [outputfile, outputtree, digits, useMC](ProcessingContext& pc) {
      static bool finished = false;
      if (finished) {
        // avoid being executed again when marked as finished;
        return;
      }

      // retrieve the digits from the input
      auto indata = pc.inputs().get<std::vector<o2::tof::Cluster>>("tofclusters");
      LOG(INFO) << "RECEIVED CLUSTERS SIZE " << indata.size();
      *digits.get() = std::move(indata);

      // connect this to a particular branch
      auto br = getOrMakeBranch(*outputtree.get(), "TOFCluster", digits.get());
      br->Fill();

      if (useMC) { // retrieve labels from the input
        auto labeldata = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("tofclusterlabels");
        LOG(INFO) << "TOF GOT " << labeldata->getNElements() << " LABELS ";
        auto labeldataraw = labeldata.get();
        // connect this to a particular branch

        auto labelbr = getOrMakeBranch(*outputtree.get(), "TOFClusterMCTruth", &labeldataraw);
        labelbr->Fill();
      }

      finished = true;
      pc.services().get<ControlService>().readyToQuit(false);
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofclusters", "TOF", "CLUSTERS", 0, Lifetime::Timeframe);
  if (useMC)
    inputs.emplace_back("tofclusterlabels", "TOF", "CLUSTERSMCTR", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFClusterWriter",
    inputs,
    {}, // no output
    AlgorithmSpec(initFunction),
    Options{
      {"tof-cluster-outfile", VariantType::String, "tofclusters.root", {"Name of the input file"}},
      {"treename", VariantType::String, "o2sim", {"Name of top-level TTree"}},
    }};
}
} // end namespace tof
} // end namespace o2
