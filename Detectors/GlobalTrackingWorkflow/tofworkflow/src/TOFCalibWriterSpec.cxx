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

#include "TOFWorkflow/TOFCalibWriterSpec.h"
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
using OutputType = std::vector<o2::dataformats::CalibInfoTOF>;

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
/// describing a processor receiving calib info for TOF writing them to file
/// TODO: make this processor generic and reusable!!
DataProcessorSpec getTOFCalibWriterSpec()
{
  auto initFunction = [](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("tof-calib-outfile");
    auto treename = ic.options().get<std::string>("treename");

    auto outputfile = std::make_shared<TFile>(filename.c_str(), "RECREATE");
    auto outputtree = std::make_shared<TTree>(treename.c_str(), treename.c_str());

    // container for incoming data
    auto calibinfo = std::make_shared<OutputType>();

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
    auto processingFct = [outputfile, outputtree, calibinfo](ProcessingContext& pc) {
      static bool finished = false;
      if (finished) {
        // avoid being executed again when marked as finished;
        return;
      }

      // retrieve the calib info from the input

      auto indata = pc.inputs().get<OutputType>("tofcalibinfo");
      LOG(INFO) << "RECEIVED CALIB INFO SIZE " << indata.size();
      *calibinfo.get() = std::move(indata);

      // connect this to a particular branch
      auto br = getOrMakeBranch(*outputtree.get(), "TOFCalibInfo", calibinfo.get());
      br->Fill();

      finished = true;
      LOG(INFO) << "TOF Calib info filled! N matched = " << calibinfo.get()->size();
      pc.services().get<ControlService>().readyToQuit(false);
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofcalibinfo", "TOF", "CALIBINFOS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFCalibWriter",
    inputs,
    {}, // no output
    AlgorithmSpec(initFunction),
    Options{
      {"tof-calib-outfile", VariantType::String, "o2calib_tof.root", {"Name of the input file"}},
      {"treename", VariantType::String, "calibTOF", {"Name of top-level TTree"}},
    }};
}
} // end namespace tof
} // end namespace o2
