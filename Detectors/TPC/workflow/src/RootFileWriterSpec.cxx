// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RootFileWriterSpec.cxx
/// @author Matthias Richter
/// @since  2018-04-19
/// @brief  Processor spec for a ROOT file writer

#include "RootFileWriterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <TFile.h>
#include <TTree.h>
#include <memory> // for make_shared, make_unique, unique_ptr
#include <stdexcept>

using namespace o2::framework;

using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

namespace o2
{
namespace TPC
{
/// create a processor spec
/// read simulated TPC digits from file and publish
DataProcessorSpec getRootFileWriterSpec(bool writeMC)
{
  auto initFunction = [writeMC](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("outfile");
    auto treename = ic.options().get<std::string>("treename");
    auto trackBranchName = ic.options().get<std::string>("track-branch-name");
    auto trackMcBranchName = ic.options().get<std::string>("trackmc-branch-name");
    auto nevents = ic.options().get<int>("nevents");

    auto outputfile = std::make_shared<TFile>(filename.c_str(), "RECREATE");
    auto outputtree = std::make_shared<TTree>(treename.c_str(), treename.c_str());
    auto tracks = std::make_shared<std::vector<o2::TPC::TrackTPC>>();
    auto mclabels = std::make_shared<MCLabelContainer>();
    outputtree->Branch(trackBranchName.c_str(), tracks.get());
    if (writeMC) {
      outputtree->Branch(trackMcBranchName.c_str(), mclabels.get());
    }

    // the callback to be set as hook at stop of processing for the framework
    auto finishWriting = [outputfile, outputtree]() {
      outputtree->Write();
      outputfile->Close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishWriting);

    // set up the processing function
    // using by-copy capture of the worker instance shared pointer
    // the shared pointer makes sure to clean up the instance when the processing
    // function gets out of scope
    auto processingFct = [outputfile, outputtree, tracks, mclabels, nevents, writeMC](ProcessingContext& pc) {
      static int eventCount = 0;
      auto indata = pc.inputs().get<std::vector<o2::TPC::TrackTPC>>("input");
      LOG(INFO) << "RootFileWriter: get " << indata.size() << " track(s)";
      *tracks.get() = std::move(indata);
      if (writeMC) {
        auto mcdata = pc.inputs().get<MCLabelContainer*>("mcinput");
        *mclabels.get() = *mcdata;
      }
      LOG(INFO) << "RootFileWriter: write " << tracks->size() << " track(s)";
      outputtree->Fill();

      if (++eventCount >= nevents) {
        pc.services().get<ControlService>().readyToQuit(true);
      }
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  auto createInputSpecs = [](bool makeMcInput) {
    std::vector<InputSpec> inputSpecs{
      InputSpec{ { "input" }, "TPC", "TRACKS", 0, Lifetime::Timeframe },
    };
    if (makeMcInput) {
      constexpr o2::header::DataDescription datadesc("TRACKMCLBL");
      inputSpecs.emplace_back(InputSpec{ "mcinput", "TPC", datadesc, 0, Lifetime::Timeframe });
    }
    return std::move(inputSpecs);
  };

  return DataProcessorSpec{ "tpc-track-writer",
                            { createInputSpecs(writeMC) },
                            {},
                            AlgorithmSpec(initFunction),
                            Options{
                              { "outfile", VariantType::String, "tpctracks.root", { "Name of the input file" } },
                              { "treename", VariantType::String, "o2sim", { "Name of output tree" } },
                              { "track-branch-name", VariantType::String, "TPCTracks", { "Branch name for TPC tracks" } },
                              { "trackmc-branch-name", VariantType::String, "TPCTracksMCTruth", { "Branch name for TPC track mc labels" } },
                              { "nevents", VariantType::Int, 1, { "terminate after n events" } },
                            } };
}
} // end namespace TPC
} // end namespace o2
