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
#include <TFile.h>
#include <TTree.h>
#include <memory> // for make_shared, make_unique, unique_ptr
#include <stdexcept>

using namespace o2::framework;

namespace o2
{
namespace TPC
{
/// create a processor spec
/// read simulated TPC digits from file and publish
DataProcessorSpec getRootFileWriterSpec()
{
  auto initFunction = [](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("outfile");
    auto treename = ic.options().get<std::string>("treename");
    auto nevents = ic.options().get<int>("nevents");

    auto outputfile = std::make_shared<TFile>(filename.c_str(), "RECREATE");
    auto outputtree = std::make_shared<TTree>(treename.c_str(), treename.c_str());
    auto tracks = std::make_shared<std::vector<o2::TPC::TrackTPC>>();
    auto trackbranch = outputtree->Branch("Tracks", tracks.get());

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
    auto processingFct = [outputfile, outputtree, tracks, nevents](ProcessingContext& pc) {
      static int eventCount = 0;
      auto indata = pc.inputs().get<std::vector<o2::TPC::TrackTPC>>("input");
      LOG(INFO) << "RootFileWriter: get " << indata.size() << " track(s)";
      *tracks.get() = std::move(indata);
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

  return DataProcessorSpec{ "writer",
                            { InputSpec{ "input", "TPC", "TRACKS", 0, Lifetime::Timeframe } }, // track input
                            {},                                                                // no output
                            AlgorithmSpec(initFunction),
                            Options{
                              { "outfile", VariantType::String, "tpctracks.root", { "Name of the input file" } },
                              { "treename", VariantType::String, "Tracks", { "Name of tree for tracks" } },
                              { "nevents", VariantType::Int, 1, { "terminate after n events" } },
                            } };
}
} // end namespace TPC
} // end namespace o2
