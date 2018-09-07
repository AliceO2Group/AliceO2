// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "TChain.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tof
{

DataProcessorSpec getTOFDigitizerSpec(int channel)
{
  // setup of some data structures shared between init and processing functions
  // (a shared pointer is used since then automatic cleanup is guaranteed with a lifetime beyond
  //  one process call)
  auto simChains = std::make_shared<std::vector<TChain*>>();

  // the actual processing function which get called whenever new data is incoming
  auto process = [simChains, channel](ProcessingContext& pc) {
    static bool finished = false;
    if (finished) {
      return;
    }

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
    auto& timesview = context->getEventRecords();
    LOG(DEBUG) << "GOT " << timesview.size() << " COLLISSION TIMES";

    // if there is nothing to do ... return
    if (timesview.size() == 0) {
      return;
    }

    TStopwatch timer;
    timer.Start();

    LOG(INFO) << " CALLING TOF DIGITIZATION ";
    // call digitizer here

    timer.Stop();
    LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(false);
    finished = true;
  };

  // init function returning the lambda taking a ProcessingContext
  auto initIt = [simChains, process](InitContext& ctx) {
    // setup the input chain for the hits
    simChains->emplace_back(new TChain("o2sim"));

    // add the main (background) file
    simChains->back()->AddFile(ctx.options().get<std::string>("simFile").c_str());

    // maybe add a particular signal file
    auto signalfilename = ctx.options().get<std::string>("simFileS");
    if (signalfilename.size() > 0) {
      simChains->emplace_back(new TChain("o2sim"));
      simChains->back()->AddFile(signalfilename.c_str());
    }

    // return the actual processing function which is now setup/configured
    return process;
  };

  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actuall processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "TOFDigitizer", Inputs{ InputSpec{ "collisioncontext", "SIM", "COLLISIONCONTEXT",
                                       static_cast<SubSpecificationType>(channel), Lifetime::Timeframe } },
    Outputs{},
    AlgorithmSpec{ initIt },
    Options{ { "simFile", VariantType::String, "o2sim.root", { "Sim (background) input filename" } },
             { "simFileS", VariantType::String, "", { "Sim (signal) input filename" } } }
  };
}
} // end namespace tof
} // end namespace o2
