// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITSDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "ITSMFTBase/Digit.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSMFTSimulation/Digitizer.h"
#include "ITSBase/GeometryTGeo.h"
#include <TGeoManager.h>
#include <TChain.h>
#include <TStopwatch.h>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace ITS
{

// helper function which will be offered as a service
template <typename T>
void retrieveHits(std::vector<TChain*> const& chains,
                  const char* brname,
                  int sourceID,
                  int entryID,
                  std::vector<T>* hits)
{
  auto br = chains[sourceID]->GetBranch(brname);
  if (!br) {
    LOG(ERROR) << "No branch found" << FairLogger::endl;
    return;
  }
  br->SetAddress(&hits);
  br->GetEntry(entryID);
}

DataProcessorSpec getITSDigitizerSpec(int channel)
{
  // setup of some data structures shared between init and processing functions
  // (a shared pointer is used since then automatic cleanup is guaranteed with a lifetime beyond
  //  one process call)
  auto simChains = std::make_shared<std::vector<TChain*>>();
  auto qedChain = std::make_shared<TChain>("o2sim");

  // the instance of the actual digitizer
  auto digitizer = std::make_shared<o2::ITSMFT::Digitizer>();

  // containers for digits and labels
  auto digits = std::make_shared<std::vector<o2::ITSMFT::Digit>>();
  auto digitsAccum = std::make_shared<std::vector<o2::ITSMFT::Digit>>(); // accumulator for all digits
  auto labels = std::make_shared<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
  auto labelsAccum = std::make_shared<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

  // the actual processing function which get called whenever new data is incoming
  auto process = [simChains, qedChain, digitizer, digits, digitsAccum, labels, labelsAccum, channel](ProcessingContext& pc) {
    static bool finished = false;
    if (finished) {
      return;
    }

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
    auto& timesview = context->getEventRecords();
    LOG(DEBUG) << "GOT " << timesview.size() << " COLLISSION TIMES" << FairLogger::endl;

    // if there is nothing to do ... return
    if (timesview.size() == 0) {
      return;
    }

    TStopwatch timer;
    timer.Start();

    LOG(INFO) << " CALLING ITS DIGITIZATION " << FairLogger::endl;

    static std::vector<o2::ITSMFT::Hit> hits;
    static std::vector<o2::ITSMFT::Hit> hitsQED;

    digitizer->setDigits(digits.get());
    digitizer->setMCLabels(labels.get());

    // attach optional QED digits branch
    static double qedEntryTimeBinNS = 1000; // time-coverage of single QED tree entry in ns (TODO: make it settable)
    static double lastQEDTimeNS = 0;
    const unsigned char qedSourceID = 99; // unique source ID for the QED (TODO: move it as a const to general class?)

    TBranch* qedBranch = nullptr;
    if (qedChain->GetEntries()) {
      qedBranch = qedChain->GetBranch("ITSHit");
      assert(qedBranch != nullptr);
      assert(qedEntryTimeBinNS >= 1.0);
      assert(qedSourceID < o2::MCCompLabel::maxSourceID());
      lastQEDTimeNS = -qedEntryTimeBinNS / 2; // time will be assigned to the middle of the bin
      auto phitsQED = &hitsQED;
      qedBranch->SetAddress(&phitsQED);
      LOG(INFO) << "Attaching QED ITS hits as sourceID=" << int(qedSourceID) << ", entry integrates "
                << qedEntryTimeBinNS << " ns" << FairLogger::endl;
    }
    // function to process QED hits
    auto processQED = [digitizer, qedBranch](double tMax) {
      static int lastQEDEntry = -1;
      auto tQEDNext = lastQEDTimeNS + qedEntryTimeBinNS; // timeslice to retrieve
      while (tQEDNext < tMax) {
        lastQEDTimeNS = tQEDNext;      // time used for current QED slot
        tQEDNext += qedEntryTimeBinNS; // prepare time for next QED slot
        if (++lastQEDEntry >= qedBranch->GetEntries()) {
          lastQEDEntry = 0; // wrapp if needed
        }
        qedBranch->GetEntry(lastQEDEntry);
        digitizer->setEventTime(lastQEDTimeNS);
        digitizer->process(&hitsQED, lastQEDEntry, qedSourceID);
        //
      }
    };

    auto& eventParts = context->getEventParts();
    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < timesview.size(); ++collID) {
      auto eventTime = timesview[collID].timeNS;
      labels->clear();
      digits->clear();

      if (qedBranch) { // QED must be processed before other inputs since done in small time steps
        processQED(eventTime);
      }

      digitizer->setEventTime(eventTime);

      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {

        // get the hits for this event and this source
        hits.clear();
        retrieveHits(*simChains.get(), "ITSHit", part.sourceID, part.entryID, &hits);

        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID
                  << " found " << hits.size() << " hits " << FairLogger::endl;

        // call actual digitization procedure
        digitizer->process(&hits, collID, part.sourceID);
        // copy digits into accumulator
      }

      std::copy(digits->begin(), digits->end(), std::back_inserter(*digitsAccum.get()));
      labelsAccum->mergeAtBack(*labels);
      LOG(INFO) << "Have " << digits->size() << " digits " << FairLogger::endl;
    }

    // finish digitization ... stream any remaining digits/labels
    labels->clear();
    digits->clear();

    if (qedBranch) { // fill last slots from QED input
      processQED(digitizer->getEndTimeOfROFMax());
    }
    digitizer->fillOutputContainer();
    std::copy(digits->begin(), digits->end(), std::back_inserter(*digitsAccum.get()));
    labelsAccum->mergeAtBack(*labels);
    LOG(INFO) << "Afterburner: Have " << digits->size() << " digits " << FairLogger::endl;

    // here we have all digits and labels and we can send them to consumer (aka snapshot it onto output)
    pc.outputs().snapshot(Output{ "ITS", "DIGITS", 0, Lifetime::Timeframe }, *digitsAccum.get());
    pc.outputs().snapshot(Output{ "ITS", "DIGITSMCTR", 0, Lifetime::Timeframe }, *labelsAccum.get());

    timer.Stop();
    LOG(INFO) << "Digitization took " << timer.CpuTime() << "s" << FairLogger::endl;

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(false);
    finished = true;
  };

  // init function returning the lambda taking a ProcessingContext
  auto initIt = [simChains, qedChain, process, digitizer](InitContext& ctx) {
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

    // init optional QED chain
    auto qedfilename = ctx.options().get<std::string>("simFileQED");
    if (qedfilename.size() > 0) {
      qedChain->AddFile(qedfilename.c_str());
    }
    LOG(INFO) << "Attach QED Tree: " << qedChain->GetEntries() << FairLogger::endl;
    // make sure that the geometry is loaded (TODO will this be done centrally?)
    if (!gGeoManager) {
      o2::Base::GeometryManager::loadGeometry();
    }

    // configure digitizer
    GeometryTGeo* geom = ITS::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::L2G)); // make sure L2G matrices are loaded
    digitizer->setGeometry(geom);

    // defaults (TODO we need a way to pass particular configuration parameters)
    digitizer->getParams().setContinuous(true);    // continuous vs per-event mode
    digitizer->getParams().setROFrameLength(6000); // RO frame in ns
    digitizer->getParams().setStrobeDelay(6000);   // Strobe delay wrt beginning of the RO frame, in ns
    digitizer->getParams().setStrobeLength(100);   // Strobe length in ns
    // parameters of signal time response: flat-top duration, max rise time and q @ which rise time is 0
    digitizer->getParams().getSignalShape().setParameters(7500., 1100., 450.);
    digitizer->getParams().setChargeThreshold(150); // charge threshold in electrons
    digitizer->getParams().setNoisePerPixel(1.e-7); // noise level
    // init digitizer
    digitizer->init();

    // return the actual processing function which is now setup/configured
    return process;
  };

  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actuall processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "ITSDigitizer", Inputs{ InputSpec{ "collisioncontext", "SIM", "COLLISIONCONTEXT",
                                       static_cast<SubSpecificationType>(channel), Lifetime::Timeframe } },
    Outputs{
      OutputSpec{ "ITS", "DIGITS", 0, Lifetime::Timeframe },
      OutputSpec{ "ITS", "DIGITSMCTR", 0, Lifetime::Timeframe } },
    AlgorithmSpec{ initIt },
    Options{
      { "simFile", VariantType::String, "o2sim.root", { "Sim (background) input filename" } },
      { "simFileS", VariantType::String, "", { "Sim (signal) input filename" } },
      { "simFileQED", VariantType::String, "", { "Sim (QED) input filename" } } }
  };
}
} // end namespace ITS
} // end namespace o2
