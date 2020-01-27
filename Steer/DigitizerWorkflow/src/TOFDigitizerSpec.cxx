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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "TChain.h"
#include "DetectorsBase/GeometryManager.h"

#include "TOFSimulation/Digitizer.h"
#include "DataFormatsParameters/GRPObject.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"
#include "TOFCalibration/CalibTOFapi.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tof
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
    LOG(ERROR) << "No branch found";
    return;
  }
  br->SetAddress(&hits);
  br->GetEntry(entryID);
}

DataProcessorSpec getTOFDigitizerSpec(int channel, bool useCCDB)
{
  // setup of some data structures shared between init and processing functions
  // (a shared pointer is used since then automatic cleanup is guaranteed with a lifetime beyond
  //  one process call)
  auto simChains = std::make_shared<std::vector<TChain*>>();

  // the instance of the actual digitizer
  auto digitizer = std::make_shared<o2::tof::Digitizer>();

  // containers for digits and labels
  auto digits = std::make_shared<std::vector<o2::tof::Digit>>();
  //  auto digitsAccum = std::make_shared<o2::tof::Digit>(); // accumulator for all digits
  auto labels = std::make_shared<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

  // the actual processing function which get called whenever new data is incoming
  auto process = [simChains, digitizer, digits, /*digitsAccum,*/ labels, channel, useCCDB](ProcessingContext& pc) {
    static bool finished = false;
    if (finished) {
      return;
    }
    // RS: at the moment using hardcoded flag for continuos readout
    static o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::CONTINUOUS;

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
    o2::dataformats::CalibLHCphaseTOF lhcPhaseObj;
    o2::dataformats::CalibTimeSlewingParamTOF channelCalibObj;

    if (useCCDB) { // read calibration objects from ccdb
      // check LHC phase
      auto lhcPhase = pc.inputs().get<o2::dataformats::CalibLHCphaseTOF*>("tofccdbLHCphase");
      auto channelCalib = pc.inputs().get<o2::dataformats::CalibTimeSlewingParamTOF*>("tofccdbChannelCalib");

      o2::dataformats::CalibLHCphaseTOF lhcPhaseObjTmp = std::move(*lhcPhase);
      o2::dataformats::CalibTimeSlewingParamTOF channelCalibObjTmp = std::move(*channelCalib);

      // make a copy in global scope
      lhcPhaseObj = lhcPhaseObjTmp;
      channelCalibObj = channelCalibObjTmp;
    } else { // calibration objects set to zero
      lhcPhaseObj.addLHCphase(0, 0);
      lhcPhaseObj.addLHCphase(2000000000, 0);

      for (int ich = 0; ich < o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELS; ich++) {
        channelCalibObj.addTimeSlewingInfo(ich, 0, 0);
        int sector = ich / o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
        int channelInSector = ich % o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
        channelCalibObj.setFractionUnderPeak(sector, channelInSector, 1);
      }
    }

    o2::tof::CalibTOFapi calibapi(long(0), &lhcPhaseObj, &channelCalibObj);
    digitizer->setCalibApi(&calibapi);

    static std::vector<o2::tof::HitType> hits;
    //    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelAccum;

    auto& eventParts = context->getEventParts();
    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < timesview.size(); ++collID) {
      digitizer->setEventTime(timesview[collID].timeNS);

      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        digitizer->setEventID(part.entryID);
        digitizer->setSrcID(part.sourceID);

        // get the hits for this event and this source
        hits.clear();
        retrieveHits(*simChains.get(), "TOFHit", part.sourceID, part.entryID, &hits);

        //        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found " << hits.size() << " hits ";

        // call actual digitization procedure
        labels->clear();
        digits->clear();
        digitizer->process(&hits, digits.get());
        // copy digits into accumulator
        //std::copy(digits->begin(), digits->end(), std::back_inserter(*digitsAccum.get()));
        //labelAccum.mergeAtBack(*labels);
        //        LOG(INFO) << "Have " << digits->size() << " digits ";
      }
    }
    if (digitizer->isContinuous()) {
      digits->clear();
      labels->clear();
      digitizer->flushOutputContainer(*digits.get());
      LOG(INFO) << "FLUSHING LEFTOVER STUFF " << digits->size();
    }

    std::vector<Digit>* digitsVector = digitizer->getDigitPerTimeFrame();
    std::vector<ReadoutWindowData>* readoutwindow = digitizer->getReadoutWindowData();
    std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* mcLabVecOfVec = digitizer->getMCTruthPerTimeFrame();

    // here we have all digits and we can send them to consumer (aka snapshot it onto output)
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe}, *digitsVector);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIGITSMCTR", 0, Lifetime::Timeframe}, *mcLabVecOfVec);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe}, *readoutwindow);
    LOG(INFO) << "TOF: Sending ROMode= " << roMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "ROMode", 0, Lifetime::Timeframe}, roMode);

    timer.Stop();
    LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    finished = true;
  };

  // init function returning the lambda taking a ProcessingContext
  auto initIt = [simChains, process, digitizer, labels](InitContext& ctx) {
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

    // make sure that the geometry is loaded (TODO will this be done centrally?)
    if (!gGeoManager) {
      o2::base::GeometryManager::loadGeometry();
    }

    // init digitizer
    digitizer->init();
    const bool isContinuous = ctx.options().get<int>("pileup");
    LOG(INFO) << "CONTINUOUS " << isContinuous;
    digitizer->setContinuous(isContinuous);
    digitizer->setMCTruthContainer(labels.get());

    // return the actual processing function which is now setup/configured
    return process;
  };

  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  if (useCCDB) {
    inputs.emplace_back("tofccdbLHCphase", o2::header::gDataOriginTOF, "LHCphase");
    inputs.emplace_back("tofccdbChannelCalib", o2::header::gDataOriginTOF, "ChannelCalib");
  }
  return DataProcessorSpec{
    "TOFDigitizer",
    inputs,
    Outputs{OutputSpec{o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe},
            OutputSpec{o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe},
            OutputSpec{o2::header::gDataOriginTOF, "DIGITSMCTR", 0, Lifetime::Timeframe},
            OutputSpec{o2::header::gDataOriginTOF, "ROMode", 0, Lifetime::Timeframe}},
    AlgorithmSpec{initIt},
    Options{{"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
            {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}},
            {"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}}}
    // I can't use VariantType::Bool as it seems to have a problem
  };
}
} // end namespace tof
} // end namespace o2
