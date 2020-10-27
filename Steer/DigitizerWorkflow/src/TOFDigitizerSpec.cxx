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
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include "DetectorsBase/GeometryManager.h"
#include "TOFSimulation/Digitizer.h"
#include "DataFormatsParameters/GRPObject.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"
#include "TOFCalibration/CalibTOFapi.h"
#include "SimConfig/DigiParams.h"
#include "DetectorsBase/BaseDPLDigitizer.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tof
{

class TOFDPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
 public:
  TOFDPLDigitizerTask(bool useCCDB) : mUseCCDB{useCCDB},
                                      o2::base::BaseDPLDigitizer(o2::base::InitServices::FIELD | o2::base::InitServices::GEOM){};

  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(INFO) << "Initializing TOF digitization";

    mSimChains = std::move(std::make_unique<std::vector<TChain*>>());

    // the instance of the actual digitizer
    mDigitizer = std::move(std::make_unique<o2::tof::Digitizer>());
    // containers for digits and labels
    mDigits = std::move(std::make_unique<std::vector<o2::tof::Digit>>());
    mLabels = std::move(std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>());

    // init digitizer
    mDigitizer->init();
    const bool isContinuous = ic.options().get<int>("pileup");
    LOG(INFO) << "CONTINUOUS " << isContinuous;
    mDigitizer->setContinuous(isContinuous);
    mDigitizer->setMCTruthContainer(mLabels.get());
    LOG(INFO) << "TOF initialization done";
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }
    // RS: at the moment using hardcoded flag for continuos readout
    static o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::CONTINUOUS;

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    auto& timesview = context->getEventRecords();
    LOG(DEBUG) << "GOT " << timesview.size() << " COLLISSION TIMES";

    context->initSimChains(o2::detectors::DetID::TOF, *mSimChains.get());

    // if there is nothing to do ... return
    if (timesview.size() == 0) {
      return;
    }

    TStopwatch timer;
    timer.Start();

    LOG(INFO) << " CALLING TOF DIGITIZATION ";
    o2::dataformats::CalibLHCphaseTOF lhcPhaseObj;
    o2::dataformats::CalibTimeSlewingParamTOF channelCalibObj;

    if (mUseCCDB) { // read calibration objects from ccdb
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
    mDigitizer->setCalibApi(&calibapi);

    static std::vector<o2::tof::HitType> hits;

    auto& eventParts = context->getEventParts();
    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < timesview.size(); ++collID) {
      mDigitizer->setEventTime(timesview[collID]);

      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        mDigitizer->setEventID(part.entryID);
        mDigitizer->setSrcID(part.sourceID);

        // get the hits for this event and this source
        hits.clear();
        context->retrieveHits(*mSimChains.get(), "TOFHit", part.sourceID, part.entryID, &hits);

        //        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found " << hits.size() << " hits ";

        // call actual digitization procedure
        mLabels->clear();
        mDigits->clear();
        mDigitizer->process(&hits, mDigits.get());
      }
    }
    if (mDigitizer->isContinuous()) {
      mDigits->clear();
      mLabels->clear();
      mDigitizer->flushOutputContainer(*mDigits.get());
    }

    std::vector<Digit>* digitsVector = mDigitizer->getDigitPerTimeFrame();
    std::vector<ReadoutWindowData>* readoutwindow = mDigitizer->getReadoutWindowData();
    std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* mcLabVecOfVec = mDigitizer->getMCTruthPerTimeFrame();

    LOG(INFO) << "Post " << digitsVector->size() << " digits in " << readoutwindow->size() << " RO windows";

    // here we have all digits and we can send them to consumer (aka snapshot it onto output)
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe}, *digitsVector);
    if (pc.outputs().isAllowed({o2::header::gDataOriginTOF, "DIGITSMCTR", 0})) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIGITSMCTR", 0, Lifetime::Timeframe}, *mcLabVecOfVec);
    }
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe}, *readoutwindow);

    // send empty pattern from digitizer (it may change in future)
    static std::vector<uint32_t> patterns;
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe}, patterns);

    DigitHeader& digitH = mDigitizer->getDigitHeader();
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIGITHEADER", 0, Lifetime::Timeframe}, digitH);

    LOG(INFO) << "TOF: Sending ROMode= " << roMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "ROMode", 0, Lifetime::Timeframe}, roMode);

    timer.Stop();
    LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().endOfStream();

    finished = true;
  }

 private:
  std::unique_ptr<std::vector<TChain*>> mSimChains;
  std::unique_ptr<o2::tof::Digitizer> mDigitizer;
  std::unique_ptr<std::vector<o2::tof::Digit>> mDigits;
  std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabels;
  bool mUseCCDB = false;
};

DataProcessorSpec getTOFDigitizerSpec(int channel, bool useCCDB, bool mctruth)
{
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
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "DIGITHEADER", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(o2::header::gDataOriginTOF, "ROMode", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "TOFDigitizer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFDPLDigitizerTask>(useCCDB)},
    Options{{"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}}}
    // I can't use VariantType::Bool as it seems to have a problem
  };
}
} // end namespace tof
} // end namespace o2
