// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALWorkflow/CalibLoader.h"
#include "EMCALWorkflow/EMCALDigitizerSpec.h"
#include "CommonConstants/Triggers.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include <TGeoManager.h>

#include "CommonDataFormat/EvIndex.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsCTP/Configuration.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFV0/Digit.h"
#include <Steer/MCKinematicsReader.h>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace emcal
{

void DigitizerSpec::initDigitizerTask(framework::InitContext& ctx)
{
  if (!gGeoManager) {
    LOG(error) << "Geometry needs to be loaded before";
  }
  // run 3 geometry == run 2 geometry for EMCAL
  // to be adapted with run numbers at a later stage
  auto geom = o2::emcal::Geometry::GetInstance("EMCAL_COMPLETE12SMV1_DCAL_8SM", "Geant4", "EMV-EMCAL");
  // init digitizer

  mSumDigitizer.setGeometry(geom);
  mSumDigitizerTRU.setGeometry(geom);
  mDigitizerTRU.setGeometry(geom);

  if (ctx.options().get<bool>("debug-stream")) {
    mDigitizer.setDebugStreaming(true);
    mDigitizerTRU.setDebugStreaming(true);
  }
  // mDigitizer.init();
  if (ctx.options().get<bool>("disable-dig-tru")) {
    mRunDigitizerTRU = false;
  }

  mFinished = false;
}

void DigitizerSpec::run(framework::ProcessingContext& ctx)
{
  if (mFinished) {
    return;
  }
  if (mCalibHandler) {
    // Load CCDB object (sim params)
    mCalibHandler->checkUpdates(ctx);
  }

  if (!mIsConfigured) {
    configure();
    mIsConfigured = true;
  }

  o2::emcal::SimParam::Instance().printKeyValues(true, true);

  mDigitizer.flush();
  mDigitizerTRU.flush();

  // read collision context from input
  auto context = ctx.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");

  // get interaction rate ... so that it can be used in digitization
  auto intRate = context->getDigitizerInteractionRate();
  context->initSimChains(o2::detectors::DetID::EMC, mSimChains);

  // init the MCKinematicsReader from the digitization context
  if (mcReader == nullptr) {
    mcReader = new o2::steer::MCKinematicsReader(context.get());
  }

  auto& timesview = context->getEventRecords();
  LOG(debug) << "GOT " << timesview.size() << " COLLISSION TIMES";

  // if there is nothing to do ... return
  if (timesview.size() == 0) {
    return;
  }

  TStopwatch timer;
  timer.Start();

  auto& eventParts = context->getEventParts();

  // ------------------------------
  // TRIGGER Simulation
  // ------------------------------
  // 1. Run SDigitizer separate loop -> map of vector of SDigits
  // 2. Run Trigger simulation chain (Digitizer -> Digits writeout buffer -> L0 Simulation)
  // 3. Run Loop SDigits -> Digits, set live only if the trigger is accepted.
  //
  // loop over all composite collisions given from context
  // (aka loop over all the interaction records)
  int collisionN = 0;
  for (int collID = 0; collID < timesview.size(); ++collID) {

    if (mRunDigitizerTRU == false) {
      break;
    }

    LOG(info) << "DIG TRU in SPEC: before mDigitizerTRU.setEventTime  collN = " << collisionN;
    collisionN++;

    mDigitizerTRU.setEventTime(timesview[collID]);

    // for each collision, loop over the constituents event and source IDs
    // (background signal merging is basically taking place here)
    for (auto& part : eventParts[collID]) {

      mSumDigitizerTRU.setCurrEvID(part.entryID);
      mSumDigitizerTRU.setCurrSrcID(part.sourceID);

      // get the hits for this event and this source
      mHits.clear();
      context->retrieveHits(mSimChains, "EMCHit", part.sourceID, part.entryID, &mHits);

      LOG(info) << "DIG TRU For collision " << collID << " eventID " << part.entryID << " found " << mHits.size() << " hits ";

      // std::vector<o2::emcal::LabeledDigit> summedLabeledDigits = mSumDigitizerTRU.process(mHits);
      // std::vector<o2::emcal::Digit> summedDigits;
      // for (auto labeledsummeddigit : summedLabeledDigits) {
      //   summedDigits.push_back(labeledsummeddigit.getDigit());
      // }

      std::vector<o2::emcal::LabeledDigit> summedLabeledDigits;
      std::vector<o2::emcal::Digit> summedDigits;
      if (mRunSDitizer) {
        summedLabeledDigits = mSumDigitizerTRU.process(mHits);
        for (auto labeledsummeddigit : summedLabeledDigits) {
          summedDigits.push_back(labeledsummeddigit.getDigit());
        }
      } else {
        for (auto& hit : mHits) {
          summedDigits.emplace_back(hit.GetDetectorID(), hit.GetEnergyLoss(), hit.GetTime());
        }
      }

      // call actual digitization procedure
      mDigitizerTRU.process(summedDigits);
    }
  }
  mDigitizerTRU.finish();
  // Result of the trigger simulation
  // -> Set of BCs with triggering patches
  auto emcalTriggers = mDigitizerTRU.getTriggerInputs();

  // Load FIT triggers if not running in self-triggered mode
  std::vector<o2::InteractionRecord> mbtriggers;
  if (mRequireCTPInput) {
    // Hopefully at some point we can replace it by CTP input digits
    // In case of CTP digits react only to trigger inputs activated in trigger configuration
    // For the moment react to FIT vertex, cent and semicent triggers
    ctx.inputs().get<o2::ctp::CTPConfiguration*>("ctpconfig");
    std::vector<uint64_t> inputmasks;
    for (const auto& trg : mCTPConfig->getCTPClasses()) {
      if (trg.cluster->maskCluster[o2::detectors::DetID::EMC]) {
        // Class triggering EMCAL cluster
        LOG(debug) << "Found trigger class for EMCAL cluster: " << trg.name << " with input mask " << std::bitset<64>(trg.descriptor->getInputsMask());
        inputmasks.emplace_back(trg.descriptor->getInputsMask());
      }
    }
    unsigned long ft0mask = 0, fv0mask = 0;
    std::map<std::string, uint64_t> detInputName2Mask =
      {{"MVBA", 1}, {"MVOR", 2}, {"MVNC", 4}, {"MVCH", 8}, {"MVIR", 0x10}, {"MT0A", 1}, {"MT0C", 2}, {"MTSC", 4}, {"MTCE", 8}, {"MTVX", 0x10}};
    // Translation 2022: EMCAL cluster received CTP input masks, need to track it to FIT trigger masks
    std::map<std::string, std::pair<o2::detectors::DetID, std::string>> ctpInput2DetInput = {
      {"0VBA", {o2::detectors::DetID::FV0, "MVBA"}}, {"0VOR", {o2::detectors::DetID::FV0, "MVOR"}}, {"0VNC", {o2::detectors::DetID::FV0, "MVNC"}}, {"0VCH", {o2::detectors::DetID::FV0, "MVCH"}}, {"0VIR", {o2::detectors::DetID::FV0, "MVIR"}}, {"0T0A", {o2::detectors::DetID::FT0, "MT0A"}}, {"0T0C", {o2::detectors::DetID::FT0, "MT0C"}}, {"0TSC", {o2::detectors::DetID::FT0, "MTSC"}}, {"0TCE", {o2::detectors::DetID::FT0, "MTCE"}}, {"0TVX", {o2::detectors::DetID::FT0, "MTVX"}}};
    for (const auto& [det, ctpinputs] : mCTPConfig->getDet2InputMap()) {
      if (!(det == o2::detectors::DetID::FT0 || det == o2::detectors::DetID::FV0 || det == o2::detectors::DetID::CTP)) {
        continue;
      }
      for (const auto& input : ctpinputs) {
        LOG(debug) << "CTP det input: " << input.name << " with mask " << std::bitset<64>(input.inputMask);
        bool isSelected = false;
        for (auto testmask : inputmasks) {
          if (testmask & input.inputMask) {
            isSelected = true;
          }
        }
        if (isSelected) {
          std::string usedInputName = input.name;
          o2::detectors::DetID usedDetID = det;
          if (det == o2::detectors::DetID::CTP) {
            auto found = ctpInput2DetInput.find(input.name);
            if (found != ctpInput2DetInput.end()) {
              usedInputName = found->second.second;
              usedDetID = found->second.first;
              LOG(debug) << "Decoded " << input.name << " -> " << usedInputName;
            }
          }
          auto maskFound = detInputName2Mask.find(usedInputName);
          if (maskFound != detInputName2Mask.end()) {
            if (usedDetID == o2::detectors::DetID::FT0) {
              ft0mask |= maskFound->second;
            } else {
              fv0mask |= maskFound->second;
            }
          }
        }
      }
    }
    LOG(debug) << "FTO mask: " << std::bitset<64>(ft0mask);
    LOG(debug) << "FVO mask: " << std::bitset<64>(fv0mask);
    for (const auto& trg : ctx.inputs().get<gsl::span<o2::ft0::DetTrigInput>>("ft0inputs")) {
      if (trg.mInputs.to_ulong() & ft0mask) {
        mbtriggers.emplace_back(trg.mIntRecord);
      }
    }
    for (const auto& trg : ctx.inputs().get<gsl::span<o2::fv0::DetTrigInput>>("fv0inputs")) {
      if (trg.mInputs.to_ulong() & fv0mask) {
        if (std::find(mbtriggers.begin(), mbtriggers.end(), trg.mIntRecord) == mbtriggers.end()) {
          mbtriggers.emplace_back(trg.mIntRecord);
        }
      }
    }
  }

  LOG(info) << " CALLING EMCAL DIGITIZATION ";
  o2::dataformats::MCTruthContainer<o2::emcal::MCLabel> labelAccum;

  // auto& eventParts = context->getEventParts();
  std::vector<std::tuple<o2::InteractionRecord, std::bitset<5>>> acceptedTriggers;
  enum EMCALTriggerBits { kMB,
                          kEMC,
                          kDMC }; // EMCAL trigger bits enum for CTP inputs
  TRandom3 mRandomGenerator(std::chrono::high_resolution_clock::now().time_since_epoch().count());

  // loop over all composite collisions given from context
  // (aka loop over all the interaction records)
  for (int collID = 0; collID < timesview.size(); ++collID) {

    std::bitset<5> trigger{0x1}; // Default: Self-triggered mode - all collisions treated as trigger
    if (mRequireCTPInput) {
      // ====================================================
      // check if we have a trigger input from CTP
      // Simulated L0 and downsampling
      // 2 cases -> Check from CTP config
      // 1) EMCAL trigger active -> MB downsampled
      //    - Accept the event in 2 conditions
      //      + MB downsampling (first condition) -> Set trigger bit kTVXinEMC
      //      + EMCAL L0 trigger -> Set bit EMCAL L0
      //      Always test both, set trigger 0x1 -> EMBA, 0x2 -> 0EMC, 0x4 -> 0DCMX, 0 -> No trigger
      // 2) EMCAL triggers not active
      // .  - Old logics kept, no downscaling, pure busy
      // ----------------------------------------------------
      // PRNG for downscaling check
      auto mbtrigger = std::find(mbtriggers.begin(), mbtriggers.end(), timesview[collID]);
      if (mbtrigger != mbtriggers.end()) {

        // ============================
        // retrieve downscaling from
        // configuration of CTP classes
        int downscaling = 0;
        for (const auto& trg : mCTPConfig->getCTPClasses()) {
          if (trg.cluster->maskCluster[o2::detectors::DetID::EMC]) {
            // Class triggering EMCAL cluster
            downscaling = trg.downScale;
          }
        }
        if (mRandomGenerator.Uniform(0., 1) < downscaling) {
          // accept as minimum bias
          trigger.set(EMCALTriggerBits::kMB, true);
        }
        // check for LO triggers in 12 BCs
        for (auto emcalTrigger : emcalTriggers) {
          auto bcTimingOfEmcalTrigger = emcalTrigger.mInterRecord.bc;
          auto bcTimingOfMBTrigger = (*mbtrigger).bc;
          if (std::abs(bcTimingOfEmcalTrigger - bcTimingOfMBTrigger) < 12) {
            if (emcalTrigger.mTriggeredTRU < 32) {
              trigger.set(EMCALTriggerBits::kEMC, true);
            } else {
              trigger.set(EMCALTriggerBits::kDMC, true);
            }
          }
        }

      } else {
        // No trigger active
        // Set busy
        trigger.set(EMCALTriggerBits::kMB, false);
      }
    }
    // Bitset
    // Trigger sim: Select event
    if (!trigger.any()) {
      continue;
    }
    // Trigger sim: Prepare CTP input digit
    acceptedTriggers.push_back(std::make_tuple(timesview[collID], trigger));
    LOG(info) << "EMCAL TRU simulation: Sending trg = " << trigger << " to CTP";

    mDigitizer.setEventTime(timesview[collID], trigger.any());

    if (!mDigitizer.isLive()) {
      continue;
    }

    // for each collision, loop over the constituents event and source IDs
    // (background signal merging is basically taking place here)
    for (auto& part : eventParts[collID]) {

      mSumDigitizer.setCurrEvID(part.entryID);
      mSumDigitizer.setCurrSrcID(part.sourceID);

      // retrieve information about the MC collision via the MCEventHeader
      auto& mcEventHeader = mcReader->getMCEventHeader(part.sourceID, part.entryID);
      mcEventHeader.print();

      // get the hits for this event and this source
      mHits.clear();
      context->retrieveHits(mSimChains, "EMCHit", part.sourceID, part.entryID, &mHits);

      LOG(info) << "For collision " << collID << " eventID " << part.entryID << " found " << mHits.size() << " hits ";

      std::vector<o2::emcal::LabeledDigit> summedDigits;
      if (mRunSDitizer) {
        summedDigits = mSumDigitizer.process(mHits);
      } else {
        for (auto& hit : mHits) {
          o2::emcal::MCLabel digitlabel(hit.GetTrackID(), part.entryID, part.sourceID, false, 1.);
          if (hit.GetEnergyLoss() < __DBL_EPSILON__) {
            digitlabel.setAmplitudeFraction(0);
          }
          summedDigits.emplace_back(hit.GetDetectorID(), hit.GetEnergyLoss(), hit.GetTime(), digitlabel);
        }
      }

      // call actual digitization procedure
      mDigitizer.process(summedDigits);
    }
  }

  mDigitizer.finish();

  // here we have all digits and we can send them to consumer (aka snapshot it onto output)
  ctx.outputs().snapshot(Output{"EMC", "DIGITS", 0}, mDigitizer.getDigits());
  ctx.outputs().snapshot(Output{"EMC", "TRGRDIG", 0}, mDigitizer.getTriggerRecords());
  if (ctx.outputs().isAllowed({"EMC", "DIGITSMCTR", 0})) {
    ctx.outputs().snapshot(Output{"EMC", "DIGITSMCTR", 0}, mDigitizer.getMCLabels());
  }
  // EMCAL is always a triggering detector
  const o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::TRIGGERING;
  LOG(info) << "EMCAL: Sending ROMode= " << roMode << " to GRPUpdater";
  ctx.outputs().snapshot(Output{"EMC", "ROMode", 0}, roMode);
  // Create CTP digits
  std::vector<o2::ctp::CTPInputDigit> triggerinputs;
  // for (auto& trg : mDigitizer.getTriggerRecords()) {
  //   // covert TriggerRecord into CTP trigger digit
  //   o2::ctp::CTPInputDigit nextdigit;
  //   nextdigit.intRecord = trg.getBCData();
  //   nextdigit.detector = o2::detectors::DetID::EMC;
  //   // Set min. bias accept trigger (input 0) as fake trigger
  //   // Other inputs will be added once available
  //   nextdigit.inputsMask.set(0);
  //   triggerinputs.push_back(nextdigit);
  // }
  for (auto& trg : acceptedTriggers) {
    // convert TriggerRecord into CTP trigger digit
    o2::ctp::CTPInputDigit nextdigit;
    nextdigit.intRecord = std::get<0>(trg);
    nextdigit.detector = o2::detectors::DetID::EMC;
    // nextdigit.inputsMask = std::get<1>(trg);
    for (size_t i = 0; i < nextdigit.inputsMask.size(); i++)
    {
      if(std::get<1>(trg)[i] != 0) {
        nextdigit.inputsMask.set(i);
      } 
    }
    LOG(info) << "EMCAL TRU simulation: assigning = " << std::get<1>(trg)     << " to nextdigit for CTP";
    LOG(info) << "EMCAL TRU simulation: assigned  = " << nextdigit.inputsMask << " as nextdigit for CTP";
    triggerinputs.push_back(nextdigit);
  }
  ctx.outputs().snapshot(Output{"EMC", "TRIGGERINPUT", 0}, triggerinputs);

  timer.Stop();
  LOG(info) << "Digitization took " << timer.CpuTime() << "s";
  // we should be only called once; tell DPL that this process is ready to exit
  ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  mFinished = true;
}

void DigitizerSpec::configure()
{
  mDigitizer.init();
  mDigitizerTRU.init();
}

void DigitizerSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (mCalibHandler->finalizeCCDB(matcher, obj)) {
    return;
  }
  if (matcher == o2::framework::ConcreteDataMatcher("CTP", "CTPCONFIG", 0)) {
    std::cout << "Loading CTP configuration" << std::endl;
    mCTPConfig = reinterpret_cast<o2::ctp::CTPConfiguration*>(obj);
    for (const auto& trg : mCTPConfig->getCTPClasses()) {
      if (trg.cluster->maskCluster[o2::detectors::DetID::EMC]) {
        // Class triggering EMCAL cluster
        LOG(debug) << "ENABLING Trigger simulation, found trigger class for EMCAL cluster: " << trg.name << " with input mask " << std::bitset<64>(trg.descriptor->getInputsMask());
        for (const auto& [det, ctpinputs] : mCTPConfig->getDet2InputMap()) {
          if (!(det == o2::detectors::DetID::EMC)) {
            continue;
          }
          // if the detector ID is EMC AND the input mask is not empty, run the trigger simulation
          for (const auto& input : ctpinputs) {
            if (input.inputMask != 0) {
              mRunDigitizerTRU = true;
            }
          }
        }
      }
    }
  }
}

o2::framework::DataProcessorSpec getEMCALDigitizerSpec(int channel, bool requireCTPInput, bool mctruth, bool useccdb)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("EMC", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("EMC", "TRGRDIG", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back("EMC", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("EMC", "ROMode", 0, Lifetime::Timeframe);
  outputs.emplace_back("EMC", "TRIGGERINPUT", 0, Lifetime::Timeframe);

  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  std::shared_ptr<CalibLoader> calibloader;
  if (useccdb) {
    calibloader = std::make_shared<CalibLoader>();
    calibloader->enableSimParams(true);
    calibloader->defineInputSpecs(inputs);
  }
  if (requireCTPInput) {
    inputs.emplace_back("ft0inputs", "FT0", "TRIGGERINPUT", 0, Lifetime::Timeframe);
    inputs.emplace_back("fv0inputs", "FV0", "TRIGGERINPUT", 0, Lifetime::Timeframe);
    inputs.emplace_back("ctpconfig", "CTP", "CTPCONFIG", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/Config", true));
  }

  return DataProcessorSpec{
    "EMCALDigitizer", // Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}, InputSpec{"EMC_SimParam", o2::header::gDataOriginEMC, "SIMPARAM", 0, Lifetime::Condition, ccdbParamSpec("EMC/Config/SimParam")}},
    inputs,
    outputs,
    AlgorithmSpec{o2::framework::adaptFromTask<DigitizerSpec>(calibloader, requireCTPInput)},
    Options{
      {"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}}, 
      {"disable-dig-tru", VariantType::Bool, false, {"Disable TRU digitisation"}},
      {"debug-stream", VariantType::Bool, false, {"Enable debug streaming"}}}
    // I can't use VariantType::Bool as it seems to have a problem
  };
}
} // end namespace emcal
} // end namespace o2
