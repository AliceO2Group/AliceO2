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

#include "ITSMFTDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/CCDBParamSpec.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/SimTraits.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTSimulation/Digitizer.h"
#include "ITSMFTSimulation/DPLDigitizerParam.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "ITSBase/GeometryTGeo.h"
#include "MFTBase/GeometryTGeo.h"
#include <TChain.h>
#include <TStopwatch.h>
#include <string>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace itsmft
{

using namespace o2::base;
class ITSMFTDPLDigitizerTask : BaseDPLDigitizer
{
 public:
  using BaseDPLDigitizer::init;
  void initDigitizerTask(framework::InitContext& ic) override
  {
    mDisableQED = ic.options().get<bool>("disable-qed");
  }

  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    mID == o2::detectors::DetID::ITS ? updateTimeDependentParams<o2::detectors::DetID::ITS>(pc) : updateTimeDependentParams<o2::detectors::DetID::MFT>(pc);
    std::string detStr = mID.getName();
    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(mID, mSimChains);
    const bool withQED = context->isQEDProvided() && !mDisableQED;
    auto& timesview = context->getEventRecords(withQED);
    LOG(info) << "GOT " << timesview.size() << " COLLISSION TIMES";
    LOG(info) << "SIMCHAINS " << mSimChains.size();

    // if there is nothing to do ... return
    if (timesview.size() == 0) {
      return;
    }
    TStopwatch timer;
    timer.Start();
    LOG(info) << " CALLING ITS DIGITIZATION ";

    mDigitizer.setDigits(&mDigits);
    mDigitizer.setROFRecords(&mROFRecords);
    mDigitizer.setMCLabels(&mLabels);

    // digits are directly put into DPL owned resource
    auto& digitsAccum = pc.outputs().make<std::vector<itsmft::Digit>>(Output{mOrigin, "DIGITS", 0});

    auto accumulate = [this, &digitsAccum]() {
      // accumulate result of single event processing, called after processing every event supplied
      // AND after the final flushing via digitizer::fillOutputContainer
      if (!mDigits.size()) {
        return; // no digits were flushed, nothing to accumulate
      }
      static int fixMC2ROF = 0; // 1st entry in mc2rofRecordsAccum to be fixed for ROFRecordID
      auto ndigAcc = digitsAccum.size();
      std::copy(mDigits.begin(), mDigits.end(), std::back_inserter(digitsAccum));

      // fix ROFrecords references on ROF entries
      auto nROFRecsOld = mROFRecordsAccum.size();

      for (int i = 0; i < mROFRecords.size(); i++) {
        auto& rof = mROFRecords[i];
        rof.setFirstEntry(ndigAcc + rof.getFirstEntry());
        rof.print();

        if (mFixMC2ROF < mMC2ROFRecordsAccum.size()) { // fix ROFRecord entry in MC2ROF records
          for (int m2rid = mFixMC2ROF; m2rid < mMC2ROFRecordsAccum.size(); m2rid++) {
            // need to register the ROFRecors entry for MC event starting from this entry
            auto& mc2rof = mMC2ROFRecordsAccum[m2rid];
            if (rof.getROFrame() == mc2rof.minROF) {
              mFixMC2ROF++;
              mc2rof.rofRecordID = nROFRecsOld + i;
              mc2rof.print();
            }
          }
        }
      }

      std::copy(mROFRecords.begin(), mROFRecords.end(), std::back_inserter(mROFRecordsAccum));
      if (mWithMCTruth) {
        mLabelsAccum.mergeAtBack(mLabels);
      }
      LOG(info) << "Added " << mDigits.size() << " digits ";
      // clean containers from already accumulated stuff
      mLabels.clear();
      mDigits.clear();
      mROFRecords.clear();
    }; // and accumulate lambda

    auto& eventParts = context->getEventParts(withQED);
    int bcShift = mDigitizer.getParams().getROFrameBiasInBC();
    // loop over all composite collisions given from context (aka loop over all the interaction records)
    for (int collID = 0; collID < timesview.size(); ++collID) {
      auto irt = timesview[collID];
      if (irt.toLong() < bcShift) { // due to the ROF misalignment the collision would go to negative ROF ID, discard
        continue;
      }
      irt -= bcShift; // account for the ROF start shift

      mDigitizer.setEventTime(irt);
      mDigitizer.resetEventROFrames(); // to estimate min/max ROF for this collID
      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {

        // get the hits for this event and this source
        mHits.clear();
        context->retrieveHits(mSimChains, o2::detectors::SimTraits::DETECTORBRANCHNAMES[mID][0].c_str(), part.sourceID, part.entryID, &mHits);

        if (mHits.size() > 0) {
          LOG(debug) << "For collision " << collID << " eventID " << part.entryID
                     << " found " << mHits.size() << " hits ";
          mDigitizer.process(&mHits, part.entryID, part.sourceID); // call actual digitization procedure
        }
      }
      mMC2ROFRecordsAccum.emplace_back(collID, -1, mDigitizer.getEventROFrameMin(), mDigitizer.getEventROFrameMax());
      accumulate();
    }
    mDigitizer.fillOutputContainer();
    accumulate();

    // here we have all digits and labels and we can send them to consumer (aka snapshot it onto output)

    pc.outputs().snapshot(Output{mOrigin, "DIGITSROF", 0}, mROFRecordsAccum);
    if (mWithMCTruth) {
      pc.outputs().snapshot(Output{mOrigin, "DIGITSMC2ROF", 0}, mMC2ROFRecordsAccum);
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{mOrigin, "DIGITSMCTR", 0});
      mLabelsAccum.flatten_to(sharedlabels);
      // free space of existing label containers
      mLabels.clear_andfreememory();
      mLabelsAccum.clear_andfreememory();
    }
    LOG(info) << mID.getName() << ": Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{mOrigin, "ROMode", 0}, mROMode);

    timer.Stop();
    LOG(info) << "Digitization took " << timer.CpuTime() << "s";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);

    mFinished = true;
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    if (matcher == ConcreteDataMatcher(mOrigin, "NOISEMAP", 0)) {
      LOG(info) << mID.getName() << " noise map updated";
      mDigitizer.setNoiseMap((const o2::itsmft::NoiseMap*)obj);
      return;
    }
    if (matcher == ConcreteDataMatcher(mOrigin, "DEADMAP", 0)) {
      LOG(info) << mID.getName() << " dead map updated";
      mDigitizer.setDeadChannelsMap((const o2::itsmft::NoiseMap*)obj);
      return;
    }
    if (matcher == ConcreteDataMatcher(mOrigin, "ALPIDEPARAM", 0)) {
      LOG(info) << mID.getName() << " Alpide param updated";
      if (mID == o2::detectors::DetID::ITS) {
        const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
        par.printKeyValues();
      } else {
        const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
        par.printKeyValues();
      }
      return;
    }
  }

 protected:
  ITSMFTDPLDigitizerTask(bool mctruth = true) : BaseDPLDigitizer(InitServices::FIELD | InitServices::GEOM), mWithMCTruth(mctruth) {}

  template <int DETID>
  void updateTimeDependentParams(ProcessingContext& pc)
  {
    std::string detstr(o2::detectors::DetID::getName(DETID));
    pc.inputs().get<o2::itsmft::NoiseMap*>(detstr + "_noise");
    pc.inputs().get<o2::itsmft::NoiseMap*>(detstr + "_dead");
    pc.inputs().get<o2::itsmft::DPLAlpideParam<DETID>*>(detstr + "_alppar");

    auto& dopt = o2::itsmft::DPLDigitizerParam<DETID>::Instance();
    auto& aopt = o2::itsmft::DPLAlpideParam<DETID>::Instance();
    auto& digipar = mDigitizer.getParams();
    digipar.setContinuous(dopt.continuous);
    digipar.setROFrameBiasInBC(aopt.roFrameBiasInBC);
    if (dopt.continuous) {
      auto frameNS = aopt.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS;
      digipar.setROFrameLengthInBC(aopt.roFrameLengthInBC);
      digipar.setROFrameLength(frameNS);                                                                       // RO frame in ns
      digipar.setStrobeDelay(aopt.strobeDelay);                                                                // Strobe delay wrt beginning of the RO frame, in ns
      digipar.setStrobeLength(aopt.strobeLengthCont > 0 ? aopt.strobeLengthCont : frameNS - aopt.strobeDelay); // Strobe length in ns
    } else {
      digipar.setROFrameLength(aopt.roFrameLengthTrig); // RO frame in ns
      digipar.setStrobeDelay(aopt.strobeDelay);         // Strobe delay wrt beginning of the RO frame, in ns
      digipar.setStrobeLength(aopt.strobeLengthTrig);   // Strobe length in ns
    }
    // parameters of signal time response: flat-top duration, max rise time and q @ which rise time is 0
    digipar.getSignalShape().setParameters(dopt.strobeFlatTop, dopt.strobeMaxRiseTime, dopt.strobeQRiseTime0);
    digipar.setChargeThreshold(dopt.chargeThreshold); // charge threshold in electrons
    digipar.setNoisePerPixel(dopt.noisePerPixel);     // noise level
    digipar.setTimeOffset(dopt.timeOffset);
    digipar.setNSimSteps(dopt.nSimSteps);
    digipar.setIBVbb(dopt.IBVbb);
    digipar.setOBVbb(dopt.OBVbb);
    digipar.setVbb(dopt.Vbb);

    mROMode = digipar.isContinuous() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
    LOG(info) << mID.getName() << " simulated in "
              << ((mROMode == o2::parameters::GRPObject::CONTINUOUS) ? "CONTINUOUS" : "TRIGGERED")
              << " RO mode";

    // configure digitizer
    o2::itsmft::GeometryTGeo* geom = nullptr;
    if (mID == o2::detectors::DetID::ITS) {
      geom = o2::its::GeometryTGeo::Instance();
    } else {
      geom = o2::mft::GeometryTGeo::Instance();
    }
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G)); // make sure L2G matrices are loaded
    mDigitizer.setGeometry(geom);
    mDigitizer.init();
  }

  bool mWithMCTruth = true;
  bool mFinished = false;
  bool mDisableQED = false;
  o2::detectors::DetID mID;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginInvalid;
  o2::itsmft::Digitizer mDigitizer;
  std::vector<o2::itsmft::Digit> mDigits;
  std::vector<o2::itsmft::ROFRecord> mROFRecords;
  std::vector<o2::itsmft::ROFRecord> mROFRecordsAccum;
  std::vector<o2::itsmft::Hit> mHits;
  std::vector<o2::itsmft::Hit>* mHitsP = &mHits;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabels;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabelsAccum;
  std::vector<o2::itsmft::MC2ROFRecord> mMC2ROFRecordsAccum;
  std::vector<TChain*> mSimChains;

  int mFixMC2ROF = 0;                                                             // 1st entry in mc2rofRecordsAccum to be fixed for ROFRecordID
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::PRESENT; // readout mode
};

//_______________________________________________
class ITSDPLDigitizerTask : public ITSMFTDPLDigitizerTask
{
 public:
  // FIXME: origin should be extractable from the DetID, the problem is 3d party header dependencies
  static constexpr o2::detectors::DetID::ID DETID = o2::detectors::DetID::ITS;
  static constexpr o2::header::DataOrigin DETOR = o2::header::gDataOriginITS;
  ITSDPLDigitizerTask(bool mctruth = true) : ITSMFTDPLDigitizerTask(mctruth)
  {
    mID = DETID;
    mOrigin = DETOR;
  }
};

constexpr o2::detectors::DetID::ID ITSDPLDigitizerTask::DETID;
constexpr o2::header::DataOrigin ITSDPLDigitizerTask::DETOR;

//_______________________________________________
class MFTDPLDigitizerTask : public ITSMFTDPLDigitizerTask
{
 public:
  // FIXME: origina should be extractable from the DetID, the problem is 3d party header dependencies
  static constexpr o2::detectors::DetID::ID DETID = o2::detectors::DetID::MFT;
  static constexpr o2::header::DataOrigin DETOR = o2::header::gDataOriginMFT;
  MFTDPLDigitizerTask(bool mctruth) : ITSMFTDPLDigitizerTask(mctruth)
  {
    mID = DETID;
    mOrigin = DETOR;
  }
};

constexpr o2::detectors::DetID::ID MFTDPLDigitizerTask::DETID;
constexpr o2::header::DataOrigin MFTDPLDigitizerTask::DETOR;

std::vector<OutputSpec> makeOutChannels(o2::header::DataOrigin detOrig, bool mctruth)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(detOrig, "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back(detOrig, "DIGITSROF", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back(detOrig, "DIGITSMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back(detOrig, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(detOrig, "ROMode", 0, Lifetime::Timeframe);
  return outputs;
}

DataProcessorSpec getITSDigitizerSpec(int channel, bool mctruth)
{
  std::string detStr = o2::detectors::DetID::getName(ITSDPLDigitizerTask::DETID);
  auto detOrig = ITSDPLDigitizerTask::DETOR;
  std::stringstream parHelper;
  parHelper << "Params as " << o2::itsmft::DPLDigitizerParam<ITSDPLDigitizerTask::DETID>::getParamName().data() << ".<param>=value;... with"
            << o2::itsmft::DPLDigitizerParam<ITSDPLDigitizerTask::DETID>::Instance()
            << "\n or " << o2::itsmft::DPLAlpideParam<ITSDPLDigitizerTask::DETID>::getParamName().data() << ".<param>=value;... with"
            << o2::itsmft::DPLAlpideParam<ITSDPLDigitizerTask::DETID>::Instance();
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  inputs.emplace_back("ITS_noise", "ITS", "NOISEMAP", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/NoiseMap"));
  inputs.emplace_back("ITS_dead", "ITS", "DEADMAP", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/DeadMap"));
  inputs.emplace_back("ITS_alppar", "ITS", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam"));

  return DataProcessorSpec{(detStr + "Digitizer").c_str(),
                           inputs, makeOutChannels(detOrig, mctruth),
                           AlgorithmSpec{adaptFromTask<ITSDPLDigitizerTask>(mctruth)},
                           Options{
                             {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}}}};
}

DataProcessorSpec getMFTDigitizerSpec(int channel, bool mctruth)
{
  std::string detStr = o2::detectors::DetID::getName(MFTDPLDigitizerTask::DETID);
  auto detOrig = MFTDPLDigitizerTask::DETOR;
  std::stringstream parHelper;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  inputs.emplace_back("MFT_noise", "MFT", "NOISEMAP", 0, Lifetime::Condition, ccdbParamSpec("MFT/Calib/NoiseMap"));
  inputs.emplace_back("MFT_dead", "MFT", "DEADMAP", 0, Lifetime::Condition, ccdbParamSpec("MFT/Calib/DeadMap"));
  inputs.emplace_back("MFT_alppar", "MFT", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("MFT/Config/AlpideParam"));
  parHelper << "Params as " << o2::itsmft::DPLDigitizerParam<ITSDPLDigitizerTask::DETID>::getParamName().data() << ".<param>=value;... with"
            << o2::itsmft::DPLDigitizerParam<ITSDPLDigitizerTask::DETID>::Instance()
            << " or " << o2::itsmft::DPLAlpideParam<ITSDPLDigitizerTask::DETID>::getParamName().data() << ".<param>=value;... with"
            << o2::itsmft::DPLAlpideParam<ITSDPLDigitizerTask::DETID>::Instance();
  return DataProcessorSpec{(detStr + "Digitizer").c_str(),
                           inputs, makeOutChannels(detOrig, mctruth),
                           AlgorithmSpec{adaptFromTask<MFTDPLDigitizerTask>(mctruth)},
                           Options{{"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}}}};
}

} // end namespace itsmft
} // end namespace o2
