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

#include "TRKDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Steer/HitProcessingManager.h"
#include "DataFormatsITSMFT/Digit.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/SimTraits.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "TRKSimulation/Digitizer.h"
#include "TRKSimulation/DPLDigitizerParam.h"
#include "TRKBase/AlmiraParam.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKBase/Specs.h"
#include "TRKBase/TRKBaseParam.h"

#include <TChain.h>
#include <TStopwatch.h>

#include <algorithm>
#include <memory>
#include <string>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace
{
std::vector<OutputSpec> makeOutChannels(o2::header::DataOrigin detOrig, bool mctruth)
{
  std::vector<OutputSpec> outputs;
  for (uint32_t iLayer = 0; iLayer < o2::trk::AlmiraParam::getNLayers(); ++iLayer) {
    outputs.emplace_back(detOrig, "DIGITS", iLayer, Lifetime::Timeframe);
    outputs.emplace_back(detOrig, "DIGITSROF", iLayer, Lifetime::Timeframe);
    if (mctruth) {
      outputs.emplace_back(detOrig, "DIGITSMC2ROF", iLayer, Lifetime::Timeframe);
      outputs.emplace_back(detOrig, "DIGITSMCTR", iLayer, Lifetime::Timeframe);
    }
  }
  outputs.emplace_back(detOrig, "ROMode", 0, Lifetime::Timeframe);
  return outputs;
}
} // namespace

namespace o2::trk
{
using namespace o2::base;
class TRKDPLDigitizerTask : BaseDPLDigitizer
{
 public:
  using BaseDPLDigitizer::init;

  TRKDPLDigitizerTask(bool mctruth = true) : BaseDPLDigitizer(InitServices::FIELD | InitServices::GEOM), mWithMCTruth(mctruth) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
    mDisableQED = ic.options().get<bool>("disable-qed");
    mLocalRespFile = ic.options().get<std::string>("local-response-file");
  }

  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    mFirstOrbitTF = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
    const o2::InteractionRecord firstIR(0, mFirstOrbitTF);
    updateTimeDependentParams(pc);

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(mID, mSimChains);
    const bool withQED = context->isQEDProvided() && !mDisableQED;
    auto& timesview = context->getEventRecords(withQED);
    LOG(info) << "GOT " << timesview.size() << " COLLISION TIMES";
    LOG(info) << "SIMCHAINS " << mSimChains.size();

    // if there is nothing to do ... return
    if (timesview.empty()) {
      return;
    }
    TStopwatch timer;
    timer.Start();
    LOG(info) << " CALLING TRK DIGITIZATION ";

    auto& eventParts = context->getEventParts(withQED);
    uint64_t nDigits{0};
    for (uint32_t iLayer = 0; iLayer < static_cast<uint32_t>(mLayers); ++iLayer) {
      mDigits[iLayer].clear();
      mROFRecords[iLayer].clear();
      mROFRecordsAccum[iLayer].clear();
      if (mWithMCTruth) {
        mLabels[iLayer].clear();
        mLabelsAccum[iLayer].clear();
        mMC2ROFRecordsAccum[iLayer].clear();
      }

      mDigitizer.setDigits(&mDigits[iLayer]);
      mDigitizer.setROFRecords(&mROFRecords[iLayer]);
      mDigitizer.setMCLabels(&mLabels[iLayer]);
      mDigitizer.resetROFrameBounds();

      // digits are directly put into DPL owned resource
      auto& digitsAccum = pc.outputs().make<std::vector<itsmft::Digit>>(Output{mOrigin, "DIGITS", iLayer});

      const int roFrameLengthInBC = mDigitizer.getParams().getROFrameLengthInBC(iLayer);
      const int nROFsPerOrbit = o2::constants::lhc::LHCMaxBunches / roFrameLengthInBC;
      const int nROFsTF = nROFsPerOrbit * raw::HBFUtils::Instance().getNOrbitsPerTF();
      mROFRecordsAccum[iLayer].reserve(nROFsTF);

      auto accumulate = [this, &digitsAccum, &iLayer]() {
        // accumulate result of single event processing on one layer, called after each collision
        // and after the final flushing via digitizer::fillOutputContainer
        if (mDigits[iLayer].empty()) {
          return;
        }
        auto ndigAcc = digitsAccum.size();
        std::copy(mDigits[iLayer].begin(), mDigits[iLayer].end(), std::back_inserter(digitsAccum));

        for (auto& rof : mROFRecords[iLayer]) {
          rof.setFirstEntry(ndigAcc + rof.getFirstEntry());
        }

        std::copy(mROFRecords[iLayer].begin(), mROFRecords[iLayer].end(), std::back_inserter(mROFRecordsAccum[iLayer]));
        if (mWithMCTruth) {
          mLabelsAccum[iLayer].mergeAtBack(mLabels[iLayer]);
        }
        LOG(info) << "Added " << mDigits[iLayer].size() << " digits on layer " << iLayer;
        mLabels[iLayer].clear();
        mDigits[iLayer].clear();
        mROFRecords[iLayer].clear();
      };

      const int bcShift = mDigitizer.getParams().getROFrameBiasInBC(iLayer);
      for (size_t collID = 0; collID < timesview.size(); ++collID) {
        auto irt = timesview[collID];
        if (irt.toLong() < bcShift) {
          continue;
        }
        irt -= bcShift;

        mDigitizer.setEventTime(irt, iLayer);
        mDigitizer.resetEventROFrames();
        for (auto& part : eventParts[collID]) {
          mHits.clear();
          context->retrieveHits(mSimChains, o2::detectors::SimTraits::DETECTORBRANCHNAMES[mID][0].c_str(), part.sourceID, part.entryID, &mHits);

          if (!mHits.empty()) {
            LOG(debug) << "For collision " << collID << " eventID " << part.entryID
                       << " found " << mHits.size() << " hits on layer " << iLayer;
            mDigitizer.process(&mHits, part.entryID, part.sourceID, iLayer);
          }
        }
        if (mWithMCTruth) {
          mMC2ROFRecordsAccum[iLayer].emplace_back(collID, -1, mDigitizer.getEventROFrameMin(), mDigitizer.getEventROFrameMax());
        }
        accumulate();
      }
      mDigitizer.fillOutputContainer(0xffffffff, iLayer);
      accumulate();
      nDigits += digitsAccum.size();

      std::vector<o2::itsmft::ROFRecord> expDigitRofVec(nROFsTF);
      for (int iROF = 0; iROF < nROFsTF; ++iROF) {
        auto& rof = expDigitRofVec[iROF];
        const int orb = iROF * roFrameLengthInBC / o2::constants::lhc::LHCMaxBunches + mFirstOrbitTF;
        const int bc = iROF * roFrameLengthInBC % o2::constants::lhc::LHCMaxBunches;
        rof.setBCData(o2::InteractionRecord(bc, orb));
        rof.setROFrame(iROF);
        rof.setNEntries(0);
        rof.setFirstEntry(-1);
      }

      for (const auto& rof : mROFRecordsAccum[iLayer]) {
        const auto& ir = rof.getBCData();
        const auto irToFirst = ir - firstIR;
        const auto irROF = irToFirst.toLong() / roFrameLengthInBC;
        if (irROF < 0 || irROF >= nROFsTF) {
          continue;
        }
        auto& expROF = expDigitRofVec[irROF];
        expROF.setFirstEntry(rof.getFirstEntry());
        expROF.setNEntries(rof.getNEntries());
        if (expROF.getBCData() != rof.getBCData()) {
          LOGP(fatal, "detected mismatch between expected {} and received {}", expROF.asString(), rof.asString());
        }
      }

      int prevFirst = 0;
      for (auto& rof : expDigitRofVec) {
        if (rof.getFirstEntry() < 0) {
          rof.setFirstEntry(prevFirst);
        }
        prevFirst = rof.getFirstEntry();
      }

      pc.outputs().snapshot(Output{mOrigin, "DIGITSROF", iLayer}, expDigitRofVec);
      if (mWithMCTruth) {
        std::vector<o2::itsmft::MC2ROFRecord> clippedMC2ROFRecords;
        clippedMC2ROFRecords.reserve(mMC2ROFRecordsAccum[iLayer].size());
        for (auto mc2rof : mMC2ROFRecordsAccum[iLayer]) {
          if (mc2rof.minROF >= static_cast<uint32_t>(nROFsTF) || mc2rof.minROF > mc2rof.maxROF) {
            mc2rof.rofRecordID = -1;
            mc2rof.minROF = 0;
            mc2rof.maxROF = 0;
          } else {
            mc2rof.maxROF = std::min<uint32_t>(mc2rof.maxROF, nROFsTF - 1);
            if (mc2rof.minROF > mc2rof.maxROF) {
              mc2rof.rofRecordID = -1;
              mc2rof.minROF = 0;
              mc2rof.maxROF = 0;
            } else {
              mc2rof.rofRecordID = mc2rof.minROF;
            }
          }
          clippedMC2ROFRecords.push_back(mc2rof);
        }
        pc.outputs().snapshot(Output{mOrigin, "DIGITSMC2ROF", iLayer}, clippedMC2ROFRecords);
        auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{mOrigin, "DIGITSMCTR", iLayer});
        mLabelsAccum[iLayer].flatten_to(sharedlabels);
        mLabels[iLayer].clear_andfreememory();
        mLabelsAccum[iLayer].clear_andfreememory();
      }
    }
    LOG(info) << mID.getName() << ": Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{mOrigin, "ROMode", 0}, mROMode);

    timer.Stop();
    LOG(info) << "Digitization took " << timer.CpuTime() << "s";
    LOG(info) << "Produced " << nDigits << " digits";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);

    mFinished = true;
  }

  void setLocalResponseFunction()
  {
    std::unique_ptr<TFile> file(TFile::Open(mLocalRespFile.data(), "READ"));
    if (!file) {
      LOG(fatal) << "Cannot open response file " << mLocalRespFile;
    }
    mDigitizer.getParams().setResponse((const o2::itsmft::AlpideSimResponse*)file->Get("response1"));
  }

  void updateTimeDependentParams(ProcessingContext& pc)
  {
    static bool initOnce{false};
    if (!initOnce) {
      initOnce = true;
      auto& digipar = mDigitizer.getParams();

      // configure digitizer
      o2::trk::GeometryTGeo* geom = o2::trk::GeometryTGeo::Instance();
      geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G)); // make sure L2G matrices are loaded
      geom->Print();
      mDigitizer.setGeometry(geom);

      const auto& dopt = o2::trk::DPLDigitizerParam<o2::detectors::DetID::TRK>::Instance();
      // pc.inputs().get<o2::trk::AlmiraParam*>("TRK_almiraparam");
      const auto& aopt = o2::trk::AlmiraParam::Instance();
      mLayers = constants::VD::petal::nLayers + geom->getNumberOfLayersMLOT();
      mDigits.resize(mLayers);
      mROFRecords.resize(mLayers);
      mROFRecordsAccum.resize(mLayers);
      mLabels.resize(mLayers);
      mLabelsAccum.resize(mLayers);
      mMC2ROFRecordsAccum.resize(mLayers);

      for (int iLayer = 0; iLayer < mLayers; ++iLayer) {
        const auto roFrameLengthInBC = aopt.getROFLengthInBC(iLayer);
        const auto frameNS = roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS;
        digipar.setROFrameLengthInBC(roFrameLengthInBC, iLayer);
        // ROF delay is treated as an additional bias from the digitizer point of view.
        digipar.setROFrameBiasInBC(aopt.getROFBiasInBC(iLayer) + aopt.getROFDelayInBC(iLayer), iLayer);
        digipar.setStrobeDelay(aopt.getStrobeDelay(iLayer), iLayer);
        const auto strobeLengthCont = aopt.getStrobeLengthCont(iLayer);
        digipar.setStrobeLength(strobeLengthCont > 0 ? strobeLengthCont : frameNS - aopt.getStrobeDelay(iLayer), iLayer);
        digipar.setROFrameLength(frameNS, iLayer);
      }
      // parameters of signal time response: flat-top duration, max rise time and q @ which rise time is 0
      digipar.getSignalShape().setParameters(dopt.strobeFlatTop, dopt.strobeMaxRiseTime, dopt.strobeQRiseTime0);
      digipar.setChargeThreshold(dopt.chargeThreshold); // charge threshold in electrons
      digipar.setNoisePerPixel(dopt.noisePerPixel);     // noise level
      digipar.setTimeOffset(dopt.timeOffset);
      digipar.setNSimSteps(dopt.nSimSteps);

      mROMode = o2::parameters::GRPObject::CONTINUOUS;
      LOG(info) << mID.getName() << " simulated in CONTINUOUS RO mode";

      // if (oTRKParams::Instance().useDeadChannelMap) {
      //   pc.inputs().get<o2::itsmft::NoiseMap*>("TRK_dead"); // trigger final ccdb update
      // }
      pc.inputs().get<o2::itsmft::AlpideSimResponse*>("TRK_aptsresp");

      // init digitizer
      mDigitizer.init();
    }
    // Other time-dependent parameters can be added below
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    if (matcher == ConcreteDataMatcher(mOrigin, "ALMIRAPARAM", 0)) {
      LOG(info) << mID.getName() << " Almira param updated";
      const auto& par = o2::trk::AlmiraParam::Instance();
      par.printKeyValues();
      return;
    }
    // if (matcher == ConcreteDataMatcher(mOrigin, "DEADMAP", 0)) {
    //   LOG(info) << mID.getName() << " static dead map updated";
    //   mDigitizer.setDeadChannelsMap((o2::itsmft::NoiseMap*)obj);
    //   return;
    // }
    if (matcher == ConcreteDataMatcher(mOrigin, "APTSRESP", 0)) {
      LOG(info) << mID.getName() << " loaded APTSResponseData";
      if (mLocalRespFile.empty()) {
        LOG(info) << "Using CCDB/APTS response file";
        mDigitizer.getParams().setResponse((const o2::itsmft::AlpideSimResponse*)obj);
        mDigitizer.setResponseName("APTS");
      } else {
        LOG(info) << "Response function will be loaded from local file: " << mLocalRespFile;
        setLocalResponseFunction();
        mDigitizer.setResponseName("ALICE3");
      }
    }
  }

 private:
  bool mWithMCTruth{true};
  bool mFinished{false};
  bool mDisableQED{false};
  unsigned long mFirstOrbitTF = 0x0;
  std::string mLocalRespFile{""};
  const o2::detectors::DetID mID{o2::detectors::DetID::TRK};
  const o2::header::DataOrigin mOrigin{o2::header::gDataOriginTRK};
  o2::trk::Digitizer mDigitizer{};
  int mLayers{0};
  std::vector<std::vector<o2::itsmft::Digit>> mDigits{};
  std::vector<std::vector<o2::itsmft::ROFRecord>> mROFRecords{};
  std::vector<std::vector<o2::itsmft::ROFRecord>> mROFRecordsAccum{};
  std::vector<o2::trk::Hit> mHits{};
  std::vector<o2::trk::Hit>* mHitsP{&mHits};
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabels{};
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabelsAccum{};
  std::vector<std::vector<o2::itsmft::MC2ROFRecord>> mMC2ROFRecordsAccum{};
  std::vector<TChain*> mSimChains{};
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::PRESENT; // readout mode
};

DataProcessorSpec getTRKDigitizerSpec(int channel, bool mctruth)
{
  std::string detStr = o2::detectors::DetID::getName(o2::detectors::DetID::TRK);
  auto detOrig = o2::header::gDataOriginTRK;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  // inputs.emplace_back("TRK_almiraparam", "TRK", "ALMIRAPARAM", 0, Lifetime::Condition, ccdbParamSpec("TRK/Config/AlmiraParam"));
  // if (oTRKParams::Instance().useDeadChannelMap) {
  //   inputs.emplace_back("TRK_dead", "TRK", "DEADMAP", 0, Lifetime::Condition, ccdbParamSpec("TRK/Calib/DeadMap"));
  // }
  inputs.emplace_back("TRK_aptsresp", "TRK", "APTSRESP", 0, Lifetime::Condition, ccdbParamSpec("IT3/Calib/APTSResponse"));

  return DataProcessorSpec{detStr + "Digitizer",
                           inputs, makeOutChannels(detOrig, mctruth),
                           AlgorithmSpec{adaptFromTask<TRKDPLDigitizerTask>(mctruth)},
                           Options{
                             {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}},
                             {"local-response-file", o2::framework::VariantType::String, "", {"use response file saved locally at this path/filename"}}}};
}

} // namespace o2::trk
