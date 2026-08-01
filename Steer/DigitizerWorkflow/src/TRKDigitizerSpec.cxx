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
#include "DataFormatsTRKFT3/Digit.h"
#include "DataFormatsTRKFT3/Hit.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/SimTraits.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsTRKFT3/ROFRecord.h"
#include "TRKFT3Simulation/Digitizer.h"
#include "TRKFT3Simulation/DPLDigitizerParam.h"
#include "FT3Base/FT3BaseParam.h"
#include "FT3Base/GeometryTGeo.h"
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
std::vector<OutputSpec> makeOutChannels(o2::header::DataOrigin detOrig, int nLayers, bool mctruth)
{
  std::vector<OutputSpec> outputs;
  for (uint32_t iLayer = 0; iLayer < static_cast<uint32_t>(nLayers); ++iLayer) {
    outputs.emplace_back(detOrig, "DIGITS", iLayer, Lifetime::Timeframe);
    outputs.emplace_back(detOrig, "DIGITSROF", iLayer, Lifetime::Timeframe);
    if (mctruth) {
      outputs.emplace_back(detOrig, "DIGITSMCTR", iLayer, Lifetime::Timeframe);
    }
  }
  outputs.emplace_back(detOrig, "ROMode", 0, Lifetime::Timeframe);
  return outputs;
}
} // namespace

namespace o2::trkft3
{
using namespace o2::base;

template <int N>
int getNLayers()
{
  if constexpr (N == o2::detectors::DetID::TRK) {
    return o2::trk::AlmiraParam::getNLayers();
  } else {
    return 2 * o2::ft3::FT3BaseParam::Instance().nLayers;
  }
}

template <int N>
class TRKFT3DPLDigitizerTask : BaseDPLDigitizer
{
 public:
  static_assert(N == o2::detectors::DetID::TRK || N == o2::detectors::DetID::FT3, "only TRK and FT3 digitizers are supported");
  static constexpr o2::detectors::DetID ID{N == o2::detectors::DetID::TRK ? o2::detectors::DetID::TRK : o2::detectors::DetID::FT3};
  static constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::TRK ? o2::header::gDataOriginTRK : o2::header::gDataOriginFT3};
  using BaseDPLDigitizer::init;

  TRKFT3DPLDigitizerTask(bool mctruth = true) : BaseDPLDigitizer(InitServices::FIELD | InitServices::GEOM), mWithMCTruth(mctruth) {}

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
    context->initSimChains(ID, mSimChains);
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
    LOG(info) << " CALLING " << ID.getName() << " DIGITIZATION ";

    auto& eventParts = context->getEventParts(withQED);
    uint64_t nDigits{0};
    for (uint32_t iLayer = 0; iLayer < static_cast<uint32_t>(mLayers); ++iLayer) {
      mDigits[iLayer].clear();
      mROFRecords[iLayer].clear();
      mROFRecordsAccum[iLayer].clear();
      if (mWithMCTruth) {
        mLabels[iLayer].clear();
        mLabelsAccum[iLayer].clear();
      }

      mDigitizer.setDigits(&mDigits[iLayer]);
      mDigitizer.setROFRecords(&mROFRecords[iLayer]);
      mDigitizer.setMCLabels(&mLabels[iLayer]);
      mDigitizer.resetROFrameBounds();

      // digits are directly put into DPL owned resource
      auto& digitsAccum = pc.outputs().make<std::vector<trkft3::Digit>>(Output{Origin, "DIGITS", iLayer});

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
          context->retrieveHits(mSimChains, o2::detectors::SimTraits::DETECTORBRANCHNAMES[ID][0].c_str(), part.sourceID, part.entryID, &mHits);

          if (!mHits.empty()) {
            LOG(debug) << "For collision " << collID << " eventID " << part.entryID
                       << " found " << mHits.size() << " hits on layer " << iLayer;
            mDigitizer.process(&mHits, part.entryID, part.sourceID, iLayer);
          }
        }
        accumulate();
      }
      mDigitizer.fillOutputContainer(0xffffffff, iLayer);
      accumulate();
      nDigits += digitsAccum.size();

      std::vector<o2::trkft3::ROFRecord> expDigitRofVec(nROFsTF);
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

      pc.outputs().snapshot(Output{Origin, "DIGITSROF", iLayer}, expDigitRofVec);
      if (mWithMCTruth) {
        auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{Origin, "DIGITSMCTR", iLayer});
        mLabelsAccum[iLayer].flatten_to(sharedlabels);
        mLabels[iLayer].clear_andfreememory();
        mLabelsAccum[iLayer].clear_andfreememory();
      }
    }
    LOG(info) << ID.getName() << ": Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{Origin, "ROMode", 0}, mROMode);

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

      const auto& dopt = o2::trkft3::DPLDigitizerParam<N>::Instance();
      const auto& aopt = o2::trk::AlmiraParam::Instance();
      if constexpr (N == o2::detectors::DetID::TRK) {
        auto* geom = o2::trk::GeometryTGeo::Instance();
        geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));
        geom->Print();
        mDigitizer.setGeometry(geom);
        mLayers = o2::trk::AlmiraParam::getNLayers();
      } else {
        auto* geom = o2::ft3::GeometryTGeo::Instance();
        geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));
        geom->Print();
        mDigitizer.setGeometry(geom);
        mLayers = geom->getNumberOfLayers();
      }
      if (mLayers > static_cast<int>(o2::trkft3::DigiParams<N>::getMaxLayers())) {
        LOGP(fatal, "{} geometry has {} layers, but DigiParams supports at most {}", ID.getName(), mLayers, o2::trkft3::DigiParams<N>::getMaxLayers());
      }
      mDigits.resize(mLayers);
      mROFRecords.resize(mLayers);
      mROFRecordsAccum.resize(mLayers);
      mLabels.resize(mLayers);
      mLabelsAccum.resize(mLayers);

      for (int iLayer = 0; iLayer < mLayers; ++iLayer) {
        const int parLayer = std::min<int>(iLayer, o2::trk::AlmiraParam::getNLayers() - 1);
        const auto roFrameLengthInBC = aopt.getROFLengthInBC(parLayer);
        const auto frameNS = roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS;
        digipar.setROFrameLengthInBC(roFrameLengthInBC, iLayer);
        // ROF delay is treated as an additional bias from the digitizer point of view.
        digipar.setROFrameBiasInBC(aopt.getROFBiasInBC(parLayer) + aopt.getROFDelayInBC(parLayer), iLayer);
        digipar.setStrobeDelay(aopt.getStrobeDelay(parLayer), iLayer);
        const auto strobeLengthCont = aopt.getStrobeLengthCont(parLayer);
        digipar.setStrobeLength(strobeLengthCont > 0 ? strobeLengthCont : frameNS - aopt.getStrobeDelay(parLayer), iLayer);
        digipar.setROFrameLength(frameNS, iLayer);
      }
      // parameters of signal time response: flat-top duration, max rise time and q @ which rise time is 0
      digipar.getSignalShape().setParameters(dopt.strobeFlatTop, dopt.strobeMaxRiseTime, dopt.strobeQRiseTime0);
      digipar.setChargeThreshold(dopt.chargeThreshold); // charge threshold in electrons
      digipar.setNoisePerPixel(dopt.noisePerPixel);     // noise level
      digipar.setTimeOffset(dopt.timeOffset);
      digipar.setNSimSteps(dopt.nSimSteps);

      mROMode = o2::parameters::GRPObject::CONTINUOUS;
      LOG(info) << ID.getName() << " simulated in CONTINUOUS RO mode";

      // if (oTRKParams::Instance().useDeadChannelMap) {
      //   pc.inputs().get<o2::itsmft::NoiseMap*>("TRK_dead"); // trigger final ccdb update
      // }
      pc.inputs().get<o2::itsmft::AlpideSimResponse*>((std::string(ID.getName()) + "_aptsresp").c_str());

      // init digitizer
      mDigitizer.init();
    }
    // Other time-dependent parameters can be added below
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    if (matcher == ConcreteDataMatcher(Origin, "ALMIRAPARAM", 0)) {
      LOG(info) << ID.getName() << " Almira param updated";
      const auto& par = o2::trk::AlmiraParam::Instance();
      par.printKeyValues();
      return;
    }
    // if (matcher == ConcreteDataMatcher(mOrigin, "DEADMAP", 0)) {
    //   LOG(info) << mID.getName() << " static dead map updated";
    //   mDigitizer.setDeadChannelsMap((o2::itsmft::NoiseMap*)obj);
    //   return;
    // }
    if (matcher == ConcreteDataMatcher(Origin, "APTSRESP", 0)) {
      LOG(info) << ID.getName() << " loaded APTSResponseData";
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
  o2::trkft3::Digitizer<N> mDigitizer{};
  int mLayers{0};
  std::vector<std::vector<o2::trkft3::Digit>> mDigits{};
  std::vector<std::vector<o2::trkft3::ROFRecord>> mROFRecords{};
  std::vector<std::vector<o2::trkft3::ROFRecord>> mROFRecordsAccum{};
  std::vector<o2::trkft3::Hit> mHits{};
  std::vector<o2::trkft3::Hit>* mHitsP{&mHits};
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabels{};
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabelsAccum{};
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
                           inputs, makeOutChannels(detOrig, getNLayers<o2::detectors::DetID::TRK>(), mctruth),
                           AlgorithmSpec{adaptFromTask<TRKFT3DPLDigitizerTask<o2::detectors::DetID::TRK>>(mctruth)},
                           Options{
                             {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}},
                             {"local-response-file", o2::framework::VariantType::String, "", {"use response file saved locally at this path/filename"}}}};
}

DataProcessorSpec getFT3DigitizerSpec(int channel, bool mctruth)
{
  std::string detStr = o2::detectors::DetID::getName(o2::detectors::DetID::FT3);
  auto detOrig = o2::header::gDataOriginFT3;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  inputs.emplace_back("FT3_aptsresp", "FT3", "APTSRESP", 0, Lifetime::Condition, ccdbParamSpec("IT3/Calib/APTSResponse"));

  return DataProcessorSpec{detStr + "Digitizer",
                           inputs, makeOutChannels(detOrig, getNLayers<o2::detectors::DetID::FT3>(), mctruth),
                           AlgorithmSpec{adaptFromTask<TRKFT3DPLDigitizerTask<o2::detectors::DetID::FT3>>(mctruth)},
                           Options{
                             {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}},
                             {"local-response-file", o2::framework::VariantType::String, "", {"use response file saved locally at this path/filename"}}}};
}

} // namespace o2::trkft3
