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
#include "ITS3Simulation/Digitizer.h"
#include "ITSMFTSimulation/DPLDigitizerParam.h"
#include "ITS3Simulation/ITS3DPLDigitizerParam.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITS3Base/ITS3Params.h"

#include <TChain.h>
#include <TStopwatch.h>

#include <string>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2::its3
{
using namespace o2::base;
class ITS3DPLDigitizerTask : BaseDPLDigitizer
{
 public:
  using BaseDPLDigitizer::init;

  ITS3DPLDigitizerTask(bool mctruth = true, bool doStag = false) : BaseDPLDigitizer(InitServices::FIELD | InitServices::GEOM), mWithMCTruth(mctruth), mDoStaggering(doStag) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
    mDisableQED = ic.options().get<bool>("disable-qed");
  }

  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    mFirstOrbitTF = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
    const o2::InteractionRecord firstIR(0, mFirstOrbitTF);
    updateTimeDependentParams(pc);

    TStopwatch timer;
    timer.Start();
    LOG(info) << " CALLING ITS3 DIGITIZATION ";

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(mID, mSimChains);
    const bool withQED = context->isQEDProvided() && !mDisableQED;
    auto& timesview = context->getEventRecords(withQED);
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    LOG(info) << "GOT " << timesview.size() << " COLLISSION TIMES";
    LOG(info) << "SIMCHAINS " << mSimChains.size();

    // if there is nothing to do ... return
    if (timesview.empty()) {
      return;
    }

    uint64_t nDigits{0};
    for (uint32_t iLayer = 0; iLayer < (mDoStaggering ? 7 : 1); ++iLayer) {
      const int layer = (mDoStaggering) ? iLayer : -1;
      mDigitizer.setDigits(&mDigits[iLayer]);
      mDigitizer.setROFRecords(&mROFRecords[iLayer]);
      mDigitizer.setMCLabels(&mLabels[iLayer]);
      mDigitizer.resetROFrameBounds();
      // digits are directly put into DPL owned resource
      auto& digitsAccum = pc.outputs().make<std::vector<itsmft::Digit>>(Output{mOrigin, "DIGITS", iLayer});

      // rofs are accumulated first and the copied
      const int nROFsPerOrbit = o2::constants::lhc::LHCMaxBunches / par.getROFLengthInBC(iLayer);
      const int nROFsTF = nROFsPerOrbit * raw::HBFUtils::Instance().getNOrbitsPerTF();
      mROFRecordsAccum[iLayer].reserve(nROFsTF);

      auto accumulate = [this, &digitsAccum, &iLayer]() {
        // accumulate result of single event processing on a specific layer, called after processing every event supplied
        // AND after the final flushing via digitizer::fillOutputContainer
        if (!mDigits[iLayer].size()) {
          return; // no digits were flushed, nothing to accumulate
        }
        auto ndigAcc = digitsAccum.size();
        std::copy(mDigits[iLayer].begin(), mDigits[iLayer].end(), std::back_inserter(digitsAccum));

        // fix ROFrecords references on ROF entries
        auto nROFRecsOld = mROFRecordsAccum[iLayer].size();

        for (int i = 0; i < mROFRecords[iLayer].size(); i++) {
          auto& rof = mROFRecords[iLayer][i];
          rof.setFirstEntry(ndigAcc + rof.getFirstEntry());
          rof.print();
        }

        std::copy(mROFRecords[iLayer].begin(), mROFRecords[iLayer].end(), std::back_inserter(mROFRecordsAccum[iLayer]));
        if (mWithMCTruth) {
          mLabelsAccum[iLayer].mergeAtBack(mLabels[iLayer]);
        }
        LOG(info) << "Added " << mDigits[iLayer].size() << " digits" << ((mDoStaggering) ? std::format(" on layer {}", iLayer) : "");
        // clean containers from already accumulated stuff
        mLabels[iLayer].clear();
        mDigits[iLayer].clear();
        mROFRecords[iLayer].clear();
      }; // and accumulate lambda

      const auto& eventParts = context->getEventParts(withQED);
      const int64_t bcShift = mDigitizer.getParams().getROFrameBiasInBC(layer); // this accounts the misalignment and the opt. imposed rof delay
      // loop over all composite collisions given from context (aka loop over all the interaction records)
      for (int collID = 0; collID < timesview.size(); ++collID) {
        auto irt = timesview[collID];
        if (irt.toLong() < bcShift) { // due to the ROF misalignment (+opt. delay) the collision would go to negative ROF ID, discard
          continue;
        }
        irt -= bcShift; // account for the ROF start shift

        mDigitizer.setEventTime(irt, layer);
        mDigitizer.resetEventROFrames(); // to estimate min/max ROF for this collID
        // for each collision, loop over the constituents event and source IDs
        // (background signal merging is basically taking place here)
        for (const auto& part : eventParts[collID]) {

          // get the hits for this event and this source
          mHits.clear();
          context->retrieveHits(mSimChains, o2::detectors::SimTraits::DETECTORBRANCHNAMES[o2::detectors::DetID::IT3][0].c_str(), part.sourceID, part.entryID, &mHits);

          if (mHits.size() > 0) {
            LOG(debug) << "For collision " << collID << " eventID " << part.entryID << " found " << mHits.size() << " hits ";
            mDigitizer.process(&mHits, part.entryID, part.sourceID, layer); // call actual digitization procedure
          }
        }
        accumulate();
      }
      mDigitizer.fillOutputContainer(0xffffffff, layer);
      accumulate();
      nDigits += digitsAccum.size();

      // here we have all digits and labels and we can send them to consumer (aka snapshot it onto output)
      // ensure that the rof output is continuous
      if (nROFsTF != mROFRecordsAccum[iLayer].size()) {
        // it can happen that in the digitization rofs without contributing hits are skipped
        // however downstream consumers of the clusters cannot know apriori the time structure
        // the cluster rofs do not account for the bias so it will start always at BC=0
        // also have to account for spillage into next TF
        const size_t nROFsLayer = std::max((size_t)nROFsTF, mROFRecordsAccum[iLayer].size());
        std::vector<o2::itsmft::ROFRecord> expDigitRofVec(nROFsLayer);
        for (int iROF{0}; iROF < nROFsLayer; ++iROF) {
          auto& rof = expDigitRofVec[iROF];
          int orb = iROF * par.getROFLengthInBC(iLayer) / o2::constants::lhc::LHCMaxBunches + mFirstOrbitTF;
          int bc = iROF * par.getROFLengthInBC(iLayer) % o2::constants::lhc::LHCMaxBunches + par.getROFDelayInBC(iLayer);
          o2::InteractionRecord ir(bc, orb);
          rof.setBCData(ir);
          rof.setROFrame(iROF);
          rof.setNEntries(0);
          rof.setFirstEntry(-1);
        }
        uint32_t prevEntry{0};
        for (const auto& rof : mROFRecordsAccum[iLayer]) {
          const auto& ir = rof.getBCData();
          if (ir < firstIR) {
            LOGP(warn, "Discard ROF {} preceding TF 1st orbit {}{}", ir.asString(), mFirstOrbitTF, ((mDoStaggering) ? std::format(" on layer {}", iLayer) : ""));
            continue;
          }
          auto irToFirst = ir - firstIR;
          if (irToFirst.toLong() - par.getROFDelayInBC(iLayer) < 0) {
            LOGP(warn, "Discard ROF {} preceding TF 1st orbit {} due to imposed ROF delay{}", ir.asString(), mFirstOrbitTF, ((mDoStaggering) ? std::format(" on layer {}", iLayer) : ""));
            continue;
          }
          irToFirst -= par.getROFDelayInBC(iLayer);
          const int irROF = irToFirst.toLong() / par.getROFLengthInBC(iLayer);
          auto& expROF = expDigitRofVec[irROF];
          expROF.setFirstEntry(rof.getFirstEntry());
          expROF.setNEntries(rof.getNEntries());
          if (expROF.getBCData() != rof.getBCData()) {
            LOGP(fatal, "detected mismatch between expected {} and received {}", expROF.asString(), rof.asString());
          }
        }
        int prevFirst{0};
        for (auto& rof : expDigitRofVec) {
          if (rof.getFirstEntry() < 0) {
            rof.setFirstEntry(prevFirst);
          }
          prevFirst = rof.getFirstEntry();
        }
        // if more rofs where accumulated than ROFs possible in the TF, cut them away
        // by construction expDigitRofVec is at least nROFsTF long
        expDigitRofVec.resize(nROFsTF);
        pc.outputs().snapshot(Output{mOrigin, "DIGITSROF", iLayer}, expDigitRofVec);
      } else {
        pc.outputs().snapshot(Output{mOrigin, "DIGITSROF", iLayer}, mROFRecordsAccum[iLayer]);
      }
      if (mWithMCTruth) {
        auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{mOrigin, "DIGITSMCTR", iLayer});
        mLabelsAccum[iLayer].flatten_to(sharedlabels);
        // free space of existing label containers
        mLabels[iLayer].clear_andfreememory();
        mLabelsAccum[iLayer].clear_andfreememory();
      }
    }

    LOG(info) << "IT3: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{mOrigin, "ROMode", 0}, mROMode);

    timer.Stop();
    LOG(info) << "Digitization took " << timer.CpuTime() << "s";
    LOG(info) << "Produced " << nDigits << " digits";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);

    mFinished = true;
  }

  void updateTimeDependentParams(ProcessingContext& pc)
  {
    static bool initOnce{false};
    if (!initOnce) {
      initOnce = true;
      auto& digipar = mDigitizer.getParams();

      // configure digitizer
      o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
      geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G)); // make sure L2G matrices are loaded
      mDigitizer.setGeometry(geom);

      const auto& dopt = o2::itsmft::DPLDigitizerParam<o2::detectors::DetID::ITS>::Instance();
      const auto& doptIB = o2::its3::ITS3DPLDigitizerParam::Instance();
      pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("ITS_alppar");
      const auto& aopt = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
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

      // ITS3 inner barrel specific parameters
      digipar.setIBChargeThreshold(doptIB.IBChargeThreshold);
      digipar.setIBNSimSteps(doptIB.nIBSimSteps);
      digipar.setIBNoisePerPixel(doptIB.IBNoisePerPixel);

      // staggering parameters
      if (mDoStaggering) {
        for (int iLayer{0}; iLayer < 7; ++iLayer) {
          auto frameNS = aopt.getROFLengthInBC(iLayer) * o2::constants::lhc::LHCBunchSpacingNS;
          digipar.addROFrameLayerLengthInBC(aopt.getROFLengthInBC(iLayer));
          // NOTE: the rof delay looks from the digitizer like an additional bias
          digipar.addROFrameLayerBiasInBC(aopt.getROFBiasInBC(iLayer) + aopt.getROFDelayInBC(iLayer));
          digipar.addStrobeDelay(aopt.strobeDelay);
          digipar.addStrobeLength(aopt.strobeLengthCont > 0 ? aopt.strobeLengthCont : frameNS - aopt.strobeDelay);
          digipar.setROFrameLength(aopt.getROFLengthInBC(iLayer) * o2::constants::lhc::LHCBunchSpacingNS, iLayer);
        }
      }

      mROMode = digipar.isContinuous() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
      LOG(info) << mID.getName() << " simulated in "
                << ((mROMode == o2::parameters::GRPObject::CONTINUOUS) ? "CONTINUOUS" : "TRIGGERED")
                << " RO mode";

      if (o2::its3::ITS3Params::Instance().useDeadChannelMap) {
        pc.inputs().get<o2::itsmft::NoiseMap*>("IT3_dead"); // trigger final ccdb update
      }

      pc.inputs().get<o2::itsmft::AlpideSimResponse*>("IT3_alpiderespvbb0");
      if (o2::its3::ITS3Params::Instance().chipResponseFunction != "Alpide") {
        pc.inputs().get<o2::itsmft::AlpideSimResponse*>("IT3_aptsresp");
      }

      // init digitizer
      mDigitizer.init();
    }
    // Other time-dependent parameters can be added below
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    if (matcher == ConcreteDataMatcher(detectors::DetID::ITS, "ALPIDEPARAM", 0)) {
      LOG(info) << mID.getName() << " Alpide param updated";
      const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
      par.printKeyValues();
      return;
    }
    if (matcher == ConcreteDataMatcher(mOrigin, "DEADMAP", 0)) {
      LOG(info) << mID.getName() << " static dead map updated";
      mDigitizer.setDeadChannelsMap((o2::itsmft::NoiseMap*)obj);
      return;
    }
    if (matcher == ConcreteDataMatcher(mOrigin, "ALPIDERESPVbb0", 0)) {
      LOG(info) << mID.getName() << " loaded AlpideResponseData for Vbb=0V";
      mDigitizer.getParams().setOBSimResponse((o2::itsmft::AlpideSimResponse*)obj);
    }
    if (matcher == ConcreteDataMatcher(mOrigin, "APTSRESP", 0)) {
      LOG(info) << mID.getName() << " loaded APTSResponseData";
      mDigitizer.getParams().setIBSimResponse((o2::itsmft::AlpideSimResponse*)obj);
    }
  }

 private:
  bool mWithMCTruth{true};
  bool mFinished{false};
  bool mDisableQED{false};
  unsigned long mFirstOrbitTF = 0x0;
  const o2::detectors::DetID mID{o2::detectors::DetID::IT3};
  const o2::header::DataOrigin mOrigin{o2::header::gDataOriginIT3};
  o2::its3::Digitizer mDigitizer{};
  std::array<std::vector<o2::itsmft::Digit>, 7> mDigits{};
  std::array<std::vector<o2::itsmft::ROFRecord>, 7> mROFRecords{};
  std::array<std::vector<o2::itsmft::ROFRecord>, 7> mROFRecordsAccum{};
  std::vector<o2::itsmft::Hit> mHits;
  std::vector<o2::itsmft::Hit>* mHitsP{&mHits};
  std::array<o2::dataformats::MCTruthContainer<o2::MCCompLabel>, 7> mLabels;
  std::array<o2::dataformats::MCTruthContainer<o2::MCCompLabel>, 7> mLabelsAccum;
  std::vector<TChain*> mSimChains{};
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::PRESENT; // readout mode
  bool mDoStaggering = false;
};

namespace
{
std::vector<OutputSpec> makeOutChannels(o2::header::DataOrigin detOrig, bool mctruth, bool doStag)
{
  std::vector<OutputSpec> outputs;
  for (int iLayer{0}; iLayer < (doStag ? 7 : 1); ++iLayer) {
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

DataProcessorSpec getITS3DigitizerSpec(int channel, bool mctruth, bool doStag)
{
  auto detOrig = o2::header::gDataOriginIT3;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  inputs.emplace_back("ITS_alppar", "ITS", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam"));
  if (o2::its3::ITS3Params::Instance().useDeadChannelMap) {
    inputs.emplace_back("IT3_dead", "IT3", "DEADMAP", 0, Lifetime::Condition, ccdbParamSpec("IT3/Calib/DeadMap"));
  }
  inputs.emplace_back("IT3_alpiderespvbb0", "IT3", "ALPIDERESPVbb0", 0, Lifetime::Condition, ccdbParamSpec("ITSMFT/Calib/ALPIDEResponseVbb0"));
  inputs.emplace_back("IT3_aptsresp", "IT3", "APTSRESP", 0, Lifetime::Condition, ccdbParamSpec("IT3/Calib/APTSResponse"));

  return DataProcessorSpec{.name = "IT3Digitizer",
                           .inputs = inputs,
                           .outputs = makeOutChannels(detOrig, mctruth, doStag),
                           .algorithm = AlgorithmSpec{adaptFromTask<ITS3DPLDigitizerTask>(mctruth)},
                           .options = Options{{"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}}}};
}

} // namespace o2::its3
