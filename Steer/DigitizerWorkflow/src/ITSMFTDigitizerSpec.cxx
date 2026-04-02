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
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/CCDBParamSpec.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "DataFormatsITSMFT/TimeDeadMap.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/SimTraits.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTSimulation/Digitizer.h"
#include "ITSMFTSimulation/DPLDigitizerParam.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "ITSBase/GeometryTGeo.h"
#include "MFTBase/GeometryTGeo.h"
#include <TChain.h>
#include <TStopwatch.h>
#include <string>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2::itsmft
{

using namespace o2::base;
template <int N>
class ITSMFTDPLDigitizerTask : BaseDPLDigitizer
{
 public:
  static constexpr o2::detectors::DetID ID{N == o2::detectors::DetID::ITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT};
  static constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};

  using BaseDPLDigitizer::init;

  void initDigitizerTask(framework::InitContext& ic) override
  {
    mDisableQED = ic.options().get<bool>("disable-qed");
    if (mDoStaggering) {
      mLayers = DPLAlpideParam<N>::getNLayers();
    }
    mDigits.resize(mLayers);
    mROFRecords.resize(mLayers);
    mROFRecordsAccum.resize(mLayers);
    mLabels.resize(mLayers);
    mLabelsAccum.resize(mLayers);
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
    LOG(info) << " CALLING ITS DIGITIZATION ";

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(ID, mSimChains);
    const bool withQED = context->isQEDProvided() && !mDisableQED;
    auto& timesview = context->getEventRecords(withQED);
    LOG(info) << "GOT " << timesview.size() << " COLLISSION TIMES";
    LOG(info) << "SIMCHAINS: " << mSimChains.size();

    // if there is nothing to do ... return
    if (timesview.size() == 0) {
      return;
    }

    uint64_t nDigits{0};
    for (uint32_t iLayer = 0; iLayer < mLayers; ++iLayer) {
      const int layer = (mDoStaggering) ? iLayer : -1;
      mDigitizer.setDigits(&mDigits[iLayer]);
      mDigitizer.setROFRecords(&mROFRecords[iLayer]);
      mDigitizer.setMCLabels(&mLabels[iLayer]);
      mDigitizer.resetROFrameBounds();

      // digits are directly put into DPL owned resource
      auto& digitsAccum = pc.outputs().make<std::vector<itsmft::Digit>>(Output{Origin, "DIGITS", iLayer});

      // rofs are accumulated first and the copied
      const int nROFsPerOrbit = o2::constants::lhc::LHCMaxBunches / DPLAlpideParam<N>::Instance().getROFLengthInBC(iLayer);
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
          context->retrieveHits(mSimChains, o2::detectors::SimTraits::DETECTORBRANCHNAMES[ID][0].c_str(), part.sourceID, part.entryID, &mHits);

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
          int orb = iROF * DPLAlpideParam<N>::Instance().getROFLengthInBC(iLayer) / o2::constants::lhc::LHCMaxBunches + mFirstOrbitTF;
          int bc = iROF * DPLAlpideParam<N>::Instance().getROFLengthInBC(iLayer) % o2::constants::lhc::LHCMaxBunches + DPLAlpideParam<N>::Instance().getROFDelayInBC(iLayer);
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
          if (irToFirst.toLong() - DPLAlpideParam<N>::Instance().getROFDelayInBC(iLayer) < 0) {
            LOGP(warn, "Discard ROF {} preceding TF 1st orbit {} due to imposed ROF delay{}", ir.asString(), mFirstOrbitTF, ((mDoStaggering) ? std::format(" on layer {}", iLayer) : ""));
            continue;
          }
          irToFirst -= DPLAlpideParam<N>::Instance().getROFDelayInBC(iLayer);
          const int irROF = irToFirst.toLong() / DPLAlpideParam<N>::Instance().getROFLengthInBC(iLayer);
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
        pc.outputs().snapshot(Output{Origin, "DIGITSROF", iLayer}, expDigitRofVec);
      } else {
        pc.outputs().snapshot(Output{Origin, "DIGITSROF", iLayer}, mROFRecordsAccum[iLayer]);
      }
      if (mWithMCTruth) {
        auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{Origin, "DIGITSMCTR", iLayer});
        mLabelsAccum[iLayer].flatten_to(sharedlabels);
        // free space of existing label containers
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

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    if (matcher == ConcreteDataMatcher(Origin, "NOISEMAP", 0)) {
      LOG(info) << ID.getName() << " noise map updated";
      mDigitizer.setNoiseMap((const o2::itsmft::NoiseMap*)obj);
      return;
    }
    if (matcher == ConcreteDataMatcher(Origin, "DEADMAP", 0)) {
      LOG(info) << ID.getName() << " static dead map updated";
      mDeadMap = (o2::itsmft::NoiseMap*)obj;
      mDigitizer.setDeadChannelsMap(mDeadMap);
      return;
    }
    if (matcher == ConcreteDataMatcher(Origin, "TimeDeadMap", 0)) {
      o2::itsmft::TimeDeadMap* timedeadmap = (o2::itsmft::TimeDeadMap*)obj;
      if (!timedeadmap->isDefault()) {
        timedeadmap->decodeMap(mFirstOrbitTF, *mDeadMap, true);
        if (mTimeDeadMapUpdated) {
          LOGP(fatal, "Attempt to add time-dependent map to already modified static map");
        }
        mTimeDeadMapUpdated = true;
        mDigitizer.setDeadChannelsMap(mDeadMap);
        LOG(info) << ID.getName() << " time-dependent dead map updated";
      } else {
        LOG(info) << ID.getName() << " time-dependent dead map is default/empty";
      }

      return;
    }
    if (matcher == ConcreteDataMatcher(Origin, "ALPIDEPARAM", 0)) {
      LOG(info) << ID.getName() << " Alpide param updated";
      const auto& par = o2::itsmft::DPLAlpideParam<N>::Instance();
      par.printKeyValues();
      return;
    }
    if (matcher == ConcreteDataMatcher(Origin, "ALPIDERESPVbb0", 0)) {
      LOG(info) << ID.getName() << " loaded AlpideResponseData for Vbb=0V";
      mDigitizer.setAlpideResponse((o2::itsmft::AlpideSimResponse*)obj, 0);
    }
    if (matcher == ConcreteDataMatcher(Origin, "ALPIDERESPVbbM3", 0)) {
      LOG(info) << ID.getName() << " loaded AlpideResponseData for Vbb=-3V";
      mDigitizer.setAlpideResponse((o2::itsmft::AlpideSimResponse*)obj, 1);
    }
  }

 protected:
  ITSMFTDPLDigitizerTask(bool mctruth = true, bool doStag = false) : BaseDPLDigitizer(InitServices::FIELD | InitServices::GEOM), mWithMCTruth(mctruth), mDoStaggering(doStag) {}

  void updateTimeDependentParams(ProcessingContext& pc)
  {
    std::string detstr(o2::detectors::DetID::getName(ID));
    pc.inputs().get<o2::itsmft::NoiseMap*>(detstr + "_noise");
    pc.inputs().get<o2::itsmft::NoiseMap*>(detstr + "_dead");
    // TODO: the code should run even if this object does not exist. Or: create default object
    pc.inputs().get<o2::itsmft::TimeDeadMap*>(detstr + "_time_dead");
    pc.inputs().get<o2::itsmft::DPLAlpideParam<N>*>(detstr + "_alppar");
    pc.inputs().get<o2::itsmft::AlpideSimResponse*>(detstr + "_alpiderespvbb0");
    pc.inputs().get<o2::itsmft::AlpideSimResponse*>(detstr + "_alpiderespvbbm3");

    auto& dopt = o2::itsmft::DPLDigitizerParam<N>::Instance();
    auto& aopt = o2::itsmft::DPLAlpideParam<N>::Instance();
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
    // staggering parameters
    if (mDoStaggering) {
      for (int iLayer{0}; iLayer < o2::itsmft::DPLAlpideParam<N>::getNLayers(); ++iLayer) {
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
    LOG(info) << detstr << " simulated in "
              << ((mROMode == o2::parameters::GRPObject::CONTINUOUS) ? "CONTINUOUS" : "TRIGGERED")
              << " RO mode";

    // configure digitizer
    o2::itsmft::GeometryTGeo* geom = nullptr;
    if constexpr (N == o2::detectors::DetID::ITS) {
      geom = o2::its::GeometryTGeo::Instance();
    } else {
      geom = o2::mft::GeometryTGeo::Instance();
    }
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G)); // make sure L2G matrices are loaded
    mDigitizer.setGeometry(geom);
    mDigitizer.init();
  }

  bool mWithMCTruth = true;
  bool mDoStaggering = false;
  bool mFinished = false;
  bool mDisableQED = false;
  int mLayers = 1;
  unsigned long mFirstOrbitTF = 0x0;
  o2::itsmft::Digitizer mDigitizer;
  std::vector<std::vector<o2::itsmft::Digit>> mDigits;
  std::vector<std::vector<o2::itsmft::ROFRecord>> mROFRecords;
  std::vector<std::vector<o2::itsmft::ROFRecord>> mROFRecordsAccum;
  std::vector<o2::itsmft::Hit> mHits;
  std::vector<o2::itsmft::Hit>* mHitsP = &mHits;
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabels;
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabelsAccum;
  std::vector<TChain*> mSimChains;
  o2::itsmft::NoiseMap* mDeadMap = nullptr;

  bool mTimeDeadMapUpdated = false;
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::PRESENT; // readout mode
};

//_______________________________________________
class ITSDPLDigitizerTask : public ITSMFTDPLDigitizerTask<o2::detectors::DetID::ITS>
{
 public:
  ITSDPLDigitizerTask(bool mctruth = true, bool doStag = false) : ITSMFTDPLDigitizerTask<o2::detectors::DetID::ITS>(mctruth, doStag) {}
};

//_______________________________________________
class MFTDPLDigitizerTask : public ITSMFTDPLDigitizerTask<o2::detectors::DetID::MFT>
{
 public:
  MFTDPLDigitizerTask(bool mctruth = true, bool doStag = false) : ITSMFTDPLDigitizerTask<o2::detectors::DetID::MFT>(mctruth, doStag) {}
};

namespace
{
template <int N>
std::vector<OutputSpec> makeOutChannels(o2::header::DataOrigin detOrig, bool mctruth, bool doStag)
{
  std::vector<OutputSpec> outputs;
  uint32_t nLayers = doStag ? DPLAlpideParam<N>::getNLayers() : 1;
  for (uint32_t iLayer = 0; iLayer < nLayers; ++iLayer) {
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

DataProcessorSpec getITSDigitizerSpec(int channel, bool mctruth, bool doStag)
{
  std::string detStr = o2::detectors::DetID::getName(ITSDPLDigitizerTask::ID);
  auto detOrig = ITSDPLDigitizerTask::Origin;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  inputs.emplace_back("ITS_noise", "ITS", "NOISEMAP", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/NoiseMap"));
  inputs.emplace_back("ITS_dead", "ITS", "DEADMAP", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/DeadMap"));
  inputs.emplace_back("ITS_time_dead", "ITS", "TimeDeadMap", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/TimeDeadMap"));
  inputs.emplace_back("ITS_alppar", "ITS", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam"));
  inputs.emplace_back("ITS_alpiderespvbb0", "ITS", "ALPIDERESPVbb0", 0, Lifetime::Condition, ccdbParamSpec("ITSMFT/Calib/ALPIDEResponseVbb0"));
  inputs.emplace_back("ITS_alpiderespvbbm3", "ITS", "ALPIDERESPVbbM3", 0, Lifetime::Condition, ccdbParamSpec("ITSMFT/Calib/ALPIDEResponseVbbM3"));
  return DataProcessorSpec{.name = detStr + "Digitizer",
                           .inputs = inputs,
                           .outputs = makeOutChannels<o2::detectors::DetID::ITS>(detOrig, mctruth, doStag),
                           .algorithm = AlgorithmSpec{adaptFromTask<ITSDPLDigitizerTask>(mctruth, doStag)},
                           .options = Options{
                             {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}}}};
}

DataProcessorSpec getMFTDigitizerSpec(int channel, bool mctruth, bool doStag)
{
  std::string detStr = o2::detectors::DetID::getName(MFTDPLDigitizerTask::ID);
  auto detOrig = MFTDPLDigitizerTask::Origin;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  inputs.emplace_back("MFT_noise", "MFT", "NOISEMAP", 0, Lifetime::Condition, ccdbParamSpec("MFT/Calib/NoiseMap"));
  inputs.emplace_back("MFT_dead", "MFT", "DEADMAP", 0, Lifetime::Condition, ccdbParamSpec("MFT/Calib/DeadMap"));
  inputs.emplace_back("MFT_time_dead", "MFT", "TimeDeadMap", 0, Lifetime::Condition, ccdbParamSpec("MFT/Calib/TimeDeadMap"));
  inputs.emplace_back("MFT_alppar", "MFT", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("MFT/Config/AlpideParam"));
  inputs.emplace_back("MFT_alpiderespvbb0", "MFT", "ALPIDERESPVbb0", 0, Lifetime::Condition, ccdbParamSpec("ITSMFT/Calib/ALPIDEResponseVbb0"));
  inputs.emplace_back("MFT_alpiderespvbbm3", "MFT", "ALPIDERESPVbbM3", 0, Lifetime::Condition, ccdbParamSpec("ITSMFT/Calib/ALPIDEResponseVbbM3"));
  return DataProcessorSpec{.name = detStr + "Digitizer",
                           .inputs = inputs,
                           .outputs = makeOutChannels<o2::detectors::DetID::MFT>(detOrig, mctruth, doStag),
                           .algorithm = AlgorithmSpec{adaptFromTask<MFTDPLDigitizerTask>(mctruth, doStag)},
                           .options = Options{{"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}}}};
}

} // namespace o2::itsmft
// end namespace o2
