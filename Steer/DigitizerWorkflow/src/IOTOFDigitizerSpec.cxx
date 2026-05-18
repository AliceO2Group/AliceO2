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

#include "IOTOFDigitizerSpec.h"
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
#include "IOTOFSimulation/Digitizer.h"
#include "Headers/DataHeader.h"

#include <TChain.h>
#include <TStopwatch.h>

#include <algorithm>
#include <memory>
#include <string>

using namespace o2::framework;

namespace o2::iotof
{

class IOTOFDPLDigitizerTask : o2::base::BaseDPLDigitizer
{
 public:
  using BaseDPLDigitizer::init;

  IOTOFDPLDigitizerTask(bool mctruth = true) : BaseDPLDigitizer(o2::base::InitServices::FIELD | o2::base::InitServices::GEOM),
                                               mWithMCTruth(mctruth) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
    mDisableQED = ic.options().get<bool>("disable-qed");
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

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(mID, mSimChains);
    const bool withQED = context->isQEDProvided() && !mDisableQED;
    auto& timesview = context->getEventRecords();
    LOG(info) << "GOT " << timesview.size() << " COLLISION TIMES";
    LOG(info) << "SIMCHAINS " << mSimChains.size();

    // if there is nothing to do ... return
    if (timesview.empty()) {
      return;
    }

    TStopwatch timer;
    timer.Start();
    LOG(info) << " CALLING TF3 DIGITIZATION ";

    auto& eventParts = context->getEventParts(withQED);
    uint64_t nDigits{0};
    for (uint32_t iLayer = 0; iLayer < static_cast<uint32_t>(mLayers); ++iLayer) {
      mDigits[iLayer].clear();
      // mROFRecords[iLayer].clear();
      // mROFRecordsAccum[iLayer].clear();
      if (mWithMCTruth) {
        // mLabels[iLayer].clear();
        // mLabelsAccum[iLayer].clear();
        // mMC2ROFRecordsAccum[iLayer].clear();
      }

      mDigitizer.setDigits(&mDigits[iLayer]);
      // mDigitizer.setROFRecords(&mROFRecords[iLayer]);
      // mDigitizer.setMCLabels(&mLabels[iLayer]);

      for (int collID = 0; collID < timesview.size(); ++collID) {
        for (const auto& part : eventParts[collID]) {

          // get the hits for this event and this source
          mHits.clear();
          context->retrieveHits(mSimChains, o2::detectors::SimTraits::DETECTORBRANCHNAMES[mID][0].c_str(), part.sourceID, part.entryID, &mHits);

          if (mHits.size() > 0) {
            LOG(debug) << "For collision " << collID << " eventID " << part.entryID << " found " << mHits.size() << " hits ";
            // mDigitizer.process(&mHits, part.entryID, part.sourceID, layer); // call actual digitization procedure
          }
        }
      }

    }

    timer.Stop();
    LOG(info) << "Digitization took " << timer.CpuTime() << "s";
    LOG(info) << "Produced " << nDigits << " digits";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);

    mFinished = true;
  }

 private:
  bool mDisableQED = false;
  bool mWithMCTruth{true};
  bool mFinished{false};
  int mLayers{2};
  unsigned long mFirstOrbitTF = 0x0;
  const o2::detectors::DetID mID{o2::detectors::DetID::TF3};
  const o2::header::DataOrigin mOrigin{o2::header::gDataOriginTF3};
  o2::iotof::Digitizer mDigitizer{};
  std::vector<std::vector<o2::iotof::Digit>> mDigits{};
  std::vector<std::vector<o2::itsmft::ROFRecord>> mROFRecords{};
  std::vector<std::vector<o2::itsmft::ROFRecord>> mROFRecordsAccum{};
  std::vector<o2::itsmft::Hit> mHits{};
  std::vector<o2::itsmft::Hit>* mHitsP{&mHits};
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabels{};
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabelsAccum{};
  std::vector<std::vector<o2::itsmft::MC2ROFRecord>> mMC2ROFRecordsAccum{};
  std::vector<TChain*> mSimChains{};
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::PRESENT; // readout mode
};

std::vector<o2::framework::OutputSpec> makeOutChannels(o2::header::DataOrigin detOrig, bool mctruth)
{
  std::vector<o2::framework::OutputSpec> outputs;
  for (uint32_t iLayer = 0; iLayer < 3; ++iLayer) {
    outputs.emplace_back(detOrig, "DIGITS", iLayer, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back(detOrig, "DIGITSROF", iLayer, o2::framework::Lifetime::Timeframe);
    if (mctruth) {
      outputs.emplace_back(detOrig, "DIGITSMC2ROF", iLayer, o2::framework::Lifetime::Timeframe);
      outputs.emplace_back(detOrig, "DIGITSMCTR", iLayer, o2::framework::Lifetime::Timeframe);
    }
  }
  outputs.emplace_back(detOrig, "ROMode", 0, o2::framework::Lifetime::Timeframe);
  return outputs;
}

o2::framework::DataProcessorSpec getIOTOFDigitizerSpec(int channel, bool mctruth)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<o2::header::DataHeader::SubSpecificationType>(channel), o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("IOTOF_aptsresp", "TF3", "APTSRESP", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("IT3/Calib/APTSResponse"));

  const std::string detStr = o2::detectors::DetID::getName(o2::detectors::DetID::TF3);
  return o2::framework::DataProcessorSpec{detStr + "Digitizer",
                                          inputs,
                                          makeOutChannels(o2::header::gDataOriginTF3, mctruth),
                                          o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<IOTOFDPLDigitizerTask>(mctruth)},
                                          o2::framework::Options{
                                            {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}},
                                            {"local-response-file", o2::framework::VariantType::String, "", {"use response file saved locally at this path/filename"}}}};
}

} // namespace o2::iotof
