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
#include "IOTOFBase/GeometryTGeo.h"
#include "IOTOFBase/IOTOFBaseParam.h"

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
    mEnableResponseFiles = ic.options().get<bool>("enable-response-files");
    mLocalEfficiencyFile = ic.options().get<std::string>("local-efficiency-file");
    mLocalTimeResoFile = ic.options().get<std::string>("local-timereso-file");
    mLocalToAFile = ic.options().get<std::string>("local-toa-file");

    auto geom = GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G)); // make sure L2G matrices are loaded

    mDigitizer.setGeometry(geom);
    mDigitizer.init();

    mROMode = mDigitizer.isContinuous() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
  }

  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }

    if (mEnableResponseFiles) {
      pc.inputs().get<TH2D*>("madpixeff");
      pc.inputs().get<TH2D*>("madpixreso");
      pc.inputs().get<TH2D*>("madpixtoa");
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

    mDigitizer.setDigits(&mDigits);
    mDigitizer.setROFRecords(&mROFRecords);
    mDigitizer.setROFRecordIR(firstIR);
    if (mWithMCTruth) {
      mDigitizer.setMCLabels(&mLabels);
    }

    auto& eventParts = context->getEventParts(withQED);
    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    // o2::InteractionTimeRecord firstorbit(o2::InteractionRecord(0, o2::raw::HBFUtils::Instance().orbitFirstSampled), 0.0);
    for (int collID = 0; collID < timesview.size(); ++collID) {
      o2::InteractionTimeRecord orbit(timesview[collID]);
      // orbit += firstorbit
      mDigitizer.setEventTime(orbit);

      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (const auto& part : eventParts[collID]) {

        // get the hits for this event and this source
        mHits.clear();
        context->retrieveHits(mSimChains, o2::detectors::SimTraits::DETECTORBRANCHNAMES[mID][0].c_str(), part.sourceID, part.entryID, &mHits);

        if (mHits.size() > 0) {
          mDigits.clear();
          if (mWithMCTruth) {
            mLabels.clear();
          }

          LOG(debug) << "For collision " << collID << " eventID " << part.entryID << " found " << mHits.size() << " hits ";
          mDigitizer.process(&mHits, part.entryID, part.sourceID); // call actual digitization procedure
        }
      }
    }
    if (mDigitizer.isContinuous()) {
      LOG(debug) << "Number of digits before final flush: " << mDigits.size();
      mDigits.clear();
      if (mWithMCTruth) {
        mLabels.clear();
      }
      LOG(debug) << "Final flushing for continuous mode";
      mDigitizer.fillOutputContainer();
      LOG(debug) << "Number of digits after final flush: " << mDigits.size();
    }

    // here we have all digits and we can send them to consumer (aka snapshot it onto output)
    LOG(debug) << "Digitization finished with " << mDigits.size() << " digits and " << mROFRecords.size() << " ROF records";
    pc.outputs().snapshot(Output{mOrigin, "DIGITS", 0}, mDigits);
    pc.outputs().snapshot(Output{mOrigin, "DIGITSROF", 0}, mROFRecords);
    if (mWithMCTruth) {
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{mOrigin, "DIGITSMCTR", 0});
      mLabels.flatten_to(sharedlabels);
      // free space of existing label containers
      mLabels.clear_andfreememory();

      // write dummy MC2ROF vector to keep writer/readers backward compatible
      // NOTE: Steer/DigitizerWorkflow/src/ITSMFTDigitizerSpec.cxx also uses dummy MC2ROF
      static std::vector<o2::itsmft::MC2ROFRecord> dummyMC2ROF;
      pc.outputs().snapshot(Output{mOrigin, "DIGITSMC2ROF", 0}, dummyMC2ROF);
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
    if (matcher == ConcreteDataMatcher(mOrigin, "MADPIXEFF", 0)) {
      if (mLocalEfficiencyFile.empty()) {
        LOG(info) << "Using CCDB/MADPIX efficiency file";
        mDigitizer.loadEfficiencyMap((TH2D*)obj);
      } else {
        LOG(info) << "Response function will be loaded from local file: " << mLocalEfficiencyFile;
        std::unique_ptr<TFile> file(TFile::Open(mLocalEfficiencyFile.data(), "READ"));
        if (!file) {
          LOG(fatal) << "Cannot open response file " << mLocalEfficiencyFile.data();
        }
        mDigitizer.loadEfficiencyMap((TH2D*)file->Get("efficiency_pixel"));
      }
      return;
    }
    if (matcher == ConcreteDataMatcher(mOrigin, "MADPIXRESO", 0)) {
      if (mLocalTimeResoFile.empty()) {
        LOG(info) << "Using CCDB/MADPIX time resolution file";
        mDigitizer.loadResolutionMap((TH2D*)obj);
      } else {
        LOG(info) << "Response function will be loaded from local file: " << mLocalTimeResoFile;
        std::unique_ptr<TFile> file(TFile::Open(mLocalTimeResoFile.data(), "READ"));
        if (!file) {
          LOG(fatal) << "Cannot open response file " << mLocalTimeResoFile.data();
        }
        mDigitizer.loadResolutionMap((TH2D*)file->Get("sigma_pixel"));
      }
      return;
    }
    if (matcher == ConcreteDataMatcher(mOrigin, "MADPIXTOA", 0)) {
      if (mLocalToAFile.empty()) {
        LOG(info) << "Using CCDB/MADPIX time-of-arrival file";
        mDigitizer.loadTimeOfArrivalMap((TH2D*)obj);
      } else {
        LOG(info) << "Response function will be loaded from local file: " << mLocalToAFile;
        std::unique_ptr<TFile> file(TFile::Open(mLocalToAFile.data(), "READ"));
        if (!file) {
          LOG(fatal) << "Cannot open response file " << mLocalToAFile.data();
        }
        mDigitizer.loadTimeOfArrivalMap((TH2D*)file->Get("toa_pixel"));
      }
    }
  }

 private:
  bool mDisableQED = false;
  bool mEnableResponseFiles = false;
  bool mWithMCTruth{true};
  bool mFinished{false};
  std::string mLocalEfficiencyFile{""};
  std::string mLocalTimeResoFile{""};
  std::string mLocalToAFile{""};
  unsigned long mFirstOrbitTF = 0x0;
  const o2::detectors::DetID mID{o2::detectors::DetID::TF3};
  const o2::header::DataOrigin mOrigin{o2::header::gDataOriginTF3};
  o2::iotof::Digitizer mDigitizer{};
  std::vector<o2::iotof::Digit> mDigits{};
  std::vector<o2::itsmft::ROFRecord> mROFRecords{};
  std::vector<o2::itsmft::Hit> mHits{};
  std::vector<o2::itsmft::Hit>* mHitsP{&mHits};
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabels{};
  std::vector<TChain*> mSimChains{};
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::PRESENT; // readout mode
};

std::vector<o2::framework::OutputSpec> makeOutChannels(o2::header::DataOrigin detOrig, bool mctruth)
{
  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(detOrig, "DIGITS", o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(detOrig, "DIGITSROF", o2::framework::Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back(detOrig, "DIGITSMC2ROF", o2::framework::Lifetime::Timeframe);
    outputs.emplace_back(detOrig, "DIGITSMCTR", o2::framework::Lifetime::Timeframe);
  }
  outputs.emplace_back(detOrig, "ROMode", 0, o2::framework::Lifetime::Timeframe);
  return outputs;
}

o2::framework::DataProcessorSpec getIOTOFDigitizerSpec(int channel, bool mctruth)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<o2::header::DataHeader::SubSpecificationType>(channel), o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("madpixeff", "TF3", "MADPIXEFF", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("Users/g/glucia/tmp/ALICE3/IOTOF/pixelEfficiency"));
  inputs.emplace_back("madpixreso", "TF3", "MADPIXRESO", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("Users/g/glucia/tmp/ALICE3/IOTOF/pixelResolution"));
  inputs.emplace_back("madpixtoa", "TF3", "MADPIXTOA", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("Users/g/glucia/tmp/ALICE3/IOTOF/pixelTimeOfArrival"));

  const std::string detStr = o2::detectors::DetID::getName(o2::detectors::DetID::TF3);
  return o2::framework::DataProcessorSpec{detStr + "Digitizer",
                                          inputs,
                                          makeOutChannels(o2::header::gDataOriginTF3, mctruth),
                                          o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<IOTOFDPLDigitizerTask>(mctruth)},
                                          o2::framework::Options{
                                            {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}},
                                            {"enable-response-files", o2::framework::VariantType::Bool, false, {"enable in-pixel response maps from data"}},
                                            {"local-efficiency-file", o2::framework::VariantType::String, "", {"use efficiency map saved locally at this path/filename"}},
                                            {"local-timereso-file", o2::framework::VariantType::String, "", {"use time resolution map saved locally at this path/filename"}},
                                            {"local-toa-file", o2::framework::VariantType::String, "", {"use ToA map saved locally at this path/filename"}}}};
}

} // namespace o2::iotof
