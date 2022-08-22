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

#ifndef O2_CALIBRATION_TOFCHANNEL_CALIBRATOR_H
#define O2_CALIBRATION_TOFCHANNEL_CALIBRATOR_H

/// @file   TOFChannelCalibratorSpec.h
/// @brief  Device to calibrate TOF channles (offsets)

#include "CommonUtils/NameConf.h"
#include "TOFCalibration/TOFChannelCalibrator.h"
#include "DetectorsCalibration/Utils.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/CalibInfoCluster.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include <limits>
#include "TFile.h"
#include "DetectorsBase/GRPGeomHelper.h"

#include <TSystem.h>

using namespace o2::framework;

namespace o2
{
namespace calibration
{

template <class T>
class TOFChannelCalibDevice : public o2::framework::Task
{

  using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;
  using LHCphase = o2::dataformats::CalibLHCphaseTOF;

 public:
  explicit TOFChannelCalibDevice(std::shared_ptr<o2::base::GRPGeomRequest> req, bool useCCDB, bool attachChannelOffsetToLHCphase, bool isCosmics, bool perstrip = false, bool safe = false) : mCCDBRequest(req), mUseCCDB(useCCDB), mAttachToLHCphase(attachChannelOffsetToLHCphase), mCosmics(isCosmics), mDoPerStrip(perstrip), mSafeMode(safe) {}

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    int minEnt = ic.options().get<int>("min-entries"); //std::max(50, ic.options().get<int>("min-entries"));
    int nb = std::max(500, ic.options().get<int>("nbins"));
    mRange = ic.options().get<float>("range");
    int isTest = ic.options().get<bool>("do-TOF-channel-calib-in-test-mode");
    bool updateAtEORonly = ic.options().get<bool>("update-at-end-of-run-only");
    auto slotL = ic.options().get<uint32_t>("tf-per-slot");
    auto delay = ic.options().get<uint32_t>("max-delay");
    auto updateInterval = ic.options().get<uint32_t>("update-interval");
    auto deltaUpdateInterval = ic.options().get<uint32_t>("delta-update-interval");
    mCalibrator = std::make_unique<o2::tof::TOFChannelCalibrator<T>>(minEnt, nb, mRange);

    mCalibrator->doPerStrip(mDoPerStrip);
    mCalibrator->doSafeMode(mSafeMode);

    // default behaviour is to have only 1 slot at a time, accepting everything for it till the
    // minimum statistics is reached;
    // if one defines a different slot length and delay,
    // the usual time slot calibration behaviour is used;
    // if one defines that the calibration should happen only at the end of the run,
    // then the slot length and delay won't matter
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setCheckIntervalInfiniteSlot(updateInterval);
    mCalibrator->setCheckDeltaIntervalInfiniteSlot(deltaUpdateInterval);
    mCalibrator->setMaxSlotsDelay(delay);

    if (updateAtEORonly) { // has priority over other settings
      mCalibrator->setUpdateAtTheEndOfRunOnly();
    }

    mCalibrator->setIsTest(isTest);
    mCalibrator->setDoCalibWithCosmics(mCosmics);

    // calibration objects set to zero
    mPhase.addLHCphase(0, 0);
    mPhase.addLHCphase(o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP_SECONDS, 0);

    if (gSystem->AccessPathName("localTimeSlewing.root") == false) {
      TFile* fsleewing = TFile::Open("localTimeSlewing.root");
      if (fsleewing) {
        TimeSlewing* ob = (TimeSlewing*)fsleewing->Get("ccdb_object");
        mTimeSlewing = *ob;
        return;
      }
    } else {
      for (int ich = 0; ich < TimeSlewing::NCHANNELS; ich++) {
        mTimeSlewing.addTimeSlewingInfo(ich, 0, 0);
        int sector = ich / TimeSlewing::NCHANNELXSECTOR;
        int channelInSector = ich % TimeSlewing::NCHANNELXSECTOR;
        mTimeSlewing.setFractionUnderPeak(sector, channelInSector, 1);
      }
    }
  }

  //_________________________________________________________________
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
    mUpdateCCDB = false;
    if (matcher == ConcreteDataMatcher("TOF", "LHCphaseCal", 0)) {
      mUpdateCCDB = true;
      return;
    }
    if (matcher == ConcreteDataMatcher("TOF", "ChannelCalibCal", 0)) {
      mUpdateCCDB = true;
      return;
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    long startTimeLHCphase = 0;
    long startTimeChCalib = 0;
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
    auto tfcounter = mCalibrator->getCurrentTFInfo().tfCounter;

    if (mUseCCDB) { // read calibration objects from ccdb
      const auto lhcPhaseIn = pc.inputs().get<LHCphase*>("tofccdbLHCphase");
      const auto channelCalibIn = pc.inputs().get<TimeSlewing*>("tofccdbChannelCalib");

      if (!mcalibTOFapi) {
        LHCphase* lhcPhase = new LHCphase(std::move(*lhcPhaseIn));
        TimeSlewing* channelCalib = new TimeSlewing(std::move(*channelCalibIn));
        mcalibTOFapi = new o2::tof::CalibTOFapi(long(0), lhcPhase, channelCalib);
        mUpdateCCDB = false;
      } else {
        // if the calib objects were updated, we need to update the mcalibTOFapi
        if (mUpdateCCDB) {
          delete mcalibTOFapi;
          LHCphase* lhcPhase = new LHCphase(*lhcPhaseIn);
          TimeSlewing* channelCalib = new TimeSlewing(std::move(*channelCalibIn));
          mcalibTOFapi = new o2::tof::CalibTOFapi(long(0), lhcPhase, channelCalib);
          mUpdateCCDB = false;
        }
      }
    } else { // we use "fake" initial calibrations
      if (!mcalibTOFapi) {
        mcalibTOFapi = new o2::tof::CalibTOFapi(long(0), &mPhase, &mTimeSlewing);
      }
    }

    mCalibrator->setCalibTOFapi(mcalibTOFapi);

    auto data = pc.inputs().get<gsl::span<T>>("input");
    LOG(info) << "Processing TF " << tfcounter << " with " << data.size() << " tracks";

    if (mUseCCDB) { // setting the timestamp to get the LHCPhase correction; if we don't use CCDB, then it can stay to 0 as set when creating the calibTOFapi above
      const auto tfOrbitFirst = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
      mcalibTOFapi->setTimeStamp(0.001 * (o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS() + tfOrbitFirst * o2::constants::lhc::LHCOrbitMUS * 0.001)); // in seconds
    }

    mCalibrator->process(data);

    sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<o2::tof::TOFChannelCalibrator<T>> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  o2::tof::CalibTOFapi* mcalibTOFapi = nullptr;
  LHCphase mPhase;
  TimeSlewing mTimeSlewing;
  float mRange = 24000.f;
  bool mUseCCDB = false;
  bool mAttachToLHCphase = false; // whether to use or not previously defined LHCphase
  bool mCosmics = false;
  bool mDoPerStrip = false;
  bool mSafeMode = false;
  bool mUpdateCCDB = false;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h
    using clbUtils = o2::calibration::Utils;
    const auto& payloadVec = mCalibrator->getTimeSlewingVector();
    auto& infoVec = mCalibrator->getTimeSlewingInfoVector(); // use non-const version as we update it
    assert(payloadVec.size() == infoVec.size());

    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
      LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();

      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_CHANCALIB", i}, *image.get()); // vector<char>
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_CHANCALIB", i}, w);            // root-serialized
    }
    if (payloadVec.size()) {
      mCalibrator->initOutput(); // reset the outputs once they are already sent
    }
  }
};

} // namespace calibration

namespace framework
{

template <class T>
DataProcessorSpec getTOFChannelCalibDeviceSpec(bool useCCDB, bool attachChannelOffsetToLHCphase = false, bool isCosmics = false, bool perstrip = false, bool safe = false)
{
  using device = o2::calibration::TOFChannelCalibDevice<T>;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_CHANCALIB"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_CHANCALIB"}, Lifetime::Sporadic);

  std::vector<InputSpec> inputs;
  if (!isCosmics) {
    inputs.emplace_back("input", "TOF", "CALIBDATA");
  } else {
    inputs.emplace_back("input", "TOF", "INFOCALCLUS");
  }
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  if (useCCDB) {
    inputs.emplace_back("tofccdbLHCphase", o2::header::gDataOriginTOF, "LHCphase", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/LHCphase"));
    inputs.emplace_back("tofccdbChannelCalib", o2::header::gDataOriginTOF, "ChannelCalib", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/ChannelCalib"));
  }

  return DataProcessorSpec{
    "calib-tofchannel-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest, useCCDB, attachChannelOffsetToLHCphase, isCosmics, perstrip, safe)},
    Options{
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit channel histos"}},
      {"nbins", VariantType::Int, 1000, {"number of bins for t-texp"}},
      {"range", VariantType::Float, 24000.f, {"range for t-text"}},
      {"do-TOF-channel-calib-in-test-mode", VariantType::Bool, false, {"to run in test mode for simplification"}},
      {"ccdb-path", VariantType::String, o2::base::NameConf::getCCDBServer(), {"Path to CCDB"}},
      {"update-at-end-of-run-only", VariantType::Bool, false, {"to update the CCDB only at the end of the run; has priority over calibrating in time slots"}},
      {"tf-per-slot", VariantType::UInt32, 0u, {"number of TFs per calibration time slot, if 0: close once statistics reached"}},
      {"max-delay", VariantType::UInt32, 0u, {"number of slots in past to consider"}},
      {"update-interval", VariantType::UInt32, 10u, {"number of TF after which to try to finalize calibration"}},
      {"delta-update-interval", VariantType::UInt32, 10u, {"number of TF after which to try to finalize calibration, if previous attempt failed"}}}};
}

} // namespace framework
} // namespace o2

#endif
