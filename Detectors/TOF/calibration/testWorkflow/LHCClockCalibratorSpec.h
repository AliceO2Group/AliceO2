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

#ifndef O2_CALIBRATION_LHCCLOCK_CALIBRATOR_H
#define O2_CALIBRATION_LHCCLOCK_CALIBRATOR_H

/// @file   LHCClockCalibratorSpec.h
/// @brief  Device to calibrate LHC clock phase using TOF data

#include "TOFCalibration/LHCClockCalibrator.h"
#include "DetectorsCalibration/Utils.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsBase/GRPGeomHelper.h"

#include <TSystem.h>

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class LHCClockCalibDevice : public o2::framework::Task
{
  using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;
  using LHCphase = o2::dataformats::CalibLHCphaseTOF;

 public:
  LHCClockCalibDevice(std::shared_ptr<o2::base::GRPGeomRequest> req, bool useCCDB) : mCCDBRequest(req), mUseCCDB(useCCDB) {}

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    int minEnt = std::max(300, ic.options().get<int>("min-entries"));
    int nb = std::max(500, ic.options().get<int>("nbins"));
    auto slotL = ic.options().get<uint32_t>("tf-per-slot");
    auto delay = ic.options().get<uint32_t>("max-delay");
    mCalibrator = std::make_unique<o2::tof::LHCClockCalibrator>(minEnt, nb);
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setMaxSlotsDelay(delay);

    if (!mUseCCDB) {
      // calibration objects set to zero
      mPhase.addLHCphase(0, 0);
      mPhase.addLHCphase(o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP_SECONDS, 0);
      if (gSystem->AccessPathName("localTimeSlewing.root") == false) {
        TFile* fsleewing = TFile::Open("localTimeSlewing.root");
        if (fsleewing) {
          TimeSlewing* ob = (TimeSlewing*)fsleewing->Get("ccdb_object");
          mTimeSlewing = *ob;
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
  }

  void run(o2::framework::ProcessingContext& pc) final
  {

    LOG(info) << "We are running LHCPhase";
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    auto data = pc.inputs().get<gsl::span<o2::dataformats::CalibInfoTOF>>("input");
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());

    if (mUseCCDB) { // read calibration objects from ccdb with the CCDB fetcher
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
    LOG(info) << "Data size = " << data.size();
    if (data.size() == 0) {
      return;
    }

    if (mUseCCDB) { // setting the timestamp to get the LHCPhase correction; if we don't use CCDB, then it can stay to 0 as set when creating the calibTOFapi above
      const auto tfOrbitFirst = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
      mcalibTOFapi->setTimeStamp(0.001 * (o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS() + tfOrbitFirst * o2::constants::lhc::LHCOrbitMUS * 0.001)); // in seconds
    }

    LOG(info) << "Processing TF " << mCalibrator->getCurrentTFInfo().tfCounter << " with " << data.size() << " tracks";
    mCalibrator->process(data);
    sendOutput(pc.outputs());
    const auto& infoVec = mCalibrator->getLHCphaseInfoVector();
    LOG(info) << "Created " << infoVec.size() << " objects for TF " << mCalibrator->getCurrentTFInfo().tfCounter;
  }

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

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(info) << "Finalizing calibration";
    mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  o2::tof::CalibTOFapi* mcalibTOFapi = nullptr;
  LHCphase mPhase;
  TimeSlewing mTimeSlewing;
  std::unique_ptr<o2::tof::LHCClockCalibrator> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  bool mUpdateCCDB = false;
  bool mUseCCDB = true;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h
    using clbUtils = o2::calibration::Utils;
    const auto& payloadVec = mCalibrator->getLHCphaseVector();
    auto& infoVec = mCalibrator->getLHCphaseInfoVector(); // use non-const version as we update it
    assert(payloadVec.size() == infoVec.size());

    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
      LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_LHCphase", i}, *image.get()); // vector<char>
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_LHCphase", i}, w);            // root-serialized
    }
    if (payloadVec.size()) {
      mCalibrator->initOutput(); // reset the outputs once they are already sent
    }
  }
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getLHCClockCalibDeviceSpec(bool useCCDB)
{
  using device = o2::calibration::LHCClockCalibDevice;
  using clbUtils = o2::calibration::Utils;
  std::vector<InputSpec> inputs{{"input", "TOF", "CALIBDATA"}};
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  if (useCCDB) {
    inputs.emplace_back("tofccdbLHCphase", "TOF", "LHCphaseCal", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/LHCphase"));
    inputs.emplace_back("tofccdbChannelCalib", "TOF", "ChannelCalibCal", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/ChannelCalib"));
    inputs.emplace_back("orbitResetTOF", o2::header::gDataOriginCTP, "ORBITRESETTOF", 0, Lifetime::Condition, ccdbParamSpec("CTP/Calib/OrbitReset"));
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_LHCphase"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_LHCphase"}, Lifetime::Sporadic);
  return DataProcessorSpec{
    "calib-lhcclock-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest, useCCDB)},
    Options{
      {"tf-per-slot", VariantType::UInt32, 5u, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::UInt32, 3u, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit single time slot"}},
      {"nbins", VariantType::Int, 1000, {"number of bins for "}}}};
}

} // namespace framework
} // namespace o2

#endif
