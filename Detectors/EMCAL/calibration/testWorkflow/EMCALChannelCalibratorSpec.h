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

/// \class EMCALChannelCalibratorSpec
/// \brief DPL Processor for EMCAL bad channel calibration data
/// \author Hannah Bossi, Yale University
/// \ingroup EMCALCalib
/// \since Feb 11, 2021

#ifndef O2_CALIBRATION_EMCALCHANNEL_CALIBRATOR_H
#define O2_CALIBRATION_EMCALCHANNEL_CALIBRATOR_H

#include "EMCALCalibration/EMCALChannelCalibrator.h"
#include "EMCALCalibration/EMCALCalibParams.h"
#include "DetectorsCalibration/Utils.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "CommonUtils/MemFileHelper.h"
#include "CommonConstants/Triggers.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/CCDBParamSpec.h"

// for time measurements
#include <chrono>

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class EMCALChannelCalibDevice : public o2::framework::Task
{

  using EMCALCalibParams = o2::emcal::EMCALCalibParams;

 public:
  EMCALChannelCalibDevice(std::shared_ptr<o2::base::GRPGeomRequest> req, bool params, std::string calibType, bool rejCalibTrg) : mCCDBRequest(req), mLoadCalibParamsFromCCDB(params), mCalibType(calibType), mRejectCalibTriggers(rejCalibTrg) {}

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);

    mCalibExtractor = std::make_shared<o2::emcal::EMCALCalibExtractor>();

    if (mCalibType.find("time") != std::string::npos) { // time calibration
      isBadChannelCalib = false;
      if (!mTimeCalibrator) {
        mTimeCalibrator = std::make_unique<o2::emcal::EMCALChannelCalibrator<o2::emcal::EMCALTimeCalibData, o2::emcal::TimeCalibrationParams>>();
      }
      mTimeCalibrator->SetCalibExtractor(mCalibExtractor);

    } else { // bad cell calibration
      isBadChannelCalib = true;
      if (!mBadChannelCalibrator) {
        mBadChannelCalibrator = std::make_unique<o2::emcal::EMCALChannelCalibrator<o2::emcal::EMCALChannelData, o2::emcal::BadChannelMap>>();
      }
      mBadChannelCalibrator->SetCalibExtractor(mCalibExtractor);
    }
  }

  //_________________________________________________________________
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
    // check if calib params need to be updated
    if (matcher == ConcreteDataMatcher("EMC", "EMCALCALIBPARAM", 0)) {
      LOG(info) << "EMCal CalibParams updated";
      EMCALCalibParams::Instance().printKeyValues(true, true);
      return;
    }
    if (matcher == ConcreteDataMatcher("EMC", "SCALEFACTORS", 0)) {
      if (mBadChannelCalibrator && EMCALCalibParams::Instance().useScaledHisto_bc) {
        LOG(info) << "Configuring scale factors for bad channel map";
        mBadChannelCalibrator->getCalibExtractor()->setBCMScaleFactors(reinterpret_cast<o2::emcal::EMCALChannelScaleFactors*>(obj));
        mScaleFactorsInitialized = true;
      }
    }
  }

  //_________________________________________________________________
  void run(o2::framework::ProcessingContext& pc) final
  {
    if (EMCALCalibParams::Instance().enableTimeProfiling) {
      timeMeas[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    if (mTimeCalibrator) {
      o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mTimeCalibrator->getCurrentTFInfo());
    } else if (mBadChannelCalibrator) {
      o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mBadChannelCalibrator->getCurrentTFInfo());
    }

    if (mLoadCalibParamsFromCCDB) {
      // for reading the calib objects from the CCDB
      pc.inputs().get<o2::emcal::EMCALCalibParams*>("EMC_CalibParam");
    }

    if (mBadChannelCalibrator && EMCALCalibParams::Instance().useScaledHisto_bc && !mScaleFactorsInitialized) {
      // Trigger reading the scale factors from the CCDB (Bad channel calib only)
      pc.inputs().get<o2::emcal::EMCALChannelScaleFactors*>("EMC_Scalefactors");
    }

    if (!mIsConfigured) {
      // configure calibrators (after calib params are loaded from the CCDB)
      configureCalibrators();
      mIsConfigured = true;
    }

    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get(getCellBinding()).header)->startTime;

    auto data = pc.inputs().get<gsl::span<o2::emcal::Cell>>(getCellBinding());

    auto InputTriggerRecord = pc.inputs().get<gsl::span<o2::emcal::TriggerRecord>>(getCellTriggerRecordBinding());
    LOG(debug) << "[EMCALCalibrator - run]  Received " << InputTriggerRecord.size() << " Trigger Records, running calibration ...";

    LOG(debug) << "Processing TF " << tfcounter << " with " << data.size() << " cells";

    // call process for every event in the trigger record to ensure correct event counting for the calibration.
    for (const auto& trg : InputTriggerRecord) {
      if (!trg.getNumberOfObjects()) {
        continue;
      }
      // reject calibration trigger from the calibration
      if (mRejectCalibTriggers) {
        LOG(debug) << "Trigger: " << trg.getTriggerBits() << "   o2::trigger::Cal " << o2::trigger::Cal;
        if (trg.getTriggerBits() & o2::trigger::Cal) {
          LOG(debug) << "skipping triggered events due to wrong trigger (no Physics trigger)";
          continue;
        }
      }

      gsl::span<const o2::emcal::Cell> eventData(data.data() + trg.getFirstEntry(), trg.getNumberOfObjects());

      // fast calibration
      if (EMCALCalibParams::Instance().enableFastCalib) {
        LOG(debug) << "fast calib not yet available!";
        // normal calibration procedure
      } else {
        if (isBadChannelCalib) {
          mBadChannelCalibrator->process(eventData);
        } else {
          mTimeCalibrator->process(eventData);
        }
      }
    }
    if (isBadChannelCalib) {
      sendOutput<o2::emcal::BadChannelMap>(pc.outputs());
    } else {
      sendOutput<o2::emcal::TimeCalibrationParams>(pc.outputs());
    }
    if (EMCALCalibParams::Instance().enableTimeProfiling) {
      timeMeas[1] = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      LOG(info) << "end of run function. Time: " << timeMeas[1] - timeMeas[0] << " [ns] for " << InputTriggerRecord.size() << " events";
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    if (isBadChannelCalib) {
      mBadChannelCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
      sendOutput<o2::emcal::BadChannelMap>(ec.outputs());
    } else {
      mTimeCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
      sendOutput<o2::emcal::TimeCalibrationParams>(ec.outputs());
    }
  }

  static const char* getCellBinding() { return "EMCCells"; }
  static const char* getCellTriggerRecordBinding() { return "EMCCellsTrgR"; }

 private:
  std::unique_ptr<o2::emcal::EMCALChannelCalibrator<o2::emcal::EMCALChannelData, o2::emcal::BadChannelMap>> mBadChannelCalibrator;     ///< Bad channel calibrator
  std::unique_ptr<o2::emcal::EMCALChannelCalibrator<o2::emcal::EMCALTimeCalibData, o2::emcal::TimeCalibrationParams>> mTimeCalibrator; ///< Time calibrator
  std::shared_ptr<o2::emcal::EMCALCalibExtractor> mCalibExtractor;                                                                     ///< Calibration postprocessing
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;                                                                              ///< CCDB request for geometry
  std::string mCalibType;                                                                                                              ///< Name of the calibration type
  bool mIsConfigured = false;                                                                                                          ///< Configure status of calibrators
  bool mScaleFactorsInitialized = false;                                                                                               ///< Scale factor init status
  bool isBadChannelCalib = true;                                                                                                       ///< Calibration mode bad channel calib (false := time calib)
  bool mLoadCalibParamsFromCCDB = true;                                                                                                ///< Switch for loading calib params from the CCDB
  bool mRejectCalibTriggers = true;                                                                                                    ///! reject calibration triggers in the online calibration
  std::array<double, 2> timeMeas;

  //________________________________________________________________
  template <typename DataOutput>
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h
    using clbUtils = o2::calibration::Utils;

    // we need this to be a vector of bad channel maps or time params, we will probably need to create this?
    std::vector<DataOutput> payloadVec;
    std::vector<o2::ccdb::CcdbObjectInfo> infoVec;
    if constexpr (std::is_same<DataOutput, o2::emcal::TimeCalibrationParams>::value) {
      payloadVec = mTimeCalibrator->getOutputVector();
      infoVec = mTimeCalibrator->getInfoVector();
    } else {
      payloadVec = mBadChannelCalibrator->getOutputVector();
      infoVec = mBadChannelCalibrator->getInfoVector();
    }
    // use non-const version as we update it
    assert(payloadVec.size() == infoVec.size());
    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
      if constexpr (std::is_same<DataOutput, o2::emcal::TimeCalibrationParams>::value) {
        LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                  << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
        output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_TIMECALIB", i}, *image.get()); // vector<char>
        output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_TIMECALIB", i}, w);            // root-serialized
      } else {
        LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                  << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
        output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_BADCHANNELS", i}, *image.get()); // vector<char>
        output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_BADCHANNELS", i}, w);            // root-serialized
      }
    }
    if (payloadVec.size()) {
      if constexpr (std::is_same<DataOutput, o2::emcal::TimeCalibrationParams>::value) {
        mTimeCalibrator->initOutput(); // reset the outputs once they are already sent
      } else {
        mBadChannelCalibrator->initOutput(); // reset the outputs once they are already sent
      }
    }
  }

  /// \brief Configure calibrators from the calib params
  void configureCalibrators()
  {
    if (mTimeCalibrator) {
      LOG(info) << "Configuring time calibrator";
      mTimeCalibrator->setSlotLength(EMCALCalibParams::Instance().slotLength_tc);
      if (EMCALCalibParams::Instance().UpdateAtEndOfRunOnly_tc) {
        mBadChannelCalibrator->setUpdateAtTheEndOfRunOnly();
      }
    }
    if (mBadChannelCalibrator) {
      LOG(info) << "Configuring bad channel calibrator";
      mBadChannelCalibrator->setSlotLength(EMCALCalibParams::Instance().slotLength_bc);
      if (EMCALCalibParams::Instance().UpdateAtEndOfRunOnly_bc) {
        mBadChannelCalibrator->setUpdateAtTheEndOfRunOnly();
      }
      mBadChannelCalibrator->setIsTest(EMCALCalibParams::Instance().enableTestMode_bc);
    }
  }
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getEMCALChannelCalibDeviceSpec(const std::string calibType, const bool loadCalibParamsFromCCDB, const bool rejectCalibTrigger)
{
  using device = o2::calibration::EMCALChannelCalibDevice;
  using clbUtils = o2::calibration::Utils;
  using EMCALCalibParams = o2::emcal::EMCALCalibParams;
  using CalibDB = o2::emcal::CalibDB;

  std::vector<OutputSpec> outputs;
  std::string processorName;
  if (calibType.find("time") != std::string::npos) { // time calibration
    processorName = "calib-emcalchannel-time";
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_TIMECALIB"}, Lifetime::Sporadic); // This needs to match with the output!
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_TIMECALIB"}, Lifetime::Sporadic); // This needs to match with the output!
  } else {                                                                                                                             // bad channel calibration
    processorName = "calib-emcalchannel-badchannel";
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_BADCHANNELS"}, Lifetime::Sporadic); // This needs to match with the output!
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_BADCHANNELS"}, Lifetime::Sporadic); // This needs to match with the output!
  }

  std::vector<InputSpec> inputs;
  inputs.emplace_back(device::getCellBinding(), o2::header::gDataOriginEMC, "CELLS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back(device::getCellTriggerRecordBinding(), o2::header::gDataOriginEMC, "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe);
  // for loading the channelCalibParams from the ccdb
  if (loadCalibParamsFromCCDB) {
    inputs.emplace_back("EMC_CalibParam", o2::header::gDataOriginEMC, "EMCALCALIBPARAM", 0, Lifetime::Condition, ccdbParamSpec("EMC/Config/CalibParam"));
  }
  if (calibType.find("badchannel") != std::string::npos) {
    inputs.emplace_back("EMC_Scalefactors", o2::header::gDataOriginEMC, "SCALEFACTORS", 0, Lifetime::Condition, ccdbParamSpec(CalibDB::getCDBPathChannelScaleFactors()));
  }
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    processorName,
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest, loadCalibParamsFromCCDB, calibType, rejectCalibTrigger)},
    Options{}};
}

} // namespace framework
} // namespace o2

#endif