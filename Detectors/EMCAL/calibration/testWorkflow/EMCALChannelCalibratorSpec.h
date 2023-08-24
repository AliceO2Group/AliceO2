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
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/Configuration.h"

// for time measurements
#include <chrono>
#include <random>
#include <optional>

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class CalibInputDownsampler
{
 public:
  CalibInputDownsampler() { initSeed(); }
  ~CalibInputDownsampler() = default;

  void setSamplingFraction(float samplingFraction) { mSamplingFraction = samplingFraction; }
  bool acceptEvent()
  {
    auto rnr = mSampler(mRandomGenerator);
    return rnr < mSamplingFraction;
  }

 private:
  void initSeed()
  {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::seed_seq randomseed{uint32_t(now & 0xffffffff), uint32_t(now >> 32)};
    mRandomGenerator.seed(randomseed);
  }
  std::mt19937_64 mRandomGenerator;
  std::uniform_real_distribution<float> mSampler{0, 1};
  float mSamplingFraction = 1;
};

class EMCALChannelCalibDevice : public o2::framework::Task
{

  using EMCALCalibParams = o2::emcal::EMCALCalibParams;

 public:
  EMCALChannelCalibDevice(std::shared_ptr<o2::base::GRPGeomRequest> req, bool params, std::string calibType, bool rejCalibTrg, bool rejL0Trig, bool applyGainCalib) : mCCDBRequest(req), mLoadCalibParamsFromCCDB(params), mCalibType(calibType), mRejectCalibTriggers(rejCalibTrg), mRejectL0Triggers(rejL0Trig), mApplyGainCalib(applyGainCalib) {}

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
      mTimeCalibrator->setSavedSlotAllowed(EMCALCalibParams::Instance().setSavedSlotAllowed_EMC);
      mTimeCalibrator->setLoadAtSOR(EMCALCalibParams::Instance().setSavedSlotAllowedSOR_EMC);
      mTimeCalibrator->setSaveFileName("emc-time-calib.root");
    } else { // bad cell calibration
      isBadChannelCalib = true;
      if (!mBadChannelCalibrator) {
        mBadChannelCalibrator = std::make_unique<o2::emcal::EMCALChannelCalibrator<o2::emcal::EMCALChannelData, o2::emcal::BadChannelMap>>();
      }
      mBadChannelCalibrator->SetCalibExtractor(mCalibExtractor);
      mBadChannelCalibrator->setSavedSlotAllowed(EMCALCalibParams::Instance().setSavedSlotAllowed_EMC);
      mBadChannelCalibrator->setLoadAtSOR(EMCALCalibParams::Instance().setSavedSlotAllowedSOR_EMC);
      mBadChannelCalibrator->setSaveFileName("emc-channel-calib.root");
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
    }
    if (matcher == ConcreteDataMatcher("EMC", "SCALEFACTORS", 0)) {
      if (mBadChannelCalibrator && EMCALCalibParams::Instance().useScaledHisto_bc) {
        LOG(info) << "Configuring scale factors for bad channel map";
        mBadChannelCalibrator->getCalibExtractor()->setBCMScaleFactors(reinterpret_cast<o2::emcal::EMCALChannelScaleFactors*>(obj));
        mScaleFactorsInitialized = true;
      }
    }
    if (mApplyGainCalib && matcher == ConcreteDataMatcher("EMC", "EMCGAINCALIB", 0)) {
      if (mBadChannelCalibrator) {
        LOG(info) << "Configuring gain calib factors for bad channel";
        if (mBadChannelCalibrator->setGainCalibrationFactors(reinterpret_cast<o2::emcal::GainCalibrationFactors*>(obj))) {
          mGainCalibFactorsInitialized = true;
        }
      }
      if (mTimeCalibrator) {
        LOG(info) << "Configuring gain calib factors for time calib";
        if (mTimeCalibrator->setGainCalibrationFactors(reinterpret_cast<o2::emcal::GainCalibrationFactors*>(obj))) {
          mGainCalibFactorsInitialized = true;
        }
      }
    }
    if (mRejectL0Triggers && matcher == ConcreteDataMatcher("CTP", "CTPCONFIG", 0)) {
      // clear current class mask and prepare to fill in the updated values
      // The trigger names are seperated by a ":" in one string in the calib params
      mSelectedClassMasks.clear();
      std::string strSelClassMasks = EMCALCalibParams::Instance().selectedClassMasks;
      std::string delimiter = ":";
      size_t pos = 0;
      std::vector<std::string> vSelMasks;
      while ((pos = strSelClassMasks.find(delimiter)) != std::string::npos) {
        vSelMasks.push_back(strSelClassMasks.substr(0, pos));
        strSelClassMasks.erase(0, pos + delimiter.length());
      }
      vSelMasks.push_back(strSelClassMasks);

      auto ctpconf = reinterpret_cast<o2::ctp::CTPConfiguration*>(obj);

      for (auto& cls : ctpconf->getCTPClasses()) {
        LOG(debug) << "CTP class: " << cls.name << "\t " << cls.classMask;

        if (std::find(vSelMasks.begin(), vSelMasks.end(), cls.name) != vSelMasks.end()) {
          mSelectedClassMasks.push_back(cls.classMask);
          LOG(info) << "Setting selected class mask " << cls.name << " to bit " << cls.classMask;
        }
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

    // prepare CTPConfiguration such that it can be loaded in finalise ccdb
    if (mRejectL0Triggers) {
      pc.inputs().get<o2::ctp::CTPConfiguration*>(getCTPConfigBinding());
    }

    if (!mIsConfigured) {
      // configure calibrators (after calib params are loaded from the CCDB)
      configureCalibrators();
      mIsConfigured = true;
    }

    if (mApplyGainCalib && !mGainCalibFactorsInitialized) {
      // process dummy data with no cells to create a slot
      std::vector<o2::emcal::Cell> cellDummyData(0);
      if (isBadChannelCalib) {
        mBadChannelCalibrator->process(cellDummyData);
      } else {
        mTimeCalibrator->process(cellDummyData);
      }
      // for reading the calib objects from the CCDB
      pc.inputs().get<o2::emcal::GainCalibrationFactors*>(getGainCalibBinding());
    }

    float samplingFraction = isBadChannelCalib ? EMCALCalibParams::Instance().fractionEvents_bc : EMCALCalibParams::Instance().fractionEvents_tc;
    if (samplingFraction < 1) {
      if (!mDownsampler) {
        mDownsampler = std::make_unique<CalibInputDownsampler>();
      }
      mDownsampler->setSamplingFraction(samplingFraction);
    }

    using ctpDigitsType = std::decay_t<decltype(pc.inputs().get<gsl::span<o2::ctp::CTPDigit>>(getCTPDigitsBinding()))>;
    std::optional<ctpDigitsType> ctpDigits;
    if (mRejectL0Triggers) {
      ctpDigits = pc.inputs().get<gsl::span<o2::ctp::CTPDigit>>(getCTPDigitsBinding());
    }

    // reset EOR behaviour
    if (mTimeCalibrator) {
      if (mTimeCalibrator->getSaveAtEOR())
        mTimeCalibrator->setSaveAtEOR(false);
    } else if (mBadChannelCalibrator) {
      if (mBadChannelCalibrator->getSaveAtEOR())
        mBadChannelCalibrator->setSaveAtEOR(false);
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

      // reject all triggers that are not included in the classMask (typically only EMC min. bias should be accepted)
      uint64_t classMaskCTP = 0;
      if (mRejectL0Triggers) {
        bool acceptEvent = false;
        // Match the EMCal bc to the CTP bc
        int64_t bcEMC = trg.getBCData().toLong();
        for (auto& ctpDigit : *ctpDigits) {
          int64_t bcCTP = ctpDigit.intRecord.toLong();
          LOG(debug) << "bcEMC " << bcEMC << "   bcCTP " << bcCTP;
          if (bcCTP == bcEMC) {
            // obtain trigger mask that belongs to the selected bc
            classMaskCTP = ctpDigit.CTPClassMask.to_ulong();
            // now check if min bias trigger is not in mask
            for (const uint64_t& selectedClassMask : mSelectedClassMasks) {
              if ((classMaskCTP & selectedClassMask) != 0) {
                LOG(debug) << "trigger " << selectedClassMask << " found! accepting event";
                acceptEvent = true;
                break;
              }
            }
            break; // break as bc was matched
          }
        }
        // if current event is not accepted (selected triggers not present), move on to next event
        if (!acceptEvent) {
          continue;
        }
      }

      if (mDownsampler && !mDownsampler->acceptEvent()) {
        continue;
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
    static bool firstCall = true;
    if (firstCall) {
      firstCall = false;
      if (mTimeCalibrator) {
        mTimeCalibrator->loadSavedSlot();
      } else if (mBadChannelCalibrator) {
        mBadChannelCalibrator->loadSavedSlot();
      }
    }

    if (pc.transitionState() == TransitionHandlingState::Requested) {
      LOG(debug) << "Run stop requested, finalizing";
      // mRunStopRequested = true;
      if (isBadChannelCalib) {
        mBadChannelCalibrator->setSaveAtEOR(true);
        mBadChannelCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
      } else {
        mTimeCalibrator->setSaveAtEOR(true);
        mTimeCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
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
      mBadChannelCalibrator->setSaveAtEOR(true);
      mBadChannelCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
      sendOutput<o2::emcal::BadChannelMap>(ec.outputs());
    } else {
      mTimeCalibrator->setSaveAtEOR(true);
      mTimeCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
      sendOutput<o2::emcal::TimeCalibrationParams>(ec.outputs());
    }
  }

  static const char* getCellBinding() { return "EMCCells"; }
  static const char* getCellTriggerRecordBinding() { return "EMCCellsTrgR"; }
  static const char* getCTPDigitsBinding() { return "CTPDigits"; }
  static const char* getCTPConfigBinding() { return "CTPConfig"; }
  static const char* getGainCalibBinding() { return "EMCGainCalib"; }

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
  bool mRejectL0Triggers = true;                                                                                                       ///! reject EMCal Gamma and Jet triggers in the online calibration
  bool mApplyGainCalib = true;                                                                                                         ///! switch if gain calibration should be applied during filling of histograms or not
  bool mGainCalibFactorsInitialized = false;                                                                                           ///! Gain calibration init status
  std::array<double, 2> timeMeas;                                                                                                      ///! Used for time measurement and holds the start and end time in the run function
  std::vector<uint64_t> mSelectedClassMasks = {};                                                                                      ///! EMCal minimum bias trigger bit. Only this bit will be used for calibration
  std::unique_ptr<CalibInputDownsampler> mDownsampler;

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
}; // namespace calibration

} // namespace calibration

namespace framework
{

DataProcessorSpec getEMCALChannelCalibDeviceSpec(const std::string calibType, const bool loadCalibParamsFromCCDB, const bool rejectCalibTrigger, const bool rejectL0Trigger, const bool ctpcfgperrun, const bool applyGainCalib)
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
  // inputs.emplace_back("EMCTriggers", "EMC", "CELLSTRGR", 0, Lifetime::Timeframe)
  // for loading the channelCalibParams from the ccdb
  if (loadCalibParamsFromCCDB) {
    inputs.emplace_back("EMC_CalibParam", o2::header::gDataOriginEMC, "EMCALCALIBPARAM", 0, Lifetime::Condition, ccdbParamSpec("EMC/Config/CalibParam"));
  }
  if (calibType.find("badchannel") != std::string::npos) {
    inputs.emplace_back("EMC_Scalefactors", o2::header::gDataOriginEMC, "SCALEFACTORS", 0, Lifetime::Condition, ccdbParamSpec(CalibDB::getCDBPathChannelScaleFactors()));
  }
  if (applyGainCalib) {
    inputs.emplace_back(device::getGainCalibBinding(), o2::header::gDataOriginEMC, "EMCGAINCALIB", 0, Lifetime::Condition, ccdbParamSpec("EMC/Calib/GainCalibFactors"));
  }

  // data request needed for rejection of EMCal trigger
  if (rejectL0Trigger) {
    inputs.emplace_back(device::getCTPConfigBinding(), "CTP", "CTPCONFIG", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/Config", ctpcfgperrun));
    inputs.emplace_back(device::getCTPDigitsBinding(), "CTP", "DIGITS", 0, Lifetime::Timeframe);
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
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest, loadCalibParamsFromCCDB, calibType, rejectCalibTrigger, rejectL0Trigger, applyGainCalib)},
    Options{}};
}

} // namespace framework
} // namespace o2

#endif
