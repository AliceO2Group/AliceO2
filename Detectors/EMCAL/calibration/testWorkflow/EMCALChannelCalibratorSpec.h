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
#include "DetectorsCalibration/Utils.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class EMCALChannelCalibDevice : public o2::framework::Task
{

  //using TimeSlewing = o2::dataformats::CalibTimeSlewingParamEMCAL;
  //using LHCphase = o2::dataformats::CalibLHCphaseEMCAL;

 public:
  EMCALChannelCalibDevice() = default;
  void init(o2::framework::InitContext& ic) final
  {
    mCalibExtractor = std::make_shared<o2::emcal::EMCALCalibExtractor>();

    int isTest = ic.options().get<bool>("do-EMCAL-channel-calib-in-test-mode");
    std::string calibType = ic.options().get<std::string>("calibType");
    minNEntries = ic.options().get<int>("minNEntries");
    std::string localStorePath = ic.options().get<std::string>("localFilePath");

    if (calibType.find("time") != std::string::npos) { // time calibration
      if (!mTimeCalibrator) {
        mTimeCalibrator = std::make_unique<o2::emcal::EMCALChannelCalibrator<o2::emcal::EMCALTimeCalibData, o2::emcal::TimeCalibrationParams, o2::emcal::TimeCalibInitParams>>();
      }
      mTimeCalibrator->SetCalibExtractor(mCalibExtractor);
      mTimeCalibrator->setMinNEntries(minNEntries);
      mTimeCalibrator->setLocalStorePath(localStorePath);
    } else { // bad cell calibration
      if (!mBadChannelCalibrator) {
        mBadChannelCalibrator = std::make_unique<o2::emcal::EMCALChannelCalibrator<o2::emcal::EMCALChannelData, o2::emcal::BadChannelMap, o2::emcal::ChannelCalibInitParams>>();
      }
      mBadChannelCalibrator->SetCalibExtractor(mCalibExtractor);
      mBadChannelCalibrator->setUpdateAtTheEndOfRunOnly();
      mBadChannelCalibrator->setIsTest(isTest);
      if (ic.options().get<bool>("useScaledHistoForBadChannelMap")) {
        mBadChannelCalibrator->getCalibExtractor()->setUseScaledHistoForBadChannels(true);
      }
      mBadChannelCalibrator->setMinNEntries(minNEntries);
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mTimeCalibrator->getCurrentTFInfo());
    auto data = pc.inputs().get<gsl::span<o2::emcal::Cell>>(getCellBinding());
    LOG(debug) << "Processing TF " << mTimeCalibrator->getCurrentTFInfo().tfCounter << " with " << data.size() << " cells";
    mTimeCalibrator->process(data);
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mBadChannelCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput<o2::emcal::BadChannelMap>(ec.outputs());
  }

  static const char* getCellBinding() { return "EMCCells"; }
  static const char* getCellTriggerRecordBinding() { return "EMCCellsTrgR"; }

 private:
  std::unique_ptr<o2::emcal::EMCALChannelCalibrator<o2::emcal::EMCALChannelData, o2::emcal::BadChannelMap, o2::emcal::ChannelCalibInitParams>> mBadChannelCalibrator;
  std::unique_ptr<o2::emcal::EMCALChannelCalibrator<o2::emcal::EMCALTimeCalibData, o2::emcal::TimeCalibrationParams, o2::emcal::TimeCalibInitParams>> mTimeCalibrator;
  std::shared_ptr<o2::emcal::EMCALCalibExtractor> mCalibExtractor;
  int minNEntries; // minimum number of entries in histogram needed for calibration

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
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getEMCALChannelCalibDeviceSpec(const int minNEntries, std::string calibType = "badcell", std::string localStorePath = "")
{
  using device = o2::calibration::EMCALChannelCalibDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_BADCHANNELS"}, Lifetime::Sporadic); // This needs to match with the output!
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_BADCHANNELS"}, Lifetime::Sporadic); // This needs to match with the output!

  std::vector<InputSpec> inputs;
  inputs.emplace_back(device::getCellBinding(), o2::header::gDataOriginEMC, "CELLS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back(device::getCellTriggerRecordBinding(), o2::header::gDataOriginEMC, "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe);
  return DataProcessorSpec{
    "calib-emcalchannel-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"do-EMCAL-channel-calib-in-test-mode", VariantType::Bool, false, {"to run in test mode for simplification"}},
      {"ccdb-path", VariantType::String, o2::base::NameConf::getCCDBServer(), {"Path to CCDB"}},
      {"minNEntries", VariantType::Int, minNEntries, {"minimum number of entries to trigger calibration"}},
      {"localFilePath", VariantType::String, localStorePath, {"path to file for local storage of TC params"}},
      {"calibType", VariantType::String, calibType, {"switch between time and bad cell calib"}},
      {"useScaledHistoForBadChannelMap", VariantType::Bool, false, {"Use scaled histogram for bad channel extraction"}}}};
}

} // namespace framework
} // namespace o2

#endif
