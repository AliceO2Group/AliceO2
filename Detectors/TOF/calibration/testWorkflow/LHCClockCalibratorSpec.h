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
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsRaw/HBFUtils.h"

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
  void init(o2::framework::InitContext& ic) final
  {
    int minEnt = std::max(300, ic.options().get<int>("min-entries"));
    int nb = std::max(500, ic.options().get<int>("nbins"));
    int slotL = ic.options().get<int>("tf-per-slot");
    int delay = ic.options().get<int>("max-delay");
    mCalibrator = std::make_unique<o2::tof::LHCClockCalibrator>(minEnt, nb);
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setMaxSlotsDelay(delay);

    // calibration objects set to zero
    mPhase.addLHCphase(0, 0);
    mPhase.addLHCphase(2000000000, 0);

    TFile* fsleewing = TFile::Open("localTimeSlewing.root");
    if (fsleewing) {
      TimeSlewing* ob = (TimeSlewing*)fsleewing->Get("ccdb_object");
      mTimeSlewing = *ob;
      return;
    }

    for (int ich = 0; ich < TimeSlewing::NCHANNELS; ich++) {
      mTimeSlewing.addTimeSlewingInfo(ich, 0, 0);
      int sector = ich / TimeSlewing::NCHANNELXSECTOR;
      int channelInSector = ich % TimeSlewing::NCHANNELXSECTOR;
      mTimeSlewing.setFractionUnderPeak(sector, channelInSector, 1);
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto data = pc.inputs().get<gsl::span<o2::dataformats::CalibInfoTOF>>("input");
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());

    if (!mcalibTOFapi) {
      mcalibTOFapi = new o2::tof::CalibTOFapi(long(0), &mPhase, &mTimeSlewing); // TODO: should we replace long(0) with tfcounter defined at the beginning of the method? we need the timestamp of the TF
      mCalibrator->setCalibTOFapi(mcalibTOFapi);
    }
    if (data.size() == 0) {
      return;
    }
    mcalibTOFapi->setTimeStamp(data[0].getTimestamp());

    LOG(info) << "Processing TF " << mCalibrator->getCurrentTFInfo().tfCounter << " with " << data.size() << " tracks";
    mCalibrator->process(data);
    sendOutput(pc.outputs());
    const auto& infoVec = mCalibrator->getLHCphaseInfoVector();
    LOG(info) << "Created " << infoVec.size() << " objects for TF " << mCalibrator->getCurrentTFInfo().tfCounter;
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(info) << "Finalizing calibration";
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  o2::tof::CalibTOFapi* mcalibTOFapi = nullptr;
  LHCphase mPhase;
  TimeSlewing mTimeSlewing;
  std::unique_ptr<o2::tof::LHCClockCalibrator> mCalibrator;

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

DataProcessorSpec getLHCClockCalibDeviceSpec()
{
  using device = o2::calibration::LHCClockCalibDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_LHCphase"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_LHCphase"}, Lifetime::Sporadic);
  return DataProcessorSpec{
    "calib-lhcclock-calibration",
    Inputs{{"input", "TOF", "CALIBDATA"}},
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"tf-per-slot", VariantType::Int, 5, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int, 3, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit single time slot"}},
      {"nbins", VariantType::Int, 1000, {"number of bins for "}}}};
}

} // namespace framework
} // namespace o2

#endif
