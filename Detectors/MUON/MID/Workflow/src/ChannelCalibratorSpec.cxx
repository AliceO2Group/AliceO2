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

/// \file   MID/Workflow/src/ChannelCalibratorSpec.cxx
/// \brief  Noise and dead channels calibrator spec for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   21 February 2022

#include "MIDWorkflow/ChannelCalibratorSpec.h"

#include <array>
#include <vector>
#include <gsl/gsl>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
#include "Framework/DeviceStateEnums.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDCalibration/ChannelCalibrator.h"
#include "MIDCalibration/ChannelCalibratorParam.h"
#include "MIDCalibration/ChannelCalibratorFinalizer.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#include "MIDFiltering/MaskMaker.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/ROBoardConfigHandler.h"
#include "DetectorsBase/GRPGeomHelper.h"

#include "TObjString.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class ChannelCalibratorDeviceDPL
{
 public:
  ChannelCalibratorDeviceDPL(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req)
  {
    mRefMasks = makeDefaultMasksFromCrateConfig(feeIdConfig, crateMasks);
  }

  void init(of::InitContext& ic)
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mCalibrators[0].setThreshold(ChannelCalibratorParam::Instance().maxNoise);
    mCalibrators[1].setThreshold(ChannelCalibratorParam::Instance().maxDead);

    for (auto& calib : mCalibrators) {
      calib.setSlotLength(calibration::INFINITE_TF);
      calib.setUpdateAtTheEndOfRunOnly();
    }
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(of::ProcessingContext& pc)
  {
    if (mRunStopRequested) {
      return;
    }
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);

    std::array<gsl::span<const ColumnData>, 2> calibData{pc.inputs().get<gsl::span<ColumnData>>("mid_noise"), pc.inputs().get<gsl::span<ColumnData>>("mid_dead")};
    auto deadRof = pc.inputs().get<gsl::span<ROFRecord>>("mid_dead_rof");
    std::array<double, 2> timeOrTriggers{o2::base::GRPGeomHelper::instance().getGRPECS()->getNHBFPerTF() * o2::constants::lhc::LHCOrbitNS * 1e-9, static_cast<double>(deadRof.size())};

    for (size_t idx = 0; idx < calibData.size(); ++idx) {
      o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrators[idx].getCurrentTFInfo());
      mCalibrators[idx].addTimeOrTriggers(timeOrTriggers[idx]);
      mCalibrators[idx].process(calibData[idx]);
    }
    if (pc.transitionState() == of::TransitionHandlingState::Requested) {
      LOG(info) << "Run stop requested, finalizing";
      mRunStopRequested = true;
      finalise(pc.outputs());
    }
  }

  void endOfStream(of::EndOfStreamContext& ec)
  {
    if (mRunStopRequested) {
      return;
    }
    LOG(info) << "EOS sent, finalizing";
    finalise(ec.outputs());
  }

 private:
  std::array<ChannelCalibrator, 2> mCalibrators{};        ///! Channels calibrators
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest; ///! CCDB request
  std::vector<ColumnData> mRefMasks{};                    ///! Reference masks
  bool mRunStopRequested = false;                         ///! Flag that the run was stopped

  void finalise(of::DataAllocator& output)
  {
    for (auto& calib : mCalibrators) {
      calib.checkSlotsToFinalize(ChannelCalibrator::INFINITE_TF);
    }
    sendOutput(output);
  }

  void sendOutput(of::DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output

    output.snapshot(of::Output{header::gDataOriginMID, "NOISY_CHANNELS", 0}, mCalibrators[0].getBadChannels());
    output.snapshot(of::Output{header::gDataOriginMID, "DEAD_CHANNELS", 0}, mCalibrators[1].getBadChannels());

    ChannelCalibratorFinalizer finalizer;
    finalizer.setReferenceMasks(mRefMasks);
    finalizer.process(mCalibrators[0].getBadChannels(), mCalibrators[1].getBadChannels());
    sendOutput(output, finalizer.getBadChannels(), mCalibrators[1].getCurrentTFInfo(), "MID/Calib/BadChannels", 0);

    TObjString masks(finalizer.getMasksAsString().c_str());
    sendOutput(output, masks, mCalibrators[1].getCurrentTFInfo(), "MID/Calib/ElectronicsMasks", 1);

    output.snapshot(of::Output{header::gDataOriginMID, "BAD_CHANNELS", 0}, finalizer.getBadChannels());

    for (auto& calib : mCalibrators) {
      calib.initOutput(); // reset the outputs once they are already sent
    }
  }

  template <typename T>
  void sendOutput(of::DataAllocator& output, const T& payload, const o2::dataformats::TFIDInfo& tfInfo, const char* path, header::DataHeader::SubSpecificationType subSpec)
  {
    o2::ccdb::CcdbObjectInfo info;
    std::map<std::string, std::string> md;
    o2::calibration::Utils::prepareCCDBobjectInfo(payload, info, path, md, tfInfo.creation, tfInfo.creation + 5 * o2::ccdb::CcdbObjectInfo::DAY);
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(of::Output{o2::calibration::Utils::gDataOriginCDBPayload, "MID_BAD_CHANNELS", subSpec}, *image.get()); // vector<char>
    output.snapshot(of::Output{o2::calibration::Utils::gDataOriginCDBWrapper, "MID_BAD_CHANNELS", subSpec}, info);         // root-serialized
  }
};

of::DataProcessorSpec getChannelCalibratorSpec(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks)
{
  std::vector<of::InputSpec> inputSpecs;
  inputSpecs.emplace_back("mid_noise", header::gDataOriginMID, "NOISE");
  inputSpecs.emplace_back("mid_noise_rof", header::gDataOriginMID, "NOISEROF");
  inputSpecs.emplace_back("mid_dead", header::gDataOriginMID, "DEAD");
  inputSpecs.emplace_back("mid_dead_rof", header::gDataOriginMID, "DEADROF");
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputSpecs);
  std::vector<of::OutputSpec> outputSpecs;
  outputSpecs.emplace_back(o2::calibration::Utils::gDataOriginCDBPayload, "MID_BAD_CHANNELS", 0, of::Lifetime::Sporadic);
  outputSpecs.emplace_back(o2::calibration::Utils::gDataOriginCDBWrapper, "MID_BAD_CHANNELS", 0, of::Lifetime::Sporadic);
  outputSpecs.emplace_back(o2::calibration::Utils::gDataOriginCDBPayload, "MID_BAD_CHANNELS", 1, of::Lifetime::Sporadic);
  outputSpecs.emplace_back(o2::calibration::Utils::gDataOriginCDBWrapper, "MID_BAD_CHANNELS", 1, of::Lifetime::Sporadic);
  outputSpecs.emplace_back(header::gDataOriginMID, "NOISY_CHANNELS", 0, of::Lifetime::Sporadic);
  outputSpecs.emplace_back(header::gDataOriginMID, "DEAD_CHANNELS", 0, of::Lifetime::Sporadic);
  outputSpecs.emplace_back(header::gDataOriginMID, "BAD_CHANNELS", 0, of::Lifetime::Sporadic);

  return of::DataProcessorSpec{
    "MIDChannelCalibrator",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::ChannelCalibratorDeviceDPL>(feeIdConfig, crateMasks, ccdbRequest)}};
}
} // namespace mid
} // namespace o2
