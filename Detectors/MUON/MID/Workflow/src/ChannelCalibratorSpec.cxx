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
#include "Framework/InputRecordWalker.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDCalibration/ChannelCalibrator.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#include "MIDFiltering/MaskMaker.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/ROBoardConfigHandler.h"

#include "TObjString.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class ChannelCalibratorDeviceDPL
{
 public:
  ChannelCalibratorDeviceDPL(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks)
  {
    mCalibrator.setReferenceMasks(makeDefaultMasksFromCrateConfig(feeIdConfig, crateMasks));
  }

  void init(of::InitContext& ic)
  {
    mThreshold = ic.options().get<double>("mid-mask-threshold");

    int slotL = ic.options().get<int>("tf-per-slot");
    int delay = ic.options().get<int>("max-delay");

    mCalibrator.setSlotLength(slotL);
    mCalibrator.setMaxSlotsDelay(delay);
    mCalibrator.setUpdateAtTheEndOfRunOnly();
  }

  void run(of::ProcessingContext& pc)
  {
    auto noiseRof = pc.inputs().get<gsl::span<ROFRecord>>("mid_noise_rof");
    unsigned long nEvents = noiseRof.size();
    if (nEvents == 0) {
      return;
    }
    auto deadRof = pc.inputs().get<gsl::span<ROFRecord>>("mid_dead_rof");
    auto noise = pc.inputs().get<gsl::span<ColumnData>>("mid_noise");
    auto dead = pc.inputs().get<gsl::span<ColumnData>>("mid_dead");

    auto tfCounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("mid_noise").header)->startTime;

    std::vector<ColumnData> calibData;
    calibData.insert(calibData.end(), noise.begin(), noise.end());
    calibData.insert(calibData.end(), dead.begin(), dead.end());

    mCalibrator.addEvents(nEvents);
    mCalibrator.process(tfCounter, calibData);
  }

  void endOfStream(of::EndOfStreamContext& ec)
  {
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mCalibrator.checkSlotsToFinalize(INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  ChannelCalibrator mCalibrator{};     ///< Calibrator
  std::vector<ColumnData> mRefMasks{}; ///< Reference masks
  double mThreshold{0.9};              ///< Occupancy threshold for producing a mask

  void sendOutput(of::DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h
    sendOutput(output, mCalibrator.getBadChannels(), "MID/BadChannels", 0);

    TObjString masks(mCalibrator.getMasksAsString().c_str());
    sendOutput(output, masks, "MID/ElectronicsMasks", 1);

    mCalibrator.initOutput(); // reset the outputs once they are already sent
  }

  template <typename T>
  void sendOutput(of::DataAllocator& output, T& payload, const char* path, header::DataHeader::SubSpecificationType subSpec)
  {
    o2::ccdb::CcdbObjectInfo info;
    std::map<std::string, std::string> md;
    o2::calibration::Utils::prepareCCDBobjectInfo(payload, info, path, md, mCalibrator.getTFEnd(), o2::calibration::Utils::INFINITE_TIME);
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(of::Output{o2::calibration::Utils::gDataOriginCDBPayload, "MID_BAD_CHANNELS", subSpec}, *image.get()); // vector<char>
    output.snapshot(of::Output{o2::calibration::Utils::gDataOriginCDBWrapper, "MID_BAD_CHANNELS", subSpec}, info);         // root-serialized
  }
};

of::DataProcessorSpec getChannelCalibratorSpec(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks)
{
  constexpr int64_t INFINITE_TF = 0xffffffffffffffff;

  std::vector<of::InputSpec> inputSpecs;
  inputSpecs.emplace_back("mid_noise", header::gDataOriginMID, "NOISE");
  inputSpecs.emplace_back("mid_noise_rof", header::gDataOriginMID, "NOISEROF");
  inputSpecs.emplace_back("mid_dead", header::gDataOriginMID, "DEAD");
  inputSpecs.emplace_back("mid_dead_rof", header::gDataOriginMID, "DEADROF");

  std::vector<of::OutputSpec> outputSpecs;
  outputSpecs.emplace_back(o2::calibration::Utils::gDataOriginCDBPayload, "MID_BAD_CHANNELS", 0, of::Lifetime::Sporadic);
  outputSpecs.emplace_back(o2::calibration::Utils::gDataOriginCDBWrapper, "MID_BAD_CHANNELS", 0, of::Lifetime::Sporadic);
  outputSpecs.emplace_back(o2::calibration::Utils::gDataOriginCDBPayload, "MID_BAD_CHANNELS", 1, of::Lifetime::Sporadic);
  outputSpecs.emplace_back(o2::calibration::Utils::gDataOriginCDBWrapper, "MID_BAD_CHANNELS", 1, of::Lifetime::Sporadic);

  return of::DataProcessorSpec{
    "MIDChannelCalibrator",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::ChannelCalibratorDeviceDPL>(feeIdConfig, crateMasks)},
    of::Options{
      {"mid-mask-threshold", of::VariantType::Double, 0.9, {"Tolerated occupancy before producing a map"}},
      {"tf-per-slot", of::VariantType::Int64, INFINITE_TF, {"number of TFs per calibration time slot"}},
      {"max-delay", of::VariantType::Int64, 0ll, {"number of slots in past to consider"}}}};
}
} // namespace mid
} // namespace o2