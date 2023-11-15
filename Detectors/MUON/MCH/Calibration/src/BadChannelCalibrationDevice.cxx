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

#include "BadChannelCalibrationDevice.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsMCH/DsChannelId.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataAllocator.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Output.h"
#include "MCHCalibration/BadChannelCalibratorParam.h"
#include "TObjString.h"
#include <TBuffer3D.h>
#include <cmath>
#include <limits>
#include <numeric>

namespace o2::mch::calibration
{

void BadChannelCalibrationDevice::init(o2::framework::InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
  mLoggingInterval = ic.options().get<int>("logging-interval") * 1000;

  mCalibrator = std::make_unique<o2::mch::calibration::BadChannelCalibrator>();
  mCalibrator->setSlotLength(o2::calibration::INFINITE_TF);
  mCalibrator->setUpdateAtTheEndOfRunOnly();
  mTimeStamp = std::numeric_limits<uint64_t>::max();
}

//_________________________________________________________________
void BadChannelCalibrationDevice::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
}

void BadChannelCalibrationDevice::logStats(size_t dataSize)
{
  static auto loggerStart = std::chrono::high_resolution_clock::now();
  static auto loggerEnd = loggerStart;
  static size_t nDigits = 0;
  static size_t nTF = 0;

  if (mLoggingInterval == 0) {
    return;
  }

  nDigits += dataSize;
  nTF += 1;

  loggerEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> loggerElapsed = loggerEnd - loggerStart;
  if (loggerElapsed.count() > 1000) {
    LOG(info) << "received " << nDigits << " digits in " << nTF << " time frames";
    nDigits = 0;
    nTF = 0;
    loggerStart = std::chrono::high_resolution_clock::now();
  }
}

void BadChannelCalibrationDevice::run(o2::framework::ProcessingContext& pc)
{
  if (mSkipData) {
    return;
  }

  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
  mTimeStamp = std::min(mTimeStamp, mCalibrator->getCurrentTFInfo().creation);
  auto data = pc.inputs().get<gsl::span<o2::mch::calibration::PedestalDigit>>("digits");
  mCalibrator->process(data);

  if (!BadChannelCalibratorParam::Instance().onlyAtEndOfStream) {
    std::string reason;
    if (mCalibrator->readyToSend(reason)) {
      mHasEnoughStat = true;
      LOGP(info, "We're ready to send output to CCDB ({})", reason);
      sendOutput(pc.outputs(), reason);
      mSkipData = true;
    }
  }
  logStats(data.size());
}

template <typename T>
ccdb::CcdbObjectInfo createCcdbInfo(const T& object, uint64_t timeStamp, std::string_view reason)
{
  auto clName = o2::utils::MemFileHelper::getClassName(object);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  std::map<std::string, std::string> md;
  md["upload-reason"] = reason;
  constexpr auto fiveDays = 5 * o2::ccdb::CcdbObjectInfo::DAY;
  return o2::ccdb::CcdbObjectInfo("MCH/Calib/BadChannel", clName, flName, md, timeStamp, timeStamp + fiveDays);
}

void BadChannelCalibrationDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  if (BadChannelCalibratorParam::Instance().onlyAtEndOfStream) {
    std::string reason;
    if (mCalibrator->readyToSend(reason)) {
      mHasEnoughStat = true;
    }
    sendOutput(ec.outputs(), "End Of Stream");
  }
}

std::string toCSV(const std::vector<o2::mch::DsChannelId>& channels)
{
  std::stringstream csv;
  csv << fmt::format("solarid,dsid,ch\n");

  for (const auto& c : channels) {
    csv << fmt::format("{},{},{}\n",
                       c.getSolarId(), c.getElinkId(), c.getChannel());
  }

  return csv.str();
}

template <typename T>
void sendCalibrationOutput(o2::framework::DataAllocator& output,
                           header::DataHeader::SubSpecificationType subSpec,
                           T* payload,
                           o2::ccdb::CcdbObjectInfo* payloadInfo)
{

  using clbUtils = o2::calibration::Utils;
  auto image = o2::ccdb::CcdbApi::createObjectImage(payload, payloadInfo);

  LOG(info) << "Sending object " << payloadInfo->getPath()
            << " of type" << payloadInfo->getObjectType()
            << " /" << payloadInfo->getFileName()
            << " of size " << image->size()
            << " bytes, valid for " << payloadInfo->getStartValidityTimestamp()
            << " : " << payloadInfo->getEndValidityTimestamp();

  output.snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBPayload, "MCH_BADCHAN", subSpec}, *image.get());
  output.snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBWrapper, "MCH_BADCHAN", subSpec}, *payloadInfo);
}

void BadChannelCalibrationDevice::sendOutput(o2::framework::DataAllocator& output,
                                             std::string_view reason)
{
  auto& slot = mCalibrator->getFirstSlot();
  const auto pedData = slot.getContainer();
  uint64_t nentries = std::accumulate(pedData->cbegin(), pedData->cend(),
                                      0,
                                      [&](uint64_t n, const PedestalChannel& c) { return n +
                                                                                         static_cast<uint64_t>(c.mEntries); });
  std::string reason_with_entries;

  if (pedData->size() > 0) {
    int mean = static_cast<int>(std::round(static_cast<float>(nentries) / pedData->size()));
    reason_with_entries = fmt::format("{} ; <entries per channel>={}", reason, mean);
  } else {
    reason_with_entries = fmt::format("{} ; no entries", reason);
  }

  LOGP(info, "sendOutput: {}", reason_with_entries);
  mCalibrator->finalize();

  // the bad channels table is only updated if there is enough statistics
  if (mHasEnoughStat) {
    // send regular bad channel object to subspec 0. This regular object
    // is meant for O2 consumption (in reconstruction and/or simulation)
    const auto& badChannels = mCalibrator->getBadChannelsVector();
    auto info = createCcdbInfo(badChannels, mTimeStamp, reason_with_entries);

    sendCalibrationOutput(output, 0, &badChannels, &info);

    // send also a simplified (in comma-separated values format) version
    // of the bad channel list to subspec 1.
    // This simplified version is meant for DCS usage
    // (to populate the Oracle online DB for electronics configuration)

    TObjString badChannelsPOD(toCSV(badChannels).c_str());
    auto infoPOD = createCcdbInfo(badChannelsPOD, mTimeStamp, reason_with_entries);

    sendCalibrationOutput(output, 1, &badChannelsPOD, &infoPOD);

    LOGP(info, "csv={}", badChannelsPOD.String().Data());
  } else {
    LOGP(error, "CCDB not updated: {}", reason_with_entries);
  }

  // and finally send also the data used to compute the bad channel map
  // on a separate channel (for QC mainly)
  // this is sent also if statistics is too low, for diagnostics purposes
  output.snapshot(o2::framework::Output{"MCH", "PEDESTALS", 0},
                  mCalibrator->getPedestalsVector());
  if (mHasEnoughStat) {
    output.snapshot(o2::framework::Output{"MCH", "BADCHAN", 0},
                    mCalibrator->getBadChannelsVector());
  } else {
    output.cookDeadBeef(o2::framework::Output{"MCH", "BADCHAN"});
  }
}
} // namespace o2::mch::calibration
