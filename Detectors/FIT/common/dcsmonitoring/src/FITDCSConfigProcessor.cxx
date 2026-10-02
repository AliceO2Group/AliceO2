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

#include "FITDCSMonitoring/FITDCSConfigProcessor.h"

namespace o2::fit
{
void FITDCSConfigProcessor::initDeadChannelMapReader()
{
  mDeadChannelMapReader = std::make_unique<FITDeadChannelMapReader>(FITDeadChannelMapReader());
}

long FITDCSConfigProcessor::getValidityTime(o2::framework::ProcessingContext& pc)
{
  long dataTime = (long)(pc.services().get<o2::framework::TimingInfo>().creation);
  if (dataTime == 0xffffffffffffffff) {                                                                                                     // means it is not set
    dataTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); // in ms
  }
  return dataTime;
}

void FITDCSConfigProcessor::setupDeadChannelMapReader(o2::framework::InitContext& ic)
{
  mDeadChannelMapReader->setFileNameDChM(ic.options().get<std::string>("filename-dchm"));
  mDeadChannelMapReader->setValidDaysDChM(ic.options().get<uint>("valid-days-dchm"));
  mDeadChannelMapReader->setCcdbPathDChM(mDetectorName + "/Calib/DeadChannelMap");
  bool verbose = ic.options().get<bool>("use-verbose-mode");
  mDeadChannelMapReader->setVerboseMode(verbose);
  bool validateUpload = !ic.options().get<bool>("no-validate");
  mDeadChannelMapReader->setValidateUploadMode(validateUpload);

  LOG(info) << "Verbose mode: " << verbose;
  LOG(info) << "Validate upload: " << validateUpload;
  LOG(info) << "Expected dead channel map file name: " << mDeadChannelMapReader->getFileNameDChM();
  LOG(info) << "Dead channel maps will be valid for " << mDeadChannelMapReader->getValidDaysDChM() << " days";
}

void FITDCSConfigProcessor::handleDeadChannelMapUpdate(o2::framework::ProcessingContext& pc, long dataTime, gsl::span<const char> dataBuffer)
{
  processDeadChannelMap(dataTime, dataBuffer);
  sendObject(pc.outputs(), mDeadChannelMapReader->getDChM(), mDeadChannelMapReader->getObjectInfoDChM(), mDataDescriptionDChM);
  mDeadChannelMapReader->resetStartValidityDChM();
  mDeadChannelMapReader->resetDChM();
}

/// Processing the dead channel map
void FITDCSConfigProcessor::processDeadChannelMap(const long& dataTime, gsl::span<const char> dataBuffer)
{
  if (!mDeadChannelMapReader->isStartValidityDChMSet()) {
    mDeadChannelMapReader->setStartValidityDChM(dataTime);
  }
  mDeadChannelMapReader->processDChM(dataBuffer);
  mDeadChannelMapReader->updateDChMCcdbObjectInfo();
}

void FITDCSConfigProcessor::setupHvConfigurationReader(o2::framework::InitContext& ic, FITHvConfigurationReader& hvReader)
{
  hvReader.setFilename(ic.options().get<std::string>("filename-hv-config"));
  mHvConfigCcdbInfo.setCcdbPath(mDetectorName + "/Config/HvConfiguration");
  mHvConfigCcdbInfo.setValidityPeriodInDays(ic.options().get<uint32_t>("valid-days-hv-config"));
}
} // namespace o2::fit