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

#include "FT0DCSMonitoring/FT0DCSConfigProcessor.h"
#include "DataFormatsFT0/HvConfiguration.h"

namespace o2::ft0
{
void FT0DCSConfigProcessor::init(o2::framework::InitContext& ic)
{
  initDeadChannelMapReader();
  setupDeadChannelMapReader(ic);
  setupFeeConfigurationReader(ic, mFeeConfigurationReader);
  setupHvConfigurationReader(ic, mHvConfigurationReader);
}

void FT0DCSConfigProcessor::run(o2::framework::ProcessingContext& pc)
{
  try {
    long dataTime = getValidityTime(pc);

    gsl::span<const char> dataBuffer = pc.inputs().get<gsl::span<char>>("inputConfig");
    std::string configFileName = pc.inputs().get<std::string>("inputConfigFileName");
    LOG(info) << "Got input file " << configFileName << " of size " << dataBuffer.size();
    if (!configFileName.compare(mDeadChannelMapReader->getFileNameDChM())) {
      handleDeadChannelMapUpdate(pc, dataTime, dataBuffer);
    } else if (mFeeConfigurationReader.matchFilename(configFileName)) {
      Ft0FeeConfiguration feeConfiguration = mFeeConfigurationReader.parseFeeConfiguration(dataBuffer);
      o2::ccdb::CcdbObjectInfo objectInfo = mFeeConfigCcdbInfo.createObjectInfo(feeConfiguration, dataTime, {});
      sendObject(pc.outputs(), feeConfiguration, objectInfo, getFeeConfigDescription());
    } else if (mHvConfigurationReader.matchFilename(configFileName)) {
      Ft0HvConfiguration hvConfig = mHvConfigurationReader.parseHvConfiguration<Ft0HvConfiguration>(dataBuffer);
      o2::ccdb::CcdbObjectInfo objectInfo = mHvConfigCcdbInfo.createObjectInfo(hvConfig, dataTime, {});
      sendObject(pc.outputs(), hvConfig, objectInfo, getHvConfigDescription());
    } else {
      LOG(error) << "Unknown input file: " << configFileName;
    }
  } catch (std::exception& e) {
    LOG(error) << "Exception: " << e.what();
  }
}

void FT0DCSConfigProcessor::endOfStream(o2::framework::EndOfStreamContext& ec)
{
}
} // namespace o2::ft0