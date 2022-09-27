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

/// @file  FITDCSDataProcessor.cxx
/// @brief Task for processing FIT DCS data
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "FITDCSMonitoring/FITDCSDataProcessor.h"

#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "FITDCSMonitoring/FITDCSDataReader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataAllocator.h"
#include "Framework/Output.h"

#include <cstdint>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace o2::fit;

using DPCOM = o2::dcs::DataPointCompositeObject;
using DPID = o2::dcs::DataPointIdentifier;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

void FITDCSDataProcessor::init(o2::framework::InitContext& ic)
{
  setVerboseMode(ic.options().get<bool>("use-verbose-mode"));
  LOG(info) << "Verbose mode: " << getVerboseMode();

  mDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval");
  if (mDPsUpdateInterval == 0) {
    LOG(error) << mDetectorName << " DPs update interval set to zero seconds --> changed to 10 min.";
    mDPsUpdateInterval = 600;
  }

  std::vector<DPID> vect;

  const bool useCcdbToConfigure = ic.options().get<bool>("use-ccdb-to-configure");
  if (useCcdbToConfigure) {
    LOG(info) << "Configuring via CCDB";
    const std::string ccdbPath = ic.options().get<std::string>("ccdb-path");
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setURL(ccdbPath);
    long timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::unordered_map<DPID, std::string>* dpid2DataDesc = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>(mDetectorName + "/Config/DCSDPconfig", timestamp);
    for (auto& i : *dpid2DataDesc) {
      vect.push_back(i.first);
    }
  } else {
    LOG(info) << "Configuring via hardcoded strings";
    vect = getHardCodedDPIDs();
  }

  if (getVerboseMode()) {
    LOGP(info, "Listing Data Points for {}:", mDetectorName);
    for (auto& i : vect) {
      LOG(info) << i;
    }
  }

  mDataReader = std::make_unique<o2::fit::FITDCSDataReader>();
  mDataReader->setVerboseMode(getVerboseMode());
  mDataReader->setCcdbPath(mDetectorName + "/Calib/DCSDPs");
  mDataReader->init(vect);
  mTimer = HighResClock::now();
}

void FITDCSDataProcessor::run(o2::framework::ProcessingContext& pc)
{
  auto timeNow = HighResClock::now();
  long dataTime = (long)(pc.services().get<o2::framework::TimingInfo>().creation);

  if (dataTime == 0xffffffffffffffff) {                                                                   // it means it is not set
    dataTime = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
  }

  if (!mDataReader->isStartValiditySet()) {
    if (getVerboseMode()) {
      LOG(info) << "Start valitidy for DPs changed to = " << dataTime;
    }
    mDataReader->setStartValidity(dataTime);
  }
  auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
  mDataReader->process(dps);
  Duration elapsedTime = timeNow - mTimer; // in seconds
  if (elapsedTime.count() >= mDPsUpdateInterval) {
    sendDPsOutput(pc.outputs());
    mTimer = timeNow;
  }
}

void FITDCSDataProcessor::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  sendDPsOutput(ec.outputs());
}

const std::string& FITDCSDataProcessor::getDetectorName() const { return mDetectorName; }
bool FITDCSDataProcessor::getVerboseMode() const { return mVerbose; }
void FITDCSDataProcessor::setVerboseMode(bool verboseMode) { mVerbose = verboseMode; }

void FITDCSDataProcessor::sendDPsOutput(o2::framework::DataAllocator& output)
{
  // extract CCDB infos and calibration object for DPs
  mDataReader->updateCcdbObjectInfo();
  const auto& payload = mDataReader->getDpData();
  auto& info = mDataReader->getccdbDPsInfo();
  auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  output.snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBPayload, mDataDescription, 0}, *image.get());
  output.snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBWrapper, mDataDescription, 0}, info);
  mDataReader->resetDpData();
  mDataReader->resetStartValidity();
}
