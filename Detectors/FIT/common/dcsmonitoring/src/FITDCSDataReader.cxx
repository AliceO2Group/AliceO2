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

/// \file FITDCSDataReader.cxx
/// \brief DCS data point reader for FIT
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "FITDCSMonitoring/FITDCSDataReader.h"

#include "DetectorsCalibration/Utils.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"

#include <cstdint>
#include <gsl/gsl>
#include <string>
#include <unordered_map>
#include <vector>

using namespace o2::fit;
using namespace o2::dcs;

using DPID = o2::dcs::DataPointIdentifier;
using DPCOM = o2::dcs::DataPointCompositeObject;

void FITDCSDataReader::init(const std::vector<DPID>& pids)
{
  // Fill the array of sub-detector specific DPIDs that will be processed
  for (const auto& it : pids) {
    mPids[it] = false;
    mDpData[it].makeEmpty();
  }
}

int FITDCSDataReader::process(const gsl::span<const DPCOM> dps)
{
  // first we check which DPs are missing - if some are, it means that
  // the delta map was sent

  if (getVerboseMode()) {
    LOG(info) << "\n\nProcessing new DCS DP map\n-------------------------";
  }

  if (false) {
    std::unordered_map<DPID, DPVAL> mapin;
    for (auto& it : dps) {
      mapin[it.id] = it.data;
    }
    for (auto& it : mPids) {
      const auto& el = mapin.find(it.first);
      if (el == mapin.end()) {
        LOG(debug) << "DP " << it.first << " not found in DPs from DCS";
      } else {
        LOG(debug) << "DP " << it.first << " found in DPs from DCS";
      }
    }
  }

  // now we process all DPs, one by one
  for (const auto& it : dps) {
    // we process only the DPs defined in the configuration
    const auto& el = mPids.find(it.id);
    if (el == mPids.end()) {
      LOG(info) << "DP " << it.id << " not found in FITDCSProcessor, we will not process it";
      continue;
    }
    processDP(it);
    mPids[it.id] = true;
  }

  return 0;
}

int FITDCSDataReader::processDP(const DPCOM& dpcom)
{
  // Processing a single DP
  const auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  const auto& val = dpcom.data;

  if (getVerboseMode()) {
    if (type == DPVAL_DOUBLE) {
      LOG(info) << "Processing DP = " << dpcom << " (epoch " << val.get_epoch_time() << "), with value = " << o2::dcs::getValue<double>(dpcom);
    } else if (type == DPVAL_UINT) {
      LOG(info) << "Processing DP = " << dpcom << " (epoch " << val.get_epoch_time() << "), with value = " << o2::dcs::getValue<uint>(dpcom);
    }
  }

  auto flags = val.get_flags();
  if (processFlags(flags, dpid.get_alias()) == 0) {
    // Store all DP values
    if (mDpData[dpid].values.empty() || val.get_epoch_time() > mDpData[dpid].values.back().first) {
      dpValueConverter.raw_data = val.payload_pt1;
      if (type == DPVAL_DOUBLE) {
        mDpData[dpid].add(val.get_epoch_time(), llround(dpValueConverter.double_value * 1000)); // store as nA
      } else if (type == DPVAL_UINT) {
        mDpData[dpid].add(val.get_epoch_time(), dpValueConverter.uint_value);
      }
    }
  }

  return 0;
}

uint64_t FITDCSDataReader::processFlags(const uint64_t flags, const char* pid)
{
  // function to process the flag. the return code zero means that all is fine.
  // anything else means that there was an issue

  // for now, I don't know how to use the flags, so I do nothing

  if (flags & DataPointValue::KEEP_ALIVE_FLAG) {
    LOG(debug) << "KEEP_ALIVE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::END_FLAG) {
    LOG(debug) << "FBI_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::NEW_FLAG) {
    LOG(debug) << "NEW_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::DIRTY_FLAG) {
    LOG(debug) << "DIRTY_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::TURN_FLAG) {
    LOG(debug) << "TURN_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::WRITE_FLAG) {
    LOG(debug) << "WRITE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::READ_FLAG) {
    LOG(debug) << "READ_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::OVERWRITE_FLAG) {
    LOG(debug) << "OVERWRITE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::VICTIM_FLAG) {
    LOG(debug) << "VICTIM_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::DIM_ERROR_FLAG) {
    LOG(debug) << "DIM_ERROR_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_DPID_FLAG) {
    LOG(debug) << "BAD_DPID_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_FLAGS_FLAG) {
    LOG(debug) << "BAD_FLAGS_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_TIMESTAMP_FLAG) {
    LOG(debug) << "BAD_TIMESTAMP_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_PAYLOAD_FLAG) {
    LOG(debug) << "BAD_PAYLOAD_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_FBI_FLAG) {
    LOG(debug) << "BAD_FBI_FLAG active for DP " << pid;
  }

  return 0;
}

void FITDCSDataReader::updateCcdbObjectInfo()
{
  // Prepare the object to be sent to CCDB
  if (getVerboseMode()) {
    for (auto& dp : mDpData) {
      // if (dp.second.values.empty()) {
      //   continue;
      // }
      LOG(info) << "PID = " << dp.first.get_alias();
      dp.second.print();
    }
  }

  std::map<std::string, std::string> metadata;
  o2::calibration::Utils::prepareCCDBobjectInfo(getDpData(), getccdbDPsInfo(), getCcdbPath(), metadata, getStartValidity(), getEndValidity());

  return;
}

const std::unordered_map<DPID, DCSDPValues>& FITDCSDataReader::getDpData() const { return mDpData; }

void FITDCSDataReader::resetDpData()
{
  mDpsMap.clear();
  mDpData.clear();
}

const std::string& FITDCSDataReader::getCcdbPath() const { return mCcdbPath; }
void FITDCSDataReader::setCcdbPath(const std::string& ccdbPath) { mCcdbPath = ccdbPath; }
long FITDCSDataReader::getStartValidity() const { return mStartValidity; }
void FITDCSDataReader::setStartValidity(const long startValidity) { mStartValidity = startValidity; }
bool FITDCSDataReader::isStartValiditySet() const { return mStartValidity != o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; }
void FITDCSDataReader::resetStartValidity() { mStartValidity = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; }
long FITDCSDataReader::getEndValidity() const { return mStartValidity + 3 * o2::ccdb::CcdbObjectInfo::DAY; }
const o2::ccdb::CcdbObjectInfo& FITDCSDataReader::getccdbDPsInfo() const { return mCcdbDpInfo; }
o2::ccdb::CcdbObjectInfo& FITDCSDataReader::getccdbDPsInfo() { return mCcdbDpInfo; }

bool FITDCSDataReader::getVerboseMode() const { return mVerbose; }
void FITDCSDataReader::setVerboseMode(bool verboseMode) { mVerbose = verboseMode; }
