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

#include <FT0Calibration/FT0DCSProcessor.h>
#include "DetectorsCalibration/Utils.h"
#include "Rtypes.h"
#include <deque>
#include <string>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <bitset>

using namespace o2::ft0;
using namespace o2::dcs;

using DeliveryType = o2::dcs::DeliveryType;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;

void FT0DCSProcessor::init(const std::vector<DPID>& pids)
{
  // fill the array of the DPIDs that will be used by FT0
  // pids should be provided by CCDB

  for (const auto& it : pids) {
    mPids[it] = false;
    mFT0DCS[it].makeEmpty();
  }
}

int FT0DCSProcessor::process(const gsl::span<const DPCOM> dps)
{
  // first we check which DPs are missing - if some are, it means that
  // the delta map was sent

  if (mVerbose) {
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
        LOG(debug) << "DP " << it.first << " not found in map";
      } else {
        LOG(debug) << "DP " << it.first << " found in map";
      }
    }
  }

  // now we process all DPs, one by one
  for (const auto& it : dps) {
    // we process only the DPs defined in the configuration
    const auto& el = mPids.find(it.id);
    if (el == mPids.end()) {
      LOG(info) << "DP " << it.id << " not found in FT0DCSProcessor, we will not process it";
      continue;
    }
    processDP(it);
    mPids[it.id] = true;
  }

  return 0;
}

int FT0DCSProcessor::processDP(const DPCOM& dpcom)
{
  // processing a single DP

  const auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  const auto& val = dpcom.data;
  if (mVerbose) {
    if (type == DPVAL_DOUBLE) {
      LOG(info) << "Processing DP = " << dpcom << " (epoch " << val.get_epoch_time() << "), with value = " << o2::dcs::getValue<double>(dpcom);
    } else if (type == DPVAL_UINT) {
      LOG(info) << "Processing DP = " << dpcom << " (epoch " << val.get_epoch_time() << "), with value = " << o2::dcs::getValue<uint>(dpcom);
    }
  }
  auto flags = val.get_flags();
  if (processFlags(flags, dpid.get_alias()) == 0) {
    // Store all DP values
    if (mFT0DCS[dpid].values.empty() || val.get_epoch_time() > mFT0DCS[dpid].values.back().first) {
      converter.raw_data = val.payload_pt1;
      if (type == DPVAL_DOUBLE) {
        mFT0DCS[dpid].add(val.get_epoch_time(), lround(converter.double_value * 1000)); // store as nA
      } else if (type == DPVAL_UINT) {
        mFT0DCS[dpid].add(val.get_epoch_time(), converter.uint_value);
      }
    }
  }
  return 0;
}

uint64_t FT0DCSProcessor::processFlags(const uint64_t flags, const char* pid)
{
  // function to process the flag. the return code zero means that all is fine.
  // anything else means that there was an issue

  // for now, I don't know how to use the flags, so I do nothing

  if (flags & DataPointValue::KEEP_ALIVE_FLAG) {
    LOG(debug) << "KEEP_ALIVE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::END_FLAG) {
    LOG(debug) << "END_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::FBI_FLAG) {
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

void FT0DCSProcessor::updateDPsCCDB()
{
  // Prepare the object to be sent to CCDB
  if (mVerbose) {
    for (auto& dp : mFT0DCS) {
      // if (dp.second.values.empty()) {
      //   continue;
      // }
      LOG(info) << "PID = " << dp.first.get_alias();
      dp.second.print();
    }
  }

  std::map<std::string, std::string> md;
  o2::calibration::Utils::prepareCCDBobjectInfo(mFT0DCS, mccdbDPsInfo, "FT0/Calib/DCSDPs", md, mStartValidity, mStartValidity + 3 * o2::ccdb::CcdbObjectInfo::DAY);

  return;
}