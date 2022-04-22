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

#include "TRDCalibration/DCSProcessor.h"
#include "DetectorsCalibration/Utils.h"

using namespace o2::trd;
using namespace o2::dcs;

using DeliveryType = o2::dcs::DeliveryType;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;

void TRDDCSMinMaxMeanInfo::print() const
{
  LOG(info) << "Min value: " << minValue;
  LOG(info) << "Max value: " << maxValue;
  LOG(info) << "Mean value: " << meanValue;
  LOG(info) << "Number of points added: " << nPoints;
}

void TRDDCSMinMaxMeanInfo::addPoint(float value)
{
  if (nPoints == 0) {
    minValue = value;
    maxValue = value;
    meanValue = value;
  } else {
    if (value < minValue) {
      minValue = value;
    }
    if (value > maxValue) {
      maxValue = value;
    }
    meanValue += (value - meanValue) / (nPoints + 1);
  }
  ++nPoints;
}

//__________________________________________________________________

void DCSProcessor::init(const std::vector<DPID>& pids)
{
  // fill the array of the DPIDs that will be used by TRD
  // pids should be provided by CCDB

  for (const auto& it : pids) {
    mPids[it] = false;
  }
}

//__________________________________________________________________

int DCSProcessor::process(const gsl::span<const DPCOM> dps)
{

  // first we check which DPs are missing - if some are, it means that
  // the delta map was sent
  if (mVerbose) {
    LOG(info) << "\n\n\nProcessing new TF\n-----------------";
  }
  if (!mStartTFset) {
    mStartTF = mStartValidity;
    mStartTFset = true;
  }

  std::unordered_map<DPID, DPVAL> mapin;
  for (auto& it : dps) {
    mapin[it.id] = it.data;
  }
  for (auto& it : mPids) {
    const auto& el = mapin.find(it.first);
    if (mVerbose) {
      if (el == mapin.end()) {
        LOG(info) << "DP " << it.first << " not found in map";
      } else {
        LOG(info) << "DP " << it.first << " found in map";
      }
    }
  }

  // now we process all DPs, one by one
  for (const auto& it : dps) {
    // we process only the DPs defined in the configuration
    const auto& el = mPids.find(it.id);
    if (el == mPids.end()) {
      if (mVerbose) {
        LOG(info) << "DP " << it.id << " not found in DCSProcessor, we will not process it";
      }
      continue;
    }
    processDP(it);
    mPids[it.id] = true;
  }

  return 0;
}

//__________________________________________________________________

int DCSProcessor::processDP(const DPCOM& dpcom)
{

  // processing single DP

  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  if (mVerbose) {
    if (type == DPVAL_DOUBLE) {
      LOG(info) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<double>(dpcom);
    } else if (type == DPVAL_INT) {
      LOG(info) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<int32_t>(dpcom);
    }
  }
  auto flags = dpcom.data.get_flags();
  if (processFlags(flags, dpid.get_alias()) == 0) {
    if (type == DPVAL_DOUBLE) {
      auto& dvect = mDpsDoublesmap[dpid];
      if (mVerbose) {
        LOG(info) << "mDpsDoublesmap[dpid].size() = " << dvect.size();
      }
      auto etime = dpcom.data.get_epoch_time();
      if (dvect.size() == 0 || etime != dvect.back().data.get_epoch_time()) {
        // only add data point in case it was not already read before
        dvect.push_back(dpcom);
      }
    }
    if (type == DPVAL_INT) {
      // TODO so far there is no processing at all for these type of DCS data points
      if (std::strstr(dpid.get_alias(), "trd_runNo") != nullptr) { // DP is trd_runNo
        std::string aliasStr(dpid.get_alias());
        auto runNumber = o2::dcs::getValue<int32_t>(dpcom);
        if (mVerbose) {
          LOG(info) << "Run number = " << runNumber;
        }
        // end processing current DP, when it is of type trd_runNo
      } else if (std::strstr(dpid.get_alias(), "trd_runType") != nullptr) { // DP is trd_runType
        std::string aliasStr(dpid.get_alias());
        auto runType = o2::dcs::getValue<int32_t>(dpcom);
        if (mVerbose) {
          LOG(info) << "Run type = " << runType;
        }
      }
    }
  }
  return 0;
}

//______________________________________________________________________

int DCSProcessor::processFlags(const uint64_t flags, const char* pid)
{

  // function to process the flag. the return code zero means that all is fine.
  // anything else means that there was an issue

  // for now, I don't know how to use the flags, so I do nothing

  if (mVerbose) {
    if (flags & DataPointValue::KEEP_ALIVE_FLAG) {
      LOG(info) << "KEEP_ALIVE_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::END_FLAG) {
      LOG(info) << "END_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::FBI_FLAG) {
      LOG(info) << "FBI_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::NEW_FLAG) {
      LOG(info) << "NEW_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::DIRTY_FLAG) {
      LOG(info) << "DIRTY_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::TURN_FLAG) {
      LOG(info) << "TURN_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::WRITE_FLAG) {
      LOG(info) << "WRITE_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::READ_FLAG) {
      LOG(info) << "READ_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::OVERWRITE_FLAG) {
      LOG(info) << "OVERWRITE_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::VICTIM_FLAG) {
      LOG(info) << "VICTIM_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::DIM_ERROR_FLAG) {
      LOG(info) << "DIM_ERROR_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::BAD_DPID_FLAG) {
      LOG(info) << "BAD_DPID_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::BAD_FLAGS_FLAG) {
      LOG(info) << "BAD_FLAGS_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::BAD_TIMESTAMP_FLAG) {
      LOG(info) << "BAD_TIMESTAMP_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::BAD_PAYLOAD_FLAG) {
      LOG(info) << "BAD_PAYLOAD_FLAG active for DP " << pid;
    }
    if (flags & DataPointValue::BAD_FBI_FLAG) {
      LOG(info) << "BAD_FBI_FLAG active for DP " << pid;
    }
  }

  return 0;
}

//______________________________________________________________________

void DCSProcessor::updateDPsCCDB()
{
  // here we create the object to then be sent to CCDB
  LOG(info) << "Finalizing";

  for (const auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      auto& trddcs = mTRDDCS[it.first];
      if (it.second == true) { // we processed the DP at least 1x
        auto& dpVect = mDpsDoublesmap[it.first];
        for (const auto& dpCom : dpVect) {
          trddcs.addPoint(o2::dcs::getValue<double>(dpCom));
        }
      }
      if (mVerbose) {
        LOG(info) << "PID = " << it.first.get_alias();
        trddcs.print();
      }
    }
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Ole Schmidt";
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCS, mCcdbDPsInfo, "TRD/Calib/DCSDPs", md, mStartValidity, o2::calibration::Utils::INFINITE_TIME);

  return;
}
