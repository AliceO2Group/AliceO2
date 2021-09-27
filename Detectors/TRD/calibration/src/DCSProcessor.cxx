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

using namespace o2::trd;
using namespace o2::dcs;

using DeliveryType = o2::dcs::DeliveryType;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;

void TRDDCSinfo::print() const
{
  LOG(INFO) << "First Value: timestamp = " << firstValue.first << ", value = " << firstValue.second;
  LOG(INFO) << "Last Value:  timestamp = " << lastValue.first << ", value = " << lastValue.second;
  LOG(INFO) << "Mid Value:   timestamp = " << midValue.first << ", value = " << midValue.second;
  LOG(INFO) << "Max Change:  timestamp = " << maxChange.first << ", value = " << maxChange.second;
}

//__________________________________________________________________

void DCSProcessor::init(const std::vector<DPID>& pids)
{
  // fill the array of the DPIDs that will be used by TRD
  // pids should be provided by CCDB

  for (const auto& it : pids) {
    mPids[it] = false;
    mTRDDCS[it].makeEmpty();
  }
}

//__________________________________________________________________

int DCSProcessor::process(const gsl::span<const DPCOM> dps)
{

  // first we check which DPs are missing - if some are, it means that
  // the delta map was sent
  if (mVerbose) {
    LOG(INFO) << "\n\n\nProcessing new TF\n-----------------";
  }
  if (!mStartTFset) {
    mStartTF = mTF;
    mStartTFset = true;
  }

  std::unordered_map<DPID, DPVAL> mapin;
  for (auto& it : dps) {
    mapin[it.id] = it.data;
  }
  for (auto& it : mPids) {
    const auto& el = mapin.find(it.first);
    if (el == mapin.end()) {
      LOG(INFO) << "DP " << it.first << " not found in map";
    } else {
      LOG(INFO) << "DP " << it.first << " found in map";
    }
  }

  // now we process all DPs, one by one
  for (const auto& it : dps) {
    // we process only the DPs defined in the configuration
    const auto& el = mPids.find(it.id);
    if (el == mPids.end()) {
      LOG(INFO) << "DP " << it.id << " not found in DCSProcessor, we will not process it";
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
  auto& val = dpcom.data;
  if (mVerbose) {
    if (type == RAW_DOUBLE) {
      LOG(INFO) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<double>(dpcom);
    } else if (type == RAW_INT) {
      LOG(INFO) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<int32_t>(dpcom);
    }
  }
  auto flags = val.get_flags();
  if (processFlags(flags, dpid.get_alias()) == 0) {
    if (type == RAW_DOUBLE) {
      auto& dvect = mDpsdoublesmap[dpid];
      LOG(INFO) << "mDpsdoublesmap[dpid].size() = " << dvect.size();
      auto etime = val.get_epoch_time();
      if (dvect.size() == 0 ||
          etime != dvect.back().get_epoch_time()) { // we check
                                                    // that we did not get the
                                                    // same timestamp as the
                                                    // latest one
        dvect.push_back(val);
      }
    }
    if (type == RAW_INT) {
      if (std::strstr(dpid.get_alias(), "trd_runNo") != nullptr) { // DP is trd_runNo
        std::string aliasStr(dpid.get_alias());
        int32_t runNumber = o2::dcs::getValue<int32_t>(dpcom);
        if (mVerbose) {
          LOG(INFO) << "Run number = " << runNumber;
        }
      } //end processing current DP, when it is of type trd_runNo
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

  if (flags & DataPointValue::KEEP_ALIVE_FLAG) {
    LOG(INFO) << "KEEP_ALIVE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::END_FLAG) {
    LOG(INFO) << "END_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::FBI_FLAG) {
    LOG(INFO) << "FBI_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::NEW_FLAG) {
    LOG(INFO) << "NEW_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::DIRTY_FLAG) {
    LOG(INFO) << "DIRTY_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::TURN_FLAG) {
    LOG(INFO) << "TURN_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::WRITE_FLAG) {
    LOG(INFO) << "WRITE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::READ_FLAG) {
    LOG(INFO) << "READ_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::OVERWRITE_FLAG) {
    LOG(INFO) << "OVERWRITE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::VICTIM_FLAG) {
    LOG(INFO) << "VICTIM_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::DIM_ERROR_FLAG) {
    LOG(INFO) << "DIM_ERROR_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_DPID_FLAG) {
    LOG(INFO) << "BAD_DPID_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_FLAGS_FLAG) {
    LOG(INFO) << "BAD_FLAGS_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_TIMESTAMP_FLAG) {
    LOG(INFO) << "BAD_TIMESTAMP_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_PAYLOAD_FLAG) {
    LOG(INFO) << "BAD_PAYLOAD_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_FBI_FLAG) {
    LOG(INFO) << "BAD_FBI_FLAG active for DP " << pid;
  }

  return 0;
}

//______________________________________________________________________

void DCSProcessor::updateDPsCCDB()
{

  // here we create the object to then be sent to CCDB
  LOG(INFO) << "Finalizing";
  union Converter {
    uint64_t raw_data;
    double double_value;
  } converter0, converter1;

  for (const auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::RAW_DOUBLE) {
      auto& trddcs = mTRDDCS[it.first];
      if (it.second == true) { // we processed the DP at least 1x
        auto& dpvect = mDpsdoublesmap[it.first];
        trddcs.firstValue.first = dpvect[0].get_epoch_time();
        converter0.raw_data = dpvect[0].payload_pt1;
        trddcs.firstValue.second = converter0.double_value;
        trddcs.lastValue.first = dpvect.back().get_epoch_time();
        converter0.raw_data = dpvect.back().payload_pt1;
        trddcs.lastValue.second = converter0.double_value;
        // now I will look for the max change
        if (dpvect.size() > 1) {
          auto deltatime = dpvect.back().get_epoch_time() - dpvect[0].get_epoch_time();
          if (deltatime < 60000) {
            // if we did not cover at least 1 minute,
            // max variation is defined as the difference between first and last value
            converter0.raw_data = dpvect[0].payload_pt1;
            converter1.raw_data = dpvect.back().payload_pt1;
            double delta = std::abs(converter0.double_value - converter1.double_value);
            trddcs.maxChange.first = deltatime; // is it ok to do like this, as in Run 2?
            trddcs.maxChange.second = delta;
          } else {
            for (auto i = 0; i < dpvect.size() - 1; ++i) {
              for (auto j = i + 1; j < dpvect.size(); ++j) {
                auto deltatime = dpvect[j].get_epoch_time() - dpvect[i].get_epoch_time();
                if (deltatime >= 60000) { // we check every min; epoch_time in ms
                  converter0.raw_data = dpvect[i].payload_pt1;
                  converter1.raw_data = dpvect[j].payload_pt1;
                  double delta = std::abs(converter0.double_value - converter1.double_value);
                  if (delta > trddcs.maxChange.second) {
                    trddcs.maxChange.first = deltatime; // is it ok to do like this, as in Run 2?
                    trddcs.maxChange.second = delta;
                  }
                }
              }
            }
          }
          // mid point
          auto midIdx = dpvect.size() / 2 - 1;
          trddcs.midValue.first = dpvect[midIdx].get_epoch_time();
          converter0.raw_data = dpvect[midIdx].payload_pt1;
          trddcs.midValue.second = converter0.double_value;
        } else {
          trddcs.maxChange.first = dpvect[0].get_epoch_time();
          converter0.raw_data = dpvect[0].payload_pt1;
          trddcs.maxChange.second = converter0.double_value;
          trddcs.midValue.first = dpvect[0].get_epoch_time();
          converter0.raw_data = dpvect[0].payload_pt1;
          trddcs.midValue.second = converter0.double_value;
        }
      }
      if (mVerbose) {
        LOG(INFO) << "PID = " << it.first.get_alias();
        trddcs.print();
      }
    }
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Ole Schmidt";
  prepareCCDBobjectInfo(mTRDDCS, mccdbDPsInfo, "TRD/Calib/DCSDPs", mTF, md);

  return;
}
