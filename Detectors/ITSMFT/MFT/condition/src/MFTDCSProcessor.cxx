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

#include <MFTCondition/MFTDCSProcessor.h>
#include "Rtypes.h"
#include <deque>
#include <string>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <bitset>

using namespace o2::mft;
using namespace o2::dcs;

using DeliveryType = o2::dcs::DeliveryType;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;

ClassImp(o2::mft::MFTDCSinfo);

void MFTDCSinfo::print() const
{
  LOG(info) << "First Value: timestamp = " << firstValue.first << ", value = " << firstValue.second;
  LOG(info) << "Last Value:  timestamp = " << lastValue.first << ", value = " << lastValue.second;
  LOG(info) << "Mid Value:   timestamp = " << midValue.first << ", value = " << midValue.second;
  LOG(info) << "Max Change:  timestamp = " << maxChange.first << ", value = " << maxChange.second;
}

//__________________________________________________________________

void MFTDCSProcessor::init(const std::vector<DPID>& pids)
{
  // fill the array of the DPIDs that will be used by MFT
  // pids should be provided by CCDB

  for (const auto& it : pids) {
    mPids[it] = false;
    mMFTDCS[it].makeEmpty();
  }
}

//__________________________________________________________________

int MFTDCSProcessor::process(const gsl::span<const DPCOM> dps)
{

  // first we check which DPs are missing - if some are, it means that
  // the delta map was sent
  if (mVerbose) {
    LOG(info) << "\n\n\nProcessing new TF\n-----------------";
  }
  if (!mStartTFset) {
    mStartTF = mTF;
    mStartTFset = true;
  }

  if (false) { // RS: why do we need to lose CPU on this dummy check?
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
      LOG(info) << "DP " << it.id << " not found in MFTDCSProcessor, we will not process it";
      continue;
    }
    /*
    //it.id = DataPointIdentifier
    //id.data = DataPointValue
    const DPCOM new_it(new_id,it.data);
    processDP(new_it);
    */
    processDP(it);
    mPids[it.id] = true;
  }

  //updateCurrentAnalogCCDB();
  updateDPsCCDB();

  return 0;
}

//__________________________________________________________________

int MFTDCSProcessor::processDP(const DPCOM& dpcom)
{

  // processing single DP

  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  auto& val = dpcom.data;
  if (mVerbose) {
    if (type == DPVAL_DOUBLE) {
      LOG(info);
      LOG(info) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<double>(dpcom);
    } else if (type == DPVAL_INT) {
      LOG(info);
      LOG(info) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<int32_t>(dpcom);
    }
  }

  auto flags = val.get_flags();

  // now I need to access the correct element
  if (type == DPVAL_DOUBLE) {
    // for these DPs, we will store the first, last, mid value, plus the value where the maximum variation occurred
    auto& dvect = mDpsdoublesmap[dpid];
    if (mVerbose) {
      LOG(info) << "mDpsdoublesmap[dpid].size() = " << dvect.size();
    }
    auto etime = val.get_epoch_time();
    if (dvect.size() == 0 || etime != dvect.back().get_epoch_time()) { // we check
                                                                       // that we did not get the
                                                                       // same timestamp as the
                                                                       // latest one
      dvect.push_back(val);
    }
  }

  return 0;
}

//______________________________________________________________________

void MFTDCSProcessor::updateDPsCCDB()
{

  // here we create the object to then be sent to CCDB
  if (mVerbose) {
    LOG(info) << "Finalizing";
  }
  union Converter {
    uint64_t raw_data;
    double double_value;
  } converter0, converter1;

  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      auto& mftdcs = mMFTDCS[it.first];
      if (it.second) {     // we processed the DP at least 1x
        it.second = false; // once the point was used, reset it
        auto& dpvect = mDpsdoublesmap[it.first];
        mftdcs.firstValue.first = dpvect[0].get_epoch_time();
        converter0.raw_data = dpvect[0].payload_pt1;
        mftdcs.firstValue.second = converter0.double_value;
        mftdcs.lastValue.first = dpvect.back().get_epoch_time();
        converter0.raw_data = dpvect.back().payload_pt1;
        mftdcs.lastValue.second = converter0.double_value;
        // now I will look for the max change
        if (dpvect.size() > 1) {
          auto deltatime = dpvect.back().get_epoch_time() - dpvect[0].get_epoch_time();
          if (deltatime < 60000) {
            // if we did not cover at least 1 minute,
            // max variation is defined as the difference between first and last value
            converter0.raw_data = dpvect[0].payload_pt1;
            converter1.raw_data = dpvect.back().payload_pt1;
            double delta = std::abs(converter0.double_value - converter1.double_value);
            mftdcs.maxChange.first = deltatime; // is it ok to do like this, as in Run 2?
            mftdcs.maxChange.second = delta;
          } else {
            for (auto i = 0; i < dpvect.size() - 1; ++i) {
              for (auto j = i + 1; j < dpvect.size(); ++j) {
                auto deltatime = dpvect[j].get_epoch_time() - dpvect[i].get_epoch_time();
                if (deltatime >= 60000) { // we check every min; epoch_time in ms
                  converter0.raw_data = dpvect[i].payload_pt1;
                  converter1.raw_data = dpvect[j].payload_pt1;
                  double delta = std::abs(converter0.double_value - converter1.double_value);
                  if (delta > mftdcs.maxChange.second) {
                    mftdcs.maxChange.first = deltatime; // is it ok to do like this, as in Run 2?
                    mftdcs.maxChange.second = delta;
                  }
                }
              }
            }
          }
          // mid point
          auto midIdx = dpvect.size() / 2 - 1;
          mftdcs.midValue.first = dpvect[midIdx].get_epoch_time();
          converter0.raw_data = dpvect[midIdx].payload_pt1;
          mftdcs.midValue.second = converter0.double_value;
        } else {
          mftdcs.maxChange.first = dpvect[0].get_epoch_time();
          converter0.raw_data = dpvect[0].payload_pt1;
          mftdcs.maxChange.second = converter0.double_value;
          mftdcs.midValue.first = dpvect[0].get_epoch_time();
          converter0.raw_data = dpvect[0].payload_pt1;
          mftdcs.midValue.second = converter0.double_value;
        }
      }
      if (mVerbose) {
        LOG(info) << "PID = " << it.first.get_alias();
        mftdcs.print();
      }
    }
  }

  std::map<std::string, std::string> md;
  md["responsible"] = "Satoshi Yano";
  prepareCCDBobjectInfo(mMFTDCS, mccdbDPsInfo, "MFT/Condition/DCSDPs", mTF, md);

  return;
}
