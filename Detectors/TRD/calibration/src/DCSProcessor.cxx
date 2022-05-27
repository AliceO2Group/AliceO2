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

//__________________________________________________________________

void DCSProcessor::init(const std::vector<DPID>& pids)
{
  // fill the array of the DPIDs that will be used by TRD
  // pids should be provided by CCDB

  for (const auto& it : pids) {
    mPids[it] = false;
    mLastDPTimeStamps[it] = 0;
  }
}

//__________________________________________________________________

int DCSProcessor::process(const gsl::span<const DPCOM> dps)
{

  // first we check which DPs are missing - if some are, it means that
  // the delta map was sent
  if (mVerbosity > 0) {
    LOG(info) << "\n\n\nProcessing new TF\n-----------------";
  }

  if (mVerbosity > 1) {
    std::unordered_map<DPID, DPVAL> mapin;
    for (auto& it : dps) {
      mapin[it.id] = it.data;
    }

    for (auto& it : mPids) {
      const auto& el = mapin.find(it.first);
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
      if (mVerbosity > 1) {
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
  if (mVerbosity > 1) {
    if (type == DPVAL_DOUBLE) {
      LOG(info) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<double>(dpcom);
    } else if (type == DPVAL_INT) {
      LOG(info) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<int32_t>(dpcom);
    } else if (type == DPVAL_STRING) {
      LOG(info) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<string>(dpcom);
    }
  }
  auto flags = dpcom.data.get_flags();
  if (processFlags(flags, dpid.get_alias()) == 0) {
    if (type == DPVAL_DOUBLE) {
      auto etime = dpcom.data.get_epoch_time();

      // check if DP is one of the gas values
      if (std::strstr(dpid.get_alias(), "trd_gas") != nullptr) {
        if (!mGasStartTSset) {
          mGasStartTS = mCurrentTS;
          mGasStartTSset = true;
        }
        auto& dpInfoGas = mTRDDCSGas[dpid];
        if (dpInfoGas.nPoints == 0 || etime != mLastDPTimeStamps[dpid]) {
          // only add data point in case it was not already read before
          dpInfoGas.addPoint(o2::dcs::getValue<double>(dpcom));
          mLastDPTimeStamps[dpid] = etime;
        }
      }

      // check if DP is HV current value
      if (std::strstr(dpid.get_alias(), "Imon") != nullptr) {
        if (!mCurrentsStartTSSet) {
          mCurrentsStartTS = mCurrentTS;
          mCurrentsStartTSSet = true;
        }
        auto& dpInfoCurrents = mTRDDCSCurrents[dpid];
        if (dpInfoCurrents.nPoints == 0 || etime != mLastDPTimeStamps[dpid]) {
          // only add data point in case it was not already read before
          dpInfoCurrents.addPoint(o2::dcs::getValue<double>(dpcom));
          mLastDPTimeStamps[dpid] = etime;
        }
      }

      // check if DP is HV voltage value
      if (std::strstr(dpid.get_alias(), "Umon") != nullptr) {
        if (!mVoltagesStartTSSet) {
          mVoltagesStartTS = mCurrentTS;
          mVoltagesStartTSSet = true;
        }
        auto& dpInfoVoltages = mTRDDCSVoltages[dpid];
        if (etime != mLastDPTimeStamps[dpid]) {
          int chamberId = getChamberIdFromAlias(dpid.get_alias());
          if (mVoltageSet.test(chamberId)) {
            if (std::fabs(dpInfoVoltages - o2::dcs::getValue<double>(dpcom)) > 1.f) {
              // trigger update of voltage CCDB object
              mShouldUpdateVoltages = true;
              // OS: this will still overwrite the current voltage value of the object going into the CCDB
              // Should instead the old value be kept until a new obect has been stored in the CCDB?
            }
          }
          dpInfoVoltages = o2::dcs::getValue<double>(dpcom);
          mLastDPTimeStamps[dpid] = etime;
          mVoltageSet.set(chamberId);
        }
      }

      // check if DP is env value
      if (std::strstr(dpid.get_alias(), "trd_aliEnv") != nullptr) {
        if (!mEnvStartTSSet) {
          mEnvStartTS = mCurrentTS;
          mEnvStartTSSet = true;
        }
        auto& dpInfoEnv = mTRDDCSEnv[dpid];
        if (dpInfoEnv.nPoints == 0 || etime != mLastDPTimeStamps[dpid]) {
          // only add data point in case it was not already read before
          dpInfoEnv.addPoint(o2::dcs::getValue<double>(dpcom));
          mLastDPTimeStamps[dpid] = etime;
        }
      }
    }
    if (type == DPVAL_INT) {
      if (std::strstr(dpid.get_alias(), "trd_runNo") != nullptr) { // DP is trd_runNo
        if (!mRunStartTSSet) {
          mRunStartTS = mCurrentTS;
          mRunStartTSSet = true;
        }
        auto& runNumber = mTRDDCSRun[dpid];
        if (mPids[dpid] && runNumber != o2::dcs::getValue<int32_t>(dpcom)) {
          LOGF(info, "Run number has already been processed and the new one %i differs from the old one %i", runNumber, o2::dcs::getValue<int32_t>(dpcom));
          mShouldUpdateRun = true;
          mRunEndTS = mCurrentTS;
        } else {
          runNumber = o2::dcs::getValue<int32_t>(dpcom);
        }
      } else if (std::strstr(dpid.get_alias(), "trd_runType") != nullptr) { // DP is trd_runType
        if (!mRunStartTSSet) {
          mRunStartTS = mCurrentTS;
          mRunStartTSSet = true;
        }
        auto& runType = mTRDDCSRun[dpid];
        if (mPids[dpid] && runType != o2::dcs::getValue<int32_t>(dpcom)) {
          LOGF(info, "Run type has already been processed and the new one %i differs from the old one %i", runType, o2::dcs::getValue<int32_t>(dpcom));
          mShouldUpdateRun = true;
          mRunEndTS = mCurrentTS;
        } else {
          runType = o2::dcs::getValue<int32_t>(dpcom);
        }
      }
    }

    if (type == DPVAL_STRING) {
      if (std::strstr(dpid.get_alias(), "trd_fedCFGtag") != nullptr) { // DP is trd_fedCFGtag
        auto cfgTag = o2::dcs::getValue<std::string>(dpcom);
        if (mVerbosity > 1) {
          LOG(info) << "CFG tag " << dpid.get_alias() << " is " << cfgTag;
        }
      }
    }
  }
  return 0;
}

int DCSProcessor::getChamberIdFromAlias(const char* alias) const
{
  // chamber ID is the last three characaters from the alias
  auto length = strlen(alias);
  std::string id(alias + length - 3, alias + length);
  return std::stoi(id);
}

//______________________________________________________________________

int DCSProcessor::processFlags(const uint64_t flags, const char* pid)
{

  // function to process the flag. the return code zero means that all is fine.
  // anything else means that there was an issue

  // for now, I don't know how to use the flags, so I do nothing

  if (mVerbosity > 0) {
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

bool DCSProcessor::updateGasDPsCCDB()
{
  // here we create the object containing the gas data points to then be sent to CCDB
  LOG(info) << "Preparing CCDB object for TRD gas DPs";

  bool retVal = false; // set to 'true' in case at least one DP for gas has been processed

  for (const auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (std::strstr(it.first.get_alias(), "trd_gas") != nullptr) {
        if (it.second == true) { // we processed the DP at least 1x
          retVal = true;
        }
        if (mVerbosity > 0) {
          LOG(info) << "PID = " << it.first.get_alias();
          mTRDDCSGas[it.first].print();
        }
      }
    }
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Ole Schmidt";
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSGas, mCcdbGasDPsInfo, "TRD/Calib/DCSDPsGas", md, mGasStartTS, mCurrentTS);

  return retVal;
}

bool DCSProcessor::updateCurrentsDPsCCDB()
{
  // here we create the object containing the currents data points to then be sent to CCDB
  LOG(info) << "Preparing CCDB object for TRD currents DPs";

  bool retVal = false; // set to 'true' in case at least one DP has been processed

  for (const auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (std::strstr(it.first.get_alias(), "Imon") != nullptr) {
        if (it.second == true) { // we processed the DP at least 1x
          retVal = true;
        }
        if (mVerbosity > 1) {
          LOG(info) << "PID = " << it.first.get_alias();
          mTRDDCSCurrents[it.first].print();
        }
      }
    }
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Ole Schmidt";
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSCurrents, mCcdbCurrentsDPsInfo, "TRD/Calib/DCSDPsI", md, mCurrentsStartTS, mCurrentTS);

  return retVal;
}

bool DCSProcessor::updateVoltagesDPsCCDB()
{
  // here we create the object containing the voltage data points to then be sent to CCDB
  LOG(info) << "Preparing CCDB object for TRD voltage DPs";

  bool retVal = false; // set to 'true' in case at least one DP has been processed

  for (const auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (std::strstr(it.first.get_alias(), "Umon") != nullptr) {
        if (it.second == true) { // we processed the DP at least 1x
          retVal = true;
        }
        if (mVerbosity > 1) {
          LOG(info) << "PID = " << it.first.get_alias() << " Value = " << mTRDDCSVoltages[it.first];
        }
      }
    }
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Ole Schmidt";
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSVoltages, mCcdbVoltagesDPsInfo, "TRD/Calib/DCSDPsU", md, mVoltagesStartTS, mCurrentTS);

  return retVal;
}

bool DCSProcessor::updateEnvDPsCCDB()
{
  // here we create the object containing the env data points to then be sent to CCDB
  LOG(info) << "Preparing CCDB object for TRD env DPs";

  bool retVal = false; // set to 'true' in case at least one DP for env has been processed

  for (const auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (std::strstr(it.first.get_alias(), "trd_aliEnv") != nullptr) {
        if (it.second == true) { // we processed the DP at least 1x
          retVal = true;
        }
        if (mVerbosity > 0) {
          LOG(info) << "PID = " << it.first.get_alias();
          mTRDDCSEnv[it.first].print();
        }
      }
    }
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Ole Schmidt";
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSEnv, mCcdbEnvDPsInfo, "TRD/Calib/DCSDPsEnv", md, mEnvStartTS, mCurrentTS);

  return retVal;
}

bool DCSProcessor::updateRunDPsCCDB()
{
  // here we create the object containing the run data points to then be sent to CCDB
  LOG(info) << "Preparing CCDB object for TRD run DPs";

  bool retVal = false; // set to 'true' in case at least one DP for run has been processed

  for (const auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (std::strstr(it.first.get_alias(), "trd_run") != nullptr) {
        if (it.second == true) { // we processed the DP at least 1x
          retVal = true;
        }
        if (mVerbosity > 0) {
          LOG(info) << "PID = " << it.first.get_alias() << ". Value = " << mTRDDCSRun[it.first];
        }
      }
    }
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Ole Schmidt";
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSRun, mCcdbRunDPsInfo, "TRD/Calib/DCSDPsRun", md, mRunStartTS, mRunEndTS);

  return retVal;
}

void DCSProcessor::clearCurrentsDPsInfo()
{
  mTRDDCSCurrents.clear();
  mCurrentsStartTSSet = false;
  // reset the 'processed' flags for the currents DPs
  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (std::strstr(it.first.get_alias(), "Imon") != nullptr) {
        it.second = false;
      }
    }
  }
}

void DCSProcessor::clearVoltagesDPsInfo()
{
  mTRDDCSVoltages.clear();
  mVoltagesStartTSSet = false;
  mVoltageSet.reset();
  mShouldUpdateVoltages = false;
  // reset the 'processed' flags for the voltages DPs
  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (std::strstr(it.first.get_alias(), "Umon") != nullptr) {
        it.second = false;
      }
    }
  }
}

void DCSProcessor::clearGasDPsInfo()
{
  // reset the data and the gas CCDB object itself
  mTRDDCSGas.clear();
  mGasStartTSset = false; // the next object will be valid from the first processed time stamp

  // reset the 'processed' flags for the gas DPs
  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (std::strstr(it.first.get_alias(), "trd_gas") != nullptr) {
        it.second = false;
      }
    }
  }
}

void DCSProcessor::clearEnvDPsInfo()
{
  mTRDDCSEnv.clear();
  mEnvStartTSSet = false;
  // reset the 'processed' flags for the gas DPs
  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (std::strstr(it.first.get_alias(), "trd_aliEnv") != nullptr) {
        it.second = false;
      }
    }
  }
}

void DCSProcessor::clearRunDPsInfo()
{
  mTRDDCSRun.clear();
  mRunStartTSSet = false;
  mShouldUpdateRun = false;
  // reset the 'processed' flags for the gas DPs
  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (std::strstr(it.first.get_alias(), "trd_run") != nullptr) {
        it.second = false;
      }
    }
  }
}
