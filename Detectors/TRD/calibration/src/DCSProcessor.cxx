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

  // LB: setup counters for ChamberStatus/CFGtag logic
  int ChamberStatusDPsCounter = 0;
  int CFGtagDPsCounter = 0;

  std::unordered_map<DPID, DPVAL> mapin;
  for (auto& it : dps) {
    mapin[it.id] = it.data;

    // LB: check if all ChamberStatus/CFGtag DPs were sent in dps
    // if counter is equal to mFedMinimunDPsForUpdate (522) => all DPs were sent
    if (std::strstr(it.id.get_alias(), "trd_chamberStatus") != nullptr) {
      ChamberStatusDPsCounter++;
    } else if (std::strstr(it.id.get_alias(), "trd_CFGtag") != nullptr) {
      CFGtagDPsCounter++;
    }
  }

  if (ChamberStatusDPsCounter >= mFedMinimunDPsForUpdate) {
    mFedChamberStatusCompleteDPs = true;
    if (mVerbosity > 1) {
      LOG(info) << "Minimum number of required DPs (" << mFedMinimunDPsForUpdate << ") for ChamberStatus update were found.";
    }
  }
  if (CFGtagDPsCounter >= mFedMinimunDPsForUpdate) {
    mFedCFGtagCompleteDPs = true;
    if (mVerbosity > 1) {
      LOG(info) << "Minimum number of required DPs (" << mFedMinimunDPsForUpdate << ") for CFGtag update were found.";
    }
  }
  if (mVerbosity > 1) {
    LOG(info) << "Number of ChamberStatus DPs = " << ChamberStatusDPsCounter;
    LOG(info) << "Number of CFGtag DPs = " << CFGtagDPsCounter;
  }

  if (mVerbosity > 1) {
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
    auto etime = dpcom.data.get_epoch_time();

    // DPs are sorted by type variable
    if (type == DPVAL_DOUBLE) {

      // check if DP is one of the gas values
      if (std::strstr(dpid.get_alias(), "trd_gas") != nullptr) {
        if (!mGasStartTSset) {
          mGasStartTS = mCurrentTS;
          mGasStartTSset = true;
        }
        auto& dpInfoGas = mTRDDCSGas[dpid];
        if (dpInfoGas.nPoints == 0 || etime != mLastDPTimeStamps[dpid]) {
          // only add data point in case it was not already read before
          dpInfoGas.addPoint(o2::dcs::getValue<double>(dpcom), etime);
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
          dpInfoCurrents.addPoint(o2::dcs::getValue<double>(dpcom), etime);
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
            if (std::fabs(dpInfoVoltages - o2::dcs::getValue<double>(dpcom)) > mUVariationTriggerForUpdate) {
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
      if (isAliasFromEnvDP(dpid.get_alias())) {
        if (!mEnvStartTSSet) {
          mEnvStartTS = mCurrentTS;
          mEnvStartTSSet = true;
        }
        auto& dpInfoEnv = mTRDDCSEnv[dpid];
        if (dpInfoEnv.nPoints == 0 || etime != mLastDPTimeStamps[dpid]) {
          // only add data point in case it was not already read before
          dpInfoEnv.addPoint(o2::dcs::getValue<double>(dpcom), etime);
          mLastDPTimeStamps[dpid] = etime;
        }
      }
    }

    if (type == DPVAL_INT) {
      if (std::strstr(dpid.get_alias(), "trd_fed_runNo") != nullptr) { // DP is trd_fed_runNo
        if (!mRunStartTSSet) {
          mRunStartTS = mCurrentTS;
          mRunStartTSSet = true;
        }

        auto& runNumber = mTRDDCSRun[dpid];

        // LB: Check if new value is a valid run number (0 = cleared variable)
        if (o2::dcs::getValue<int32_t>(dpcom) > 0) {
          // If value has changed from previous one, new run has begun and update
          if (o2::dcs::getValue<int32_t>(dpcom) != mCurrentRunNumber) {
            LOG(info) << "New run number " << o2::dcs::getValue<int32_t>(dpcom) << " differs from the old one " << mCurrentRunNumber;
            mShouldUpdateRun = true;
            // LB: two different flags as they reset separately, after upload of CCDB, for each object
            mFirstRunEntryForFedChamberStatusUpdate = true;
            mFirstRunEntryForFedCFGtagUpdate = true;
            // LB: reset alarm counters
            mFedChamberStatusAlarmCounter = 0;
            mFedCFGtagAlarmCounter = 0;
            mRunEndTS = mCurrentTS;
          }

          // LB: Save current run number
          mCurrentRunNumber = o2::dcs::getValue<int32_t>(dpcom);
          // Save to mTRDDCSRun
          runNumber = mCurrentRunNumber;
        }

        if (mVerbosity > 2) {
          LOG(info) << "Current Run Number: " << mCurrentRunNumber;
        }

      } else if (std::strstr(dpid.get_alias(), "trd_chamberStatus") != nullptr) { // DP is trd_chamberStatus
        if (!mFedChamberStatusStartTSSet) {
          mFedChamberStatusStartTS = mCurrentTS;
          mFedChamberStatusStartTSSet = true;
        }

        // LB: for ChamberStatus, grab the chamber number from alias
        int chamberId = getChamberIdFromAlias(dpid.get_alias());
        auto& dpInfoFedChamberStatus = mTRDDCSFedChamberStatus[chamberId];
        if (etime != mLastDPTimeStamps[dpid]) {
          if (dpInfoFedChamberStatus != o2::dcs::getValue<int>(dpcom)) {
            // If value changes after processing and DPs should not be updated, log change as warning (for now)
            if (mPids[dpid] && !(mFedChamberStatusCompleteDPs && mFirstRunEntryForFedChamberStatusUpdate)) {
              // Issue an alarm if counter is lower than maximum, warning otherwise
              // LB: set both to warnings, conditions are kept if future changes are needed
              if (mFedChamberStatusAlarmCounter < mFedAlarmCounterMax) {
                LOG(warn) << "ChamberStatus change " << dpid.get_alias() << " : " << dpInfoFedChamberStatus << " -> " << o2::dcs::getValue<int>(dpcom) << ", run = " << mCurrentRunNumber;
                mFedChamberStatusAlarmCounter++;
              } else if (mVerbosity > 0) {
                LOG(warn) << "ChamberStatus change " << dpid.get_alias() << " : " << dpInfoFedChamberStatus << " -> " << o2::dcs::getValue<int>(dpcom) << ", run = " << mCurrentRunNumber;
              }
            }
          }

          dpInfoFedChamberStatus = o2::dcs::getValue<int>(dpcom);
          mLastDPTimeStamps[dpid] = etime;
        }
      }
    }

    if (type == DPVAL_STRING) {
      if (std::strstr(dpid.get_alias(), "trd_CFGtag") != nullptr) { // DP is trd_CFGtag
        if (!mFedCFGtagStartTSSet) {
          mFedCFGtagStartTS = mCurrentTS;
          mFedCFGtagStartTSSet = true;
        }

        // LB: for CFGtag, grab the chamber number from alias
        int chamberId = getChamberIdFromAlias(dpid.get_alias());
        auto& dpInfoFedCFGtag = mTRDDCSFedCFGtag[chamberId];
        if (etime != mLastDPTimeStamps[dpid]) {
          if (dpInfoFedCFGtag != o2::dcs::getValue<string>(dpcom)) {
            // If value changes after processing and DPs should not be updated, log change as warning (for now)
            if (mPids[dpid] && !(mFedCFGtagCompleteDPs && mFirstRunEntryForFedCFGtagUpdate)) {
              // Issue an alarm if counter is lower than maximum, warning otherwise
              if (mFedCFGtagAlarmCounter < mFedAlarmCounterMax) {
                LOG(alarm) << "CFGtag change " << dpid.get_alias() << " : " << dpInfoFedCFGtag << " -> " << o2::dcs::getValue<string>(dpcom) << ", run = " << mCurrentRunNumber;
                mFedCFGtagAlarmCounter++;
              } else if (mVerbosity > 0) {
                LOG(warn) << "CFGtag change " << dpid.get_alias() << " : " << dpInfoFedCFGtag << " -> " << o2::dcs::getValue<string>(dpcom) << ", run = " << mCurrentRunNumber;
              }
            }
          }

          dpInfoFedCFGtag = o2::dcs::getValue<std::string>(dpcom);
          mLastDPTimeStamps[dpid] = etime;
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
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSGas, mCcdbGasDPsInfo, "TRD/Calib/DCSDPsGas", md, mGasStartTS, mGasStartTS + 3 * o2::ccdb::CcdbObjectInfo::DAY);

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
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSCurrents, mCcdbCurrentsDPsInfo, "TRD/Calib/DCSDPsI", md, mCurrentsStartTS, mCurrentsStartTS + 3 * o2::ccdb::CcdbObjectInfo::DAY);

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
          LOG(info) << "PID = " << it.first.get_alias() << ". Value = " << mTRDDCSVoltages[it.first];
        }
      }
    }
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Ole Schmidt";
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSVoltages, mCcdbVoltagesDPsInfo, "TRD/Calib/DCSDPsU", md, mVoltagesStartTS, mVoltagesStartTS + 7 * o2::ccdb::CcdbObjectInfo::DAY);

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
      if (isAliasFromEnvDP(it.first.get_alias())) {
        if (it.second == true) { // we processed the DP at least 1x
          retVal = true;
        }
        if (mVerbosity > 1) {
          LOG(info) << "PID = " << it.first.get_alias();
          mTRDDCSEnv[it.first].print();
        }
      }
    }
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Leonardo Barreto";
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSEnv, mCcdbEnvDPsInfo, "TRD/Calib/DCSDPsEnv", md, mEnvStartTS, mEnvStartTS + 3 * o2::ccdb::CcdbObjectInfo::DAY);

  return retVal;
}

bool DCSProcessor::updateRunDPsCCDB()
{
  // here we create the object containing the run data points to then be sent to CCDB
  LOG(info) << "Preparing CCDB object for TRD run DPs";

  bool retVal = false; // set to 'true' in case at least one DP for run has been processed

  for (const auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_INT) {
      if (std::strstr(it.first.get_alias(), "trd_fed_run") != nullptr) {
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
  md["responsible"] = "Leonardo Barreto";
  // Redundancy for testing, this object is updated after run ended, so need to write old run number, not current
  // md["runNumber"] = std::to_string(mFinishedRunNumber);
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSRun, mCcdbRunDPsInfo, "TRD/Calib/DCSDPsRun", md, mRunStartTS, mRunEndTS);

  // LB: Deactivated upload of Run DPs to CCDB even if processed
  // To turn it back on just comment the next line
  retVal = false;
  return retVal;
}

bool DCSProcessor::updateFedChamberStatusDPsCCDB()
{
  // here we create the object containing the fedChamberStatus data points to then be sent to CCDB
  LOG(info) << "Preparing CCDB object for TRD fedChamberStatus DPs";

  bool retVal = false; // set to 'true' in case at least one DP for run has been processed

  for (const auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_INT) {
      if (std::strstr(it.first.get_alias(), "trd_chamberStatus") != nullptr) {
        if (it.second == true) { // we processed the DP at least 1x
          retVal = true;
        }
        if (mVerbosity > 1) {
          int chamberId = getChamberIdFromAlias(it.first.get_alias());
          LOG(info) << "PID = " << it.first.get_alias() << ". Value = " << mTRDDCSFedChamberStatus[chamberId];
        }
      }
    }
  }

  std::map<std::string, std::string> md;
  md["responsible"] = "Leonardo Barreto";
  md["runNumber"] = std::to_string(mCurrentRunNumber);
  // LB: set start timestamp 30 seconds before DPs are received
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSFedChamberStatus, mCcdbFedChamberStatusDPsInfo,
                                                "TRD/Calib/DCSDPsFedChamberStatus", md, mFedChamberStatusStartTS - 30,
                                                mFedChamberStatusStartTS + 3 * o2::ccdb::CcdbObjectInfo::DAY);

  return retVal;
}

bool DCSProcessor::updateFedCFGtagDPsCCDB()
{
  // here we create the object containing the fedCFGtag data points to then be sent to CCDB
  LOG(info) << "Preparing CCDB object for TRD fedCFGtag DPs";

  bool retVal = false; // set to 'true' in case at least one DP for run has been processed

  for (const auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_STRING) {
      if (std::strstr(it.first.get_alias(), "trd_CFGtag") != nullptr) {
        if (it.second == true) { // we processed the DP at least 1x
          retVal = true;
        }
        if (mVerbosity > 1) {
          int chamberId = getChamberIdFromAlias(it.first.get_alias());
          LOG(info) << "PID = " << it.first.get_alias() << ". Value = " << mTRDDCSFedCFGtag[chamberId];
        }
      }
    }
  }

  std::map<std::string, std::string> md;
  md["responsible"] = "Leonardo Barreto";
  md["runNumber"] = std::to_string(mCurrentRunNumber);
  // LB: set start timestamp 30 seconds before DPs are received
  o2::calibration::Utils::prepareCCDBobjectInfo(mTRDDCSFedCFGtag, mCcdbFedCFGtagDPsInfo,
                                                "TRD/Calib/DCSDPsFedCFGtag", md, mFedCFGtagStartTS - 30,
                                                mFedCFGtagStartTS + 3 * o2::ccdb::CcdbObjectInfo::DAY);

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
  // reset the 'processed' flags for the env DPs
  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      if (isAliasFromEnvDP(it.first.get_alias())) {
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
  // reset the 'processed' flags for the run DPs
  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_INT) {
      if (std::strstr(it.first.get_alias(), "trd_fed_run") != nullptr) {
        it.second = false;
      }
    }
  }
}

void DCSProcessor::clearFedChamberStatusDPsInfo()
{
  // mTRDDCSFedChamberStatus should not be cleared after upload giving alarm/warn logic
  mFedChamberStatusStartTSSet = false;
  mFedChamberStatusCompleteDPs = false;
  mFirstRunEntryForFedChamberStatusUpdate = false;
  // reset the 'processed' flags for the fed DPs
  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_INT) {
      if (std::strstr(it.first.get_alias(), "trd_chamberStatus") != nullptr) {
        it.second = false;
      }
    }
  }
}

void DCSProcessor::clearFedCFGtagDPsInfo()
{
  // mTRDDCSFedCFGtag should not be cleared after upload giving alarm/warn logic
  mFedCFGtagStartTSSet = false;
  mFedCFGtagCompleteDPs = false;
  mFirstRunEntryForFedCFGtagUpdate = false;
  // reset the 'processed' flags for the fed DPs
  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_STRING) {
      if (std::strstr(it.first.get_alias(), "trd_CFGtag") != nullptr) {
        it.second = false;
      }
    }
  }
}
