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

/// \file ctpCCDBManager.cxx
/// \author Roman Lietava
#include "CTPWorkflowScalers/ctpCCDBManager.h"
#include "DataFormatsCTP/Configuration.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include <sstream>
#include <regex>
#include "CommonUtils/StringUtils.h"
#include <fairlogger/Logger.h>
using namespace o2::ctp;
std::string ctpCCDBManager::mCCDBHost = "http://o2-ccdb.internal";
std::string ctpCCDBManager::mQCDBHost = "http://ali-qcdb.cern.ch:8083";
// std::string ctpCCDBManager::mQCDBHost = "none";
//
int ctpCCDBManager::saveRunScalersToCCDB(CTPRunScalers& scalers, long timeStart, long timeStop)
{
  // data base
  if (mCCDBHost == "none") {
    LOG(info) << "Scalers not written to CCDB none";
    return 0;
  }
  // CTPActiveRun* run = mActiveRuns[i];
  using namespace std::chrono_literals;
  std::chrono::seconds days3 = 259200s;
  std::chrono::seconds min10 = 600s;
  long time3days = std::chrono::duration_cast<std::chrono::milliseconds>(days3).count();
  long time10min = std::chrono::duration_cast<std::chrono::milliseconds>(min10).count();
  long tmin = timeStart - time10min;
  long tmax = timeStop + time3days;
  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  metadata["runNumber"] = std::to_string(scalers.getRunNumber());
  api.init(mCCDBHost.c_str()); // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  int ret = api.storeAsTFileAny(&(scalers), mCCDBPathCTPScalers, metadata, tmin, tmax);
  if (ret == 0) {
    LOG(info) << "CTP scalers saved in ccdb:" << mCCDBHost << " run:" << scalers.getRunNumber() << " tmin:" << tmin << " tmax:" << tmax;
  } else {
    LOG(fatal) << "Problem writing to database ret:" << ret;
  }
  return ret;
}
int ctpCCDBManager::saveRunScalersToQCDB(CTPRunScalers& scalers, long timeStart, long timeStop)
{
  // data base
  if (mQCDBHost == "none") {
    LOG(info) << "Scalers not written to QCDB none";
    return 0;
  }
  // CTPActiveRun* run = mActiveRuns[i];q
  using namespace std::chrono_literals;
  std::chrono::seconds days3 = 259200s;
  std::chrono::seconds min10 = 600s;
  long time3days = std::chrono::duration_cast<std::chrono::milliseconds>(days3).count();
  long time10min = std::chrono::duration_cast<std::chrono::milliseconds>(min10).count();
  long tmin = timeStart - time10min;
  long tmax = timeStop + time3days;
  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  metadata["runNumber"] = std::to_string(scalers.getRunNumber());
  api.init(mQCDBHost.c_str()); // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  int ret = api.storeAsTFileAny(&(scalers), mQCDBPathCTPScalers, metadata, tmin, tmax);
  if (ret == 0) {
    LOG(info) << "CTP scalers saved in qcdb:" << mQCDBHost << " run:" << scalers.getRunNumber() << " tmin:" << tmin << " tmax:" << tmax;
  } else {
    LOG(fatal) << "CTP scalers Problem writing to database qcdb ret:" << ret;
  }
  return ret;
}
int ctpCCDBManager::saveRunConfigToCCDB(CTPConfiguration* cfg, long timeStart)
{
  // data base
  if (mCCDBHost == "none") {
    LOG(info) << "CTP config not written to CCDB none";
    return 0;
  }
  using namespace std::chrono_literals;
  std::chrono::seconds days3 = 259200s;
  std::chrono::seconds min10 = 600s;
  long time3days = std::chrono::duration_cast<std::chrono::milliseconds>(days3).count();
  long time10min = std::chrono::duration_cast<std::chrono::milliseconds>(min10).count();
  long tmin = timeStart - time10min;
  long tmax = timeStart + time3days;
  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  metadata["runNumber"] = std::to_string(cfg->getRunNumber());
  api.init(mCCDBHost.c_str()); // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  int ret = api.storeAsTFileAny(cfg, CCDBPathCTPConfig, metadata, tmin, tmax);
  if (ret == 0) {
    LOG(info) << "CTP config  saved in ccdb:" << mCCDBHost << " run:" << cfg->getRunNumber() << " tmin:" << tmin << " tmax:" << tmax;
  } else {
    LOG(fatal) << "CTPConfig: Problem writing to database ret:" << ret;
  }
  return ret;
}
CTPConfiguration ctpCCDBManager::getConfigFromCCDB(long timestamp, std::string run, bool& ok)
{
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCCDBHost);
  map<string, string> metadata; // can be empty
  metadata["runNumber"] = run;
  auto ctpconfigdb = mgr.getSpecific<CTPConfiguration>(CCDBPathCTPConfig, timestamp, metadata);
  if (ctpconfigdb == nullptr) {
    LOG(info) << "CTP config not in database, timestamp:" << timestamp;
    ok = 0;
  } else {
    // ctpconfigdb->printStream(std::cout);
    LOG(info) << "CTP config found. Run:" << run;
    ok = 1;
  }
  return *ctpconfigdb;
}
CTPConfiguration ctpCCDBManager::getConfigFromCCDB(long timestamp, std::string run)
{
  bool ok;
  auto ctpconfig = getConfigFromCCDB(timestamp, run, ok);
  if (ok == 0) {
    LOG(error) << "CTP config not in CCDB";
    return CTPConfiguration();
  }
  return ctpconfig;
}
CTPRunScalers ctpCCDBManager::getScalersFromCCDB(long timestamp, std::string run, bool& ok)
{
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCCDBHost);
  map<string, string> metadata; // can be empty
  metadata["runNumber"] = run;
  auto ctpscalers = mgr.getSpecific<CTPRunScalers>(mCCDBPathCTPScalers, timestamp, metadata);
  if (ctpscalers == nullptr) {
    LOG(info) << "CTPRunScalers not in database, timestamp:" << timestamp;
    ok = 0;
  } else {
    // ctpscalers->printStream(std::cout);
    ok = 1;
  }
  return *ctpscalers;
}
