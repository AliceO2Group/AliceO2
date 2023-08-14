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

/// \file RunManager.cxx
/// \author Roman Lietava

#include "DataFormatsCTP/Configuration.h"
#include "DataFormatsCTP/RunManager.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include <sstream>
#include <regex>
#include "CommonUtils/StringUtils.h"
#include <fairlogger/Logger.h>
using namespace o2::ctp;
std::string CTPRunManager::mCCDBHost = "http://o2-ccdb.internal";
///
/// Run Manager to manage Config and Scalers
///
void CTPRunManager::init()
{
  for (auto r : mActiveRuns) {
    r = nullptr;
  }
  loadScalerNames();
  LOG(info) << "CCDB host:" << mCCDBHost;
  LOG(info) << "CTP vNew:" << mNew;
  LOG(info) << "CTPRunManager initialised.";
}
int CTPRunManager::loadRun(const std::string& cfg)
{
  LOG(info) << "Loading run: " << cfg;
  auto now = std::chrono::system_clock::now();
  long timeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  LOG(info) << "Timestamp real time:" << timeStamp;
  size_t pos = cfg.find(" ");
  std::string cfgmod = cfg;
  if (pos == std::string::npos) {
    LOG(warning) << "Space not found in CTP config";
  } else {
    std::string f = cfg.substr(0, pos);
    if (f.find("run") == std::string::npos) {
      double_t tt = std::stold(f);
      timeStamp = (tt * 1000.);
      LOG(info) << "Timestamp file:" << timeStamp;
      cfgmod = cfg.substr(pos, cfg.size());
      LOG(info) << "ctpcfg: using ctp time";
    }
  }
  CTPActiveRun* activerun = new CTPActiveRun;
  activerun->timeStart = timeStamp;
  activerun->cfg.loadConfigurationRun3(cfgmod);
  activerun->cfg.printStream(std::cout);
  //
  uint32_t runnumber = activerun->cfg.getRunNumber();
  activerun->scalers.setRunNumber(runnumber);
  activerun->scalers.setClassMask(activerun->cfg.getTriggerClassMask());
  o2::detectors::DetID::mask_t detmask = activerun->cfg.getDetectorMask();
  activerun->scalers.setDetectorMask(detmask);
  //
  mRunsLoaded[runnumber] = activerun;
  saveRunConfigToCCDB(&activerun->cfg, timeStamp);
  return 0;
}
int CTPRunManager::startRun(const std::string& cfg)
{
  return 0;
}
int CTPRunManager::stopRun(uint32_t irun, long timeStamp)
{
  LOG(info) << "Stopping run index: " << irun;
  if (mActiveRuns[irun] == nullptr) {
    LOG(error) << "No config for run index:" << irun;
    return 1;
  }
  // const auto now = std::chrono::system_clock::now();
  // const long timeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  mActiveRuns[irun]->timeStop = timeStamp * 1000.;
  saveRunScalersToCCDB(irun);
  delete mActiveRuns[irun];
  mActiveRuns[irun] = nullptr;
  return 0;
}
int CTPRunManager::addScalers(uint32_t irun, std::time_t time)
{
  if (mActiveRuns[irun] == nullptr) {
    LOG(error) << "No config for run index:" << irun;
    return 1;
  }
  std::string orb = "extorb";
  std::string orbitid = "orbitid";
  CTPScalerRecordRaw scalrec;
  scalrec.epochTime = time;
  std::vector<int> clslist = mActiveRuns[irun]->cfg.getTriggerClassList();
  for (auto const& cls : clslist) {
    std::string cmb = "clamb" + std::to_string(cls + 1);
    std::string cma = "clama" + std::to_string(cls + 1);
    std::string c0b = "cla0b" + std::to_string(cls + 1);
    std::string c0a = "cla0a" + std::to_string(cls + 1);
    std::string c1b = "cla1b" + std::to_string(cls + 1);
    std::string c1a = "cla1a" + std::to_string(cls + 1);
    CTPScalerRaw scalraw;
    scalraw.classIndex = (uint32_t)cls;
    // std::cout << "cls:" << cls << " " << scalraw.classIndex << std::endl;
    scalraw.lmBefore = mCounters[mScalerName2Position[cmb]];
    scalraw.lmAfter = mCounters[mScalerName2Position[cma]];
    scalraw.l0Before = mCounters[mScalerName2Position[c0b]];
    scalraw.l0After = mCounters[mScalerName2Position[c0a]];
    scalraw.l1Before = mCounters[mScalerName2Position[c1b]];
    scalraw.l1After = mCounters[mScalerName2Position[c1a]];
    // std::cout << "positions:" << cmb << " " <<  mScalerName2Position[cmb] << std::endl;
    // std::cout << "positions:" << cma << " " <<  mScalerName2Position[cma] << std::endl;
    scalrec.scalers.push_back(scalraw);
  }
  // detectors
  // std::vector<std::string> detlist = mActiveRuns[irun]->cfg.getDetectorList();
  o2::detectors::DetID::mask_t detmask = mActiveRuns[irun]->cfg.getDetectorMask();
  for (uint32_t i = 0; i < 32; i++) {
    o2::detectors::DetID::mask_t deti = 1ul << i;
    bool detin = (detmask & deti).count();
    if (detin) {
      std::string detname(o2::detectors::DetID::getName(i));
      std::string countername = "ltg" + CTPConfiguration::detName2LTG.at(detname) + "_PH";
      uint32_t detcount = mCounters[mScalerName2Position[countername]];
      scalrec.scalersDets.push_back(detcount);
      // LOG(info) << "Scaler for detector:" << countername << ":" << detcount;
    }
  }
  //
  if (mNew == 0) {
    scalrec.intRecord.orbit = mCounters[mScalerName2Position[orb]];
  } else {
    scalrec.intRecord.orbit = mCounters[mScalerName2Position[orbitid]];
  }
  scalrec.intRecord.bc = 0;
  mActiveRuns[irun]->scalers.addScalerRacordRaw(scalrec);
  LOG(info) << "Adding scalers for orbit:" << scalrec.intRecord.orbit;
  // scalrec.printStream(std::cout);
  // printCounters();
  return 0;
}
int CTPRunManager::processMessage(std::string& topic, const std::string& message)
{
  LOG(info) << "Processing message with topic:" << topic;
  std::string firstcounters;
  if (topic.find("ctpconfig") != std::string::npos) {
    LOG(info) << "ctpconfig received";
    loadRun(message);
    return 0;
  }
  if (topic.find("sox") != std::string::npos) {
    // get config
    size_t irun = message.find("run");
    if (irun == std::string::npos) {
      LOG(warning) << "run keyword not found in SOX";
      irun = message.size();
    }
    LOG(info) << "SOX received, Run keyword position:" << irun;
    std::string cfg = message.substr(irun, message.size() - irun);
    startRun(cfg);
    firstcounters = message.substr(0, irun);
  }
  if (topic.find("eox") != std::string::npos) {
    LOG(info) << "EOX received";
    mEOX = 1;
  }
  //
  std::vector<std::string> tokens;
  if (firstcounters.size() > 0) {
    tokens = o2::utils::Str::tokenize(firstcounters, ' ');
  } else {
    tokens = o2::utils::Str::tokenize(message, ' ');
  }
  if (tokens.size() != (CTPRunScalers::NCOUNTERS + 1)) {
    if (tokens.size() == (CTPRunScalers::NCOUNTERS)) {
      mNew = 0;
      LOG(warning) << "v1 scaler size, using external orbit";
    } else {
      LOG(error) << "Scalers size wrong:" << tokens.size() << " expected:" << CTPRunScalers::NCOUNTERS + 1;
      return 1;
    }
  }
  double timeStamp = std::stold(tokens.at(0));
  std::time_t tt = timeStamp;
  LOG(info) << "Processing scalers, all good, time:" << tokens.at(0) << " " << std::asctime(std::localtime(&tt));
  for (int i = 1; i < tokens.size(); i++) {
    mCounters[i - 1] = std::stoull(tokens.at(i));
    if (i < (NRUNS + 1)) {
      std::cout << mCounters[i - 1] << " ";
    }
  }
  std::cout << std::endl;
  LOG(info) << "Counter size:" << tokens.size();
  //
  for (uint32_t i = 0; i < NRUNS; i++) {
    if ((mCounters[i] == 0) && (mActiveRunNumbers[i] == 0)) {
      // not active
    } else if ((mCounters[i] != 0) && (mActiveRunNumbers[i] == mCounters[i])) {
      // active , do scalers
      LOG(info) << "Run continue:" << mCounters[i];
      addScalers(i, tt);
    } else if ((mCounters[i] != 0) && (mActiveRunNumbers[i] == 0)) {
      LOG(info) << "Run started:" << mCounters[i];
      auto run = mRunsLoaded.find(mCounters[i]);
      if (run != mRunsLoaded.end()) {
        mActiveRunNumbers[i] = mCounters[i];
        mActiveRuns[i] = run->second;
        mRunsLoaded.erase(run);
        addScalers(i, tt);
      } else {
        LOG(error) << "Trying to start run which is not loaded:" << mCounters[i];
      }
    } else if ((mCounters[i] == 0) && (mActiveRunNumbers[i] != 0)) {
      if (mEOX != 1) {
        LOG(error) << "Internal error in processMessage: mEOX != 1 expected 0: mEOX:" << mEOX;
      }
      LOG(info) << "Run stopped:" << mActiveRunNumbers[i];
      addScalers(i, tt);
      mActiveRunNumbers[i] = 0;
      mEOX = 0;
      stopRun(i, tt);
    }
  }
  mEOX = 0;
  printActiveRuns();
  return 0;
}
void CTPRunManager::printActiveRuns() const
{
  std::cout << "Active runs:";
  for (auto const& arun : mActiveRunNumbers) {
    std::cout << arun << " ";
  }
  std::cout << " #loaded runs:" << mRunsLoaded.size();
  for (auto const& lrun : mRunsLoaded) {
    std::cout << " " << lrun.second->cfg.getRunNumber();
  }
  std::cout << std::endl;
}
int CTPRunManager::saveRunScalersToCCDB(int i)
{
  // data base
  CTPActiveRun* run = mActiveRuns[i];
  using namespace std::chrono_literals;
  std::chrono::seconds days3 = 259200s;
  std::chrono::seconds min10 = 600s;
  long time3days = std::chrono::duration_cast<std::chrono::milliseconds>(days3).count();
  long time10min = std::chrono::duration_cast<std::chrono::milliseconds>(min10).count();
  long tmin = run->timeStart - time10min;
  long tmax = run->timeStop + time3days;
  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  metadata["runNumber"] = std::to_string(run->cfg.getRunNumber());
  api.init(mCCDBHost.c_str()); // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&(run->scalers), mCCDBPathCTPScalers, metadata, tmin, tmax);
  LOG(info) << "CTP scalers saved in ccdb:" << mCCDBHost << " run:" << run->cfg.getRunNumber() << " tmin:" << tmin << " tmax:" << tmax;
  return 0;
}
int CTPRunManager::saveRunConfigToCCDB(CTPConfiguration* cfg, long timeStart)
{
  // data base
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
  api.storeAsTFileAny(cfg, CCDBPathCTPConfig, metadata, tmin, tmax);
  LOG(info) << "CTP config  saved in ccdb:" << mCCDBHost << " run:" << cfg->getRunNumber() << " tmin:" << tmin << " tmax:" << tmax;
  return 0;
}
CTPConfiguration CTPRunManager::getConfigFromCCDB(long timestamp, std::string run)
{
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCCDBHost);
  map<string, string> metadata; // can be empty
  metadata["runNumber"] = run;
  auto ctpconfigdb = mgr.getSpecific<CTPConfiguration>(CCDBPathCTPConfig, timestamp, metadata);
  if (ctpconfigdb == nullptr) {
    LOG(info) << "CTP config not in database, timestamp:" << timestamp;
  } else {
    // ctpconfigdb->printStream(std::cout);
    LOG(info) << "CTP config found. Run:" << run;
  }
  return *ctpconfigdb;
}
CTPRunScalers CTPRunManager::getScalersFromCCDB(long timestamp, std::string run, bool& ok)
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
int CTPRunManager::loadScalerNames()
{
  if (CTPRunScalers::NCOUNTERS != CTPRunScalers::scalerNames.size()) {
    LOG(fatal) << "NCOUNTERS:" << CTPRunScalers::NCOUNTERS << " different from names vector:" << CTPRunScalers::scalerNames.size();
    return 1;
  }
  // try to open files of no success use default
  for (uint32_t i = 0; i < CTPRunScalers::scalerNames.size(); i++) {
    mScalerName2Position[CTPRunScalers::scalerNames[i]] = i;
  }
  return 0;
}
void CTPRunManager::printCounters()
{
  int NDET = 18;
  int NINPS = 48;
  int NCLKFP = 7;
  int NLTG_start = NRUNS;
  int NCLKFP_start = NLTG_start + NDET * 32;
  int NINPS_start = NCLKFP_start + 7;
  int NCLS_start = NINPS_start + NINPS;
  std::cout << "====> CTP counters:" << std::endl;
  std::cout << "RUNS:" << std::endl;
  int ipos = 0;
  for (int i = 0; i < NRUNS; i++) {
    std::cout << ipos << ":" << mCounters[i] << " ";
    ipos++;
  }
  std::cout << std::endl;
  for (int i = 0; i < NDET; i++) {
    std::cout << "LTG" << i + 1 << std::endl;
    for (int j = NLTG_start + i * 32; j < NLTG_start + (i + 1) * 32; j++) {
      std::cout << ipos << ":" << mCounters[j] << " ";
      ipos++;
    }
    std::cout << std::endl;
  }
  std::cout << "BC40,BC240,Orbit,pulser, fastlm, busy,spare" << std::endl;
  for (int i = NCLKFP_start; i < NCLKFP_start + 7; i++) {
    std::cout << ipos << ":" << mCounters[i] << " ";
    ipos++;
  }
  std::cout << std::endl;
  std::cout << "INPUTS:" << std::endl;
  for (int i = NINPS_start; i < NINPS_start + NINPS; i++) {
    std::cout << ipos << ":" << mCounters[i] << " ";
    ipos++;
  }
  std::cout << std::endl;
  std::cout << "CLASS M Before" << std::endl;
  for (int i = NCLS_start; i < NCLS_start + 64; i++) {
    std::cout << ipos << ":" << mCounters[i] << " ";
    ipos++;
  }
  std::cout << std::endl;
  std::cout << "CLASS M After" << std::endl;
  for (int i = NCLS_start + 64; i < NCLS_start + 2 * 64; i++) {
    std::cout << ipos << ":" << mCounters[i] << " ";
    ipos++;
  }
  std::cout << std::endl;
  std::cout << "CLASS 0 Before" << std::endl;
  for (int i = NCLS_start + 2 * 64; i < NCLS_start + 3 * 64; i++) {
    std::cout << ipos << ":" << mCounters[i] << " ";
    ipos++;
  }
  std::cout << std::endl;
  std::cout << "CLASS 0 After" << std::endl;
  for (int i = NCLS_start + 3 * 64; i < NCLS_start + 4 * 64; i++) {
    std::cout << ipos << ":" << mCounters[i] << " ";
    ipos++;
  }
  std::cout << std::endl;
  std::cout << "CLASS 1 Before" << std::endl;
  for (int i = NCLS_start + 4 * 64; i < NCLS_start + 5 * 64; i++) {
    std::cout << ipos << ":" << mCounters[i] << " ";
    ipos++;
  }
  std::cout << std::endl;
  std::cout << "CLASS 1 After" << std::endl;
  for (int i = NCLS_start + 5 * 64; i < NCLS_start + 6 * 64; i++) {
    std::cout << ipos << ":" << mCounters[i] << " ";
    ipos++;
  }
  std::cout << std::endl;
  std::cout << " REST:" << std::endl;
  for (int i = NCLS_start + 6 * 64; i < mCounters.size(); i++) {
    if ((ipos % 10) == 0) {
      std::cout << std::endl;
      std::cout << ipos << ":";
    }
    std::cout << mCounters[i] << " ";
  }
}
