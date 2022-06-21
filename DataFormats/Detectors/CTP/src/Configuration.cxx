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

/// \file Configuration.cxx
/// \author Roman Lietava

#include "DataFormatsCTP/Configuration.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include <iostream>
#include <sstream>
#include <regex>
#include "CommonUtils/StringUtils.h"
#include "FairLogger.h"

using namespace o2::ctp;
//
const std::map<std::string, std::string> CTPConfiguration::detName2LTG = {{"FV0", "1"}, {"FT0", "2"}, {"FDD", "3"}, {"ITS", "4"}, {"TOF", "5"}, {"MFT", "6"}, {"TPC", "7"}, {"MCH", "8"}, {"MID", "9"}, {"TST", "10"}, {"TRD", "13"}, {"HMP", "14"}, {"ZDC", "15"}, {"PHS", "16"}, {"EMC", "17"}, {"CPV", "18"}};
//
bool CTPConfiguration::isDetector(const o2::detectors::DetID& det)
{
  bool isdet = det.getID() >= det.getNDetectors();
  isdet |= det.getID() < 0;
  if (isdet) {
    LOG(error) << " Detector does not exist: " << det.getID();
    return false;
  }
  return true;
}
void CTPConfiguration::capitaliseString(std::string& str)
{
  for (auto& c : str) {
    c = std::toupper(c);
  }
}
//
void BCMask::printStream(std::ostream& stream) const
{
  stream << "CTP BC mask:" << name << ":" << mask << std::endl;
  /// <<  ":" << BCmask << std::endl;
}
//
const std::set<std::string> CTPGenerator::Generators = {"bcd1m", "bcd2m", "bcd10", "bcd20", "rnd1m", "rnd2m", "rnd10", "rnd20"};
void CTPGenerator::printStream(std::ostream& stream) const
{
  stream << "CTP generator:" << name << " frequency:" << frequency << std::endl;
}
//
void CTPInput::printStream(std::ostream& stream) const
{
  stream << "CTP Input:" << name << " Detector:" << getInputDetName() << " Level:" << level << " Hardware mask:0x" << std::hex << inputMask << std::dec << std::endl;
}
//
std::uint64_t CTPDescriptor::getInputsMask() const
{
  uint64_t mask = 0;
  for (const auto& inp : inputs) {
    mask |= inp->inputMask;
  }
  return mask;
}
void CTPDescriptor::printStream(std::ostream& stream) const
{
  stream << "CTP Descriptor:" << name << " Inputs:";
  for (const auto& inp : inputs) {
    stream << inp->name << " ";
  }
  stream << std::endl;
}
//
void CTPDetector::printStream(std::ostream& stream) const
{
  stream << "CTP Detector:" << getName() << " HBaccepted:" << HBaccepted;
  stream << " Mode:" << mode << " FErst:" << ferst << std::endl;
}

void CTPCluster::printStream(std::ostream& stream) const
{
  stream << "CTP Cluster:" << name << " " << getClusterDetNames();
  stream << " det mask:0b" << std::hex << maskCluster << " " << std::dec;
  stream << " clust index:" << hwMask;
  stream << std::endl;
}
//
void CTPClass::printStream(std::ostream& stream) const
{
  stream << "CTP Class:" << name << " Hardware mask:" << classMask << " Cluster index:" << clusterIndex << " Desc index:" << descriptorIndex;
  if (descriptor != nullptr) {
    stream << " Descriptor:" << descriptor->name;
  }
  if (cluster != nullptr) {
    stream << " Cluster:" << cluster->name;
  }
  stream << std::endl;
}
/// CTP configuration
/// Assuming Run2 format + LTG
int CTPConfiguration::loadConfigurationRun3(const std::string& ctpconfiguration)
{
  LOG(info) << "Loading CTP configuration.";
  mConfigString = ctpconfiguration;
  std::istringstream iss(ctpconfiguration);
  int ret = 0;
  int level = MASKS;
  std::string line;
  while (std::getline(iss, line)) {
    o2::utils::Str::trim(line);
    if ((ret = processConfigurationLineRun3(line, level)) != 0) {
      return ret;
    }
  }
  return ret;
  return 0;
}
int CTPConfiguration::processConfigurationLineRun3(std::string& line, int& level)
{
  LOG(info) << "Processing line";
  LOG(info) << "line:" << line << " lev:" << level;
  if (line.size() == 0) {
    return 0;
  }
  if (line.at(0) == '#') {
    return 0;
  }
  //
  std::vector<std::string> tokens = o2::utils::Str::tokenize(line, ' ');
  size_t ntokens = tokens.size();
  if (ntokens == 0) {
    LOG(warning) << "# of tokens zero in line:" << line;
    return 0;
  }
  size_t first;
  if ((first = line.find("run")) != std::string::npos) {
    level = RUN;
  } else if (CTPGenerator::Generators.count(tokens[0])) {
    if (level != CLASS) {
      level = GENS;
    }
    level = GENS;
  } else if ((first = line.find("bcm")) != std::string::npos) {
    if (level == MASKS) {
      level = MASKS;
    }
  } else if ((first = line.find("LTG")) != std::string::npos) {
    level = LTG;
  } else if ((first = line.find("cluster")) != std::string::npos) {
    level = CLUSTER;
  } else {
    bool knownlevels = ((level == LTGitems) || (level == CLASS));
    if (knownlevels == false) {
      level = UNKNOWN;
    }
  }
  LOG(info) << "Level:" << level;
  switch (level) {
    case RUN: {
      mRunNumber = std::stoul(tokens[1]);
      level = MASKS;
      break;
    }
    case MASKS: {
      BCMask bcmask;
      if (tokens.size() < 3) {
        LOG(error) << "Wrong bc mask:" << line;
        break;
      }
      bcmask.name = tokens[1];
      for (int i = 2; i < tokens.size(); i++) {
        bcmask.mask = tokens[i] = " ";
      }
      bool coded = tokens[2].find("L") != std::string::npos;
      coded |= tokens[2].find("H") != std::string::npos;
      std::cout << "coded:" << coded << std::endl;
      if (coded) {
        // jusko notation
      } else {
        // list of integers
        for (int i = 2; i < ntokens; i++) {
          uint32_t bc;
          try {
            bc = std::stoull(tokens[i]);
          } catch (...) {
            LOG(info) << "mask syntax:" << tokens[i];
            continue;
          }
          bcmask.BCmask.set(bc, 1);
        }
      }
      mBCMasks.push_back(bcmask);
      LOG(info) << "BC mask added:" << bcmask.name;
      break;
    }
    case GENS: {
      CTPGenerator gen;
      gen.name = tokens[0];
      gen.frequency = tokens[1];
      LOG(info) << "Gen added:" << line;
      break;
    }
    case LTG: {
      CTPDetector ctpdet;
      std::string detname = tokens[1];
      capitaliseString(detname);
      o2::detectors::DetID det(detname.c_str());
      if (isDetector(det)) {
        ctpdet.detID = det.getID();
        LOG(info) << "Detector found:" << det.getID() << " " << detname;
      } else {
        LOG(info) << "Unknown detectors:" << line;
      }
      mDetectors.push_back(ctpdet);
      level = LTGitems;
      break;
    }
    case LTGitems: {
      if (ntokens == 1) {
        mDetectors.back().mode = tokens[0];
      }
      LOG(info) << "LTGitem:" << line;
      break;
    }
    case CLUSTER: {
      CTPCluster cluster;
      try {
        cluster.hwMask = std::stoull(tokens[0]);
      } catch (...) {
        LOG(info) << "Cluster syntax error:" << line;
        return level;
      }
      LOG(info) << "Cluster:" << line;
      cluster.name = tokens[2];
      o2::detectors::DetID::mask_t mask;
      for (int item = 3; item < ntokens; item++) {
        std::string detname = tokens[item];
        capitaliseString(detname);
        // LOG(info) << "Detector:" << detname;
        o2::detectors::DetID det(detname.c_str());
        isDetector(det);
        mask |= det.getMask();
      }
      cluster.maskCluster = mask;
      mClusters.push_back(cluster);
      level = CLASS;
      // LOG(info) << "Cluster done:" << cluster.name << std::endl;
      break;
    }
    case CLASS: {
      // add to the last cluster
      if (tokens.size() < 3) {
        LOG(error) << "CTPClass: less than 3 items in class:" << line;
        break;
      }
      uint64_t index;
      try {
        index = std::stoull(tokens[0]);
      } catch (...) {
        LOG(info) << "Class syntax error:" << line;
        return level;
      }
      LOG(info) << "Class:" << line;
      CTPClass cls;
      cls.classMask = 1ull << index;
      cls.name = tokens[1];
      cls.clusterIndex = mClusters.size() - 1;
      // LOG(info) << "point:" << cls.cluster << " " << &mClusters.front();
      // descriptor
      CTPDescriptor* desc = new CTPDescriptor;
      desc->name = tokens[2];
      mDescriptors.push_back(*desc);
      cls.descriptorIndex = mDescriptors.size() - 1;
      //
      mCTPClasses.push_back(cls);
      break;
    }
    default: {
      LOG(info) << "unknown line:" << line;
    }
  }
  for (auto& cls : mCTPClasses) {
    cls.cluster = &mClusters[cls.clusterIndex];
  }
  return 0;
}
void CTPConfiguration::printStream(std::ostream& stream) const
{
  stream << "Configuration:" << mName << "\n Version:" << mVersion << std::endl;
  stream << "Run:" << mRunNumber << " cfg name:" << mName;
  stream << "CTP BC  masks:" << std::endl;
  for (const auto& i : mBCMasks) {
    i.printStream(stream);
  }
  stream << "CTP inputs:" << std::endl;
  for (const auto& i : mInputs) {
    i.printStream(stream);
  }
  stream << "CTP descriptors:" << std::endl;
  for (const auto& i : mDescriptors) {
    i.printStream(stream);
  }
  stream << "CTP detectors:" << mDetectors.size() << std::endl;
  for (const auto& i : mDetectors) {
    i.printStream(stream);
  }
  stream << "CTP clusters:" << std::endl;
  for (const auto& i : mClusters) {
    i.printStream(stream);
  }
  stream << "CTP classes:" << std::endl;
  for (const auto& i : mCTPClasses) {
    i.printStream(stream);
  }
}
uint64_t CTPConfiguration::getInputMask(const std::string& name)
{
  for (auto const& inp : mInputs) {
    if (inp.name == name) {
      return inp.inputMask;
    }
  }
  return 0;
}
bool CTPConfiguration::isMaskInInputs(const uint64_t& mask) const
{
  for (auto const& inp : mInputs) {
    if (inp.inputMask == mask) {
      return true;
    }
  }
  return false;
}
bool CTPConfiguration::isBCMaskInConfig(const std::string maskname) const
{
  for (auto& bcm : mBCMasks) {
    if (bcm.name == maskname) {
      return true;
    }
  }
  return false;
}
CTPInput* CTPConfiguration::isInputInConfig(const std::string inpname)
{
  for (auto& inp : mInputs) {
    if (inp.name == inpname) {
      return &inp;
    }
  }
  return nullptr;
}
uint64_t CTPConfiguration::getDecrtiptorInputsMask(const std::string& name) const
{
  for (auto const& desc : mDescriptors) {
    if (desc.name == name) {
      return desc.getInputsMask();
    }
  }
  return 0xffffffff;
}
std::map<o2::detectors::DetID::ID, std::vector<CTPInput>> CTPConfiguration::getDet2InputMap()
{
  std::map<o2::detectors::DetID::ID, std::vector<CTPInput>> det2inp;
  for (auto const& inp : mInputs) {
    det2inp[inp.detID].push_back(inp);
  }
  return det2inp;
}
uint64_t CTPConfiguration::getTriggerClassMask() const
{
  uint64_t clsmask = 0;
  for (auto const& cls : mCTPClasses) {
    clsmask |= cls.classMask;
  }
  return clsmask;
}
std::vector<int> CTPConfiguration::getTriggerClassList() const
{
  uint64_t clsmask = getTriggerClassMask();
  std::vector<int> classlist;
  for (int i = 0; i < 64; i++) {
    if ((1ull << i) & clsmask) {
      classlist.push_back(i);
    }
  }
  return classlist;
}
std::vector<std::string> CTPConfiguration::getDetectorList() const
{
  std::vector<std::string> detlist;
  for (auto const& det : mDetectors) {
    std::string sdet(det.getName());
    detlist.push_back(sdet);
  }
  return detlist;
}
o2::detectors::DetID::mask_t CTPConfiguration::getDetectorMask() const
{
  o2::detectors::DetID::mask_t mask = 0;
  for (auto const& det : mDetectors) {
    mask |= det.getMask();
  }
  return mask;
}
//===============================================
//
void CTPRunManager::init()
{
  for (auto r : mActiveRuns) {
    r = nullptr;
  }
  loadScalerNames();
  LOG(info) << "CCDB host:" << mCCDBHost;
  LOG(info) << "CTP QC:" << mQC;
  LOG(info) << "CTPRunManager initialised.";
}
int CTPRunManager::startRun(const std::string& cfg)
{
  LOG(info) << "Starting run: " << cfg;
  const auto now = std::chrono::system_clock::now();
  const long timeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  CTPActiveRun* activerun = new CTPActiveRun;
  activerun->timeStart = timeStamp;
  activerun->cfg.loadConfigurationRun3(cfg);
  activerun->cfg.printStream(std::cout);
  //
  activerun->scalers.setRunNumber(activerun->cfg.getRunNumber());
  activerun->scalers.setClassMask(activerun->cfg.getTriggerClassMask());
  o2::detectors::DetID::mask_t detmask = activerun->cfg.getDetectorMask();
  activerun->scalers.setDetectorMask(detmask);
  //
  mRunInStart = activerun;
  saveRunConfigToCCDB(&activerun->cfg, timeStamp);
  return 0;
}
int CTPRunManager::stopRun(uint32_t irun)
{
  LOG(info) << "Stopping run index: " << irun;
  if (mActiveRuns[irun] == nullptr) {
    LOG(error) << "No config for run index:" << irun;
    return 1;
  }
  const auto now = std::chrono::system_clock::now();
  const long timeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  mActiveRuns[irun]->timeStop = timeStamp;
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
      std::string countername = detname + CTPConfiguration::detName2LTG.at(detname) + "_PH";
      uint32_t detcount = mCounters[mScalerName2Position[countername]];
      scalrec.scalersDets.push_back(detcount);
      LOG(info) << "Scaler for detector:" << countername << ":" << detcount;
    }
  }
  //
  scalrec.intRecord.orbit = mCounters[mScalerName2Position[orb]];
  scalrec.intRecord.bc = 0;
  mActiveRuns[irun]->scalers.addScalerRacordRaw(scalrec);
  LOG(info) << "Adding scalers for orbit:" << scalrec.intRecord.orbit;
  // scalrec.printStream(std::cout);
  // printCounters();
  return 0;
}
int CTPRunManager::processMessage(std::string& topic, const std::string& message)
{
  if (mQC == 1) {
    LOG(info) << "processMessage: skipping, QC=1";
    return 1;
  }
  LOG(info) << "Processing message with topic:" << topic;
  std::string firstcounters;
  if (topic.find("ctpconfig") != std::string::npos) {
    LOG(info) << "ctpcfg received";
    startRun(message);
    mCtpcfg = 1;
    return 0;
  }
  if (topic.find("sox") != std::string::npos) {
    // get config
    size_t irun = message.find("run");
    if (irun == std::string::npos) {
      LOG(error) << "run keyword not found in SOX:\n"
                 << message;
      return 1;
    }
    LOG(info) << "SOX received, Run keyword position:" << irun;
    if (mCtpcfg == 0) {
      std::string cfg = message.substr(irun, message.size() - irun);
      LOG(info) << "Config:" << cfg;
      startRun(cfg);
    } else {
      mCtpcfg = 0;
    }
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
    LOG(error) << "Scalers size wrong:" << tokens.size() << " expected:" << CTPRunScalers::NCOUNTERS + 1;
    return 1;
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
      mActiveRunNumbers[i] = mCounters[i];
      if (mRunInStart == nullptr) {
        LOG(error) << "Internal error in processMessage: nullptr != 0 expected";
      }
      mActiveRuns[i] = mRunInStart;
      mRunInStart = nullptr;
      addScalers(i, tt);
    } else if ((mCounters[i] == 0) && (mActiveRunNumbers[i] != 0)) {
      if (mEOX != 1) {
        LOG(error) << "Internal error in processMessage: mEOX != 1 expected 0: mEOX:" << mEOX;
      }
      LOG(info) << "Run stopped:" << mActiveRunNumbers[i];
      addScalers(i, tt);
      mActiveRunNumbers[i] = 0;
      mEOX = 0;
      stopRun(i);
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
    ctpconfigdb->printStream(std::cout);
  }
  return *ctpconfigdb;
}
CTPRunScalers CTPRunManager::getScalersFromCCDB(long timestamp, std::string run)
{
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCCDBHost);
  map<string, string> metadata; // can be empty
  metadata["runNumber"] = run;
  auto ctpscalers = mgr.getSpecific<CTPRunScalers>(mCCDBPathCTPScalers, timestamp, metadata);
  if (ctpscalers == nullptr) {
    LOG(info) << "CTPRunScalers not in database, timestamp:" << timestamp;
  } else {
    ctpscalers->printStream(std::cout);
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
