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

#include "DataFormatsCTP/CTPRateFetcher.h"
#include "CCDB/BasicCCDBManager.h"

#include <map>
#include <vector>
#include "CommonConstants/LHCConstants.h"

using namespace o2::ctp;
double CTPRateFetcher::fetch(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, std::string sourceName)
{
  auto triggerRate = fetchNoPuCorr(ccdb, timeStamp, runNumber, sourceName);
  if (triggerRate >= 0) {
    return pileUpCorrection(triggerRate);
  }
  return -1;
}
double CTPRateFetcher::fetchNoPuCorr(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, std::string sourceName)
{
  setupRun(runNumber, ccdb, timeStamp, 1);
  if (sourceName.find("ZNC") != std::string::npos) {
    if (runNumber < 544448) {
      return fetchCTPratesInputsNoPuCorr(timeStamp, 25) / (sourceName.find("hadronic") != std::string::npos ? 28. : 1.);
    } else {
      return fetchCTPratesClassesNoPuCorr(timeStamp, "C1ZNC-B-NOPF-CRU", 6) / (sourceName.find("hadronic") != std::string::npos ? 28. : 1.);
    }
  } else if (sourceName == "T0CE") {
    return fetchCTPratesClassesNoPuCorr(timeStamp, "CMTVXTCE-B-NOPF");
  } else if (sourceName == "T0SC") {
    return fetchCTPratesClassesNoPuCorr(timeStamp, "CMTVXTSC-B-NOPF");
  } else if (sourceName == "T0VTX") {
    if (runNumber < 534202) {
      return fetchCTPratesClassesNoPuCorr(timeStamp, "minbias_TVX_L0", 3); // 2022
    } else {
      double ret = fetchCTPratesClassesNoPuCorr(timeStamp, "CMTVX-B-NOPF");
      if (ret == -2.) {
        LOG(info) << "Trying different class";
        ret = fetchCTPratesClassesNoPuCorr(timeStamp, "CMTVX-NONE");
        if (ret < 0) {
          LOG(error) << "None of the classes used for lumi found";
          return -1.;
        }
      }
      return ret;
    }
  }
  LOG(error) << "CTP rate for " << sourceName << " not available";
  return -1.;
}
void CTPRateFetcher::updateScalers(ctp::CTPRunScalers& scalers)
{
  mScalers = scalers;
  mScalers.convertRawToO2();
}
//
int CTPRateFetcher::getRates(std::array<double, 3>& rates, o2::ccdb::BasicCCDBManager* ccdb, int runNumber, const std::string sourceName) // rates at start,stop and middle of the run
{
  setupRun(runNumber, ccdb, 0, 1);
  mOrbit = 1;
  mOutsideLimits = 1;
  auto orbitlimits = mScalers.getOrbitLimit();
  // std::cout << "1st orbit:" << orbitlimits.first << " last:" << orbitlimits.second << " Middle:" << (orbitlimits.first + orbitlimits.second)/2 << std::endl;
  double rate0 = fetch(ccdb, orbitlimits.first, mRunNumber, sourceName);
  double rateLast = fetch(ccdb, orbitlimits.second, mRunNumber, sourceName);
  double rateM = fetch(ccdb, (orbitlimits.first + orbitlimits.second) / 2, mRunNumber, sourceName);
  // std::cout << rate0 << " " << rateLast << " " << rateM << std::endl;
  rates[0] = rate0;
  rates[1] = rateLast;
  rates[2] = rateM;
  return 0;
}
double CTPRateFetcher::getLumiNoPuCorr(const std::string& classname, int type)
{
  if (classname == "zncinp") {
    return mScalers.getLumiNoPuCorr(26, 7);
  }
  std::vector<ctp::CTPClass>& ctpcls = mConfig.getCTPClasses();
  std::vector<int> clslist = mConfig.getTriggerClassList();
  int classIndex = -1;
  for (size_t i = 0; i < clslist.size(); i++) {
    if (ctpcls[i].name.find(classname) != std::string::npos) {
      classIndex = i;
      break;
    }
  }
  if (classIndex == -1) {
    LOG(warn) << "Trigger class " << classname << " not found in CTPConfiguration";
    return -1;
  }
  return mScalers.getLumiNoPuCorr(classIndex, type);
}
double CTPRateFetcher::getLumiWPuCorr(const std::string& classname, int type)
{
  std::vector<std::pair<double, double>> scals;
  if (classname == "zncinp") {
    scals = mScalers.getRatesForIndex(26, 7);
  } else {
    std::vector<ctp::CTPClass>& ctpcls = mConfig.getCTPClasses();
    std::vector<int> clslist = mConfig.getTriggerClassList();
    int classIndex = -1;
    for (size_t i = 0; i < clslist.size(); i++) {
      if (ctpcls[i].name.find(classname) != std::string::npos) {
        classIndex = i;
        break;
      }
    }
    if (classIndex == -1) {
      LOG(warn) << "Trigger class " << classname << " not found in CTPConfiguration";
      return -1;
    }
    scals = mScalers.getRatesForIndex(classIndex, type);
  }
  double lumi = 0;
  for (auto const& ss : scals) {
    // std::cout << ss.first << " " << ss.second << " " << pileUpCorrection(ss.first/ss.second) << std::endl;
    lumi += pileUpCorrection(ss.first / ss.second) * ss.second;
  }
  return lumi;
}
double CTPRateFetcher::getLumi(const std::string& classname, int type, int puCorr)
{
  if (puCorr) {
    return getLumiWPuCorr(classname, type);
  } else {
    return getLumiNoPuCorr(classname, type);
  }
}

double CTPRateFetcher::getLumi(o2::ccdb::BasicCCDBManager* ccdb, int runNumber, const std::string sourceName, int puCorr)
{
  // setupRun(runNumber, ccdb, timeStamp, 1);
  if (sourceName.find("ZNC") != std::string::npos) {
    if (runNumber < 544448) {
      return getLumi("zncinp", 1, puCorr) / (sourceName.find("hadronic") != std::string::npos ? 28. : 1.);
    } else {
      return getLumi("C1ZNC-B-NOPF-CRU", 6, puCorr) / (sourceName.find("hadronic") != std::string::npos ? 28. : 1.);
    }
  } else if (sourceName == "T0CE") {
    return getLumi("CMTVXTCE-B-NOPF", 1, puCorr);
  } else if (sourceName == "T0SC") {
    return getLumi("CMTVXTSC-B-NOPF", 1, puCorr);
  } else if (sourceName == "T0VTX") {
    if (runNumber < 534202) {
      return getLumi("minbias_TVX_L0", 3, puCorr); // 2022
    } else {
      double ret = getLumi("CMTVX-B-NOPF", 1, puCorr);
      if (ret == -1.) {
        LOG(info) << "Trying different class";
        ret = getLumi("CMTVX-NONE", 1, puCorr);
        if (ret < 0) {
          LOG(fatal) << "None of the classes used for lumi found";
        }
      }
      return ret;
    }
  }
  LOG(error) << "CTP Lumi for " << sourceName << " not available";
  return 0;
}
//
double CTPRateFetcher::fetchCTPratesClasses(uint64_t timeStamp, const std::string& className, int inputType)
{
  auto triggerRate = fetchCTPratesClassesNoPuCorr(timeStamp, className, inputType);
  if (triggerRate >= 0) {
    return pileUpCorrection(triggerRate);
  }
  return -1;
}
double CTPRateFetcher::fetchCTPratesClassesNoPuCorr(uint64_t timeStamp, const std::string& className, int inputType)
{
  std::vector<ctp::CTPClass>& ctpcls = mConfig.getCTPClasses();
  std::vector<int> clslist = mConfig.getTriggerClassList();
  int classIndex = -1;
  for (size_t i = 0; i < clslist.size(); i++) {
    if (ctpcls[i].name.find(className) != std::string::npos) {
      classIndex = i;
      break;
    }
  }
  if (classIndex == -1) {
    LOG(warn) << "Trigger class " << className << " not found in CTPConfiguration";
    return -2.;
  }
  if (mOrbit) {
    auto rate{mScalers.getRate((uint32_t)timeStamp, classIndex, inputType, mOutsideLimits)};
    return rate.second;
  } else {
    auto rate{mScalers.getRateGivenT(timeStamp * 1.e-3, classIndex, inputType, mOutsideLimits)};
    return rate.second;
  }
}
double CTPRateFetcher::fetchCTPratesInputs(uint64_t timeStamp, int input)
{
  std::vector<ctp::CTPScalerRecordO2>& recs = mScalers.getScalerRecordO2();
  if (recs[0].scalersInps.size() == 48) {
    if (mOrbit) {
      return pileUpCorrection(mScalers.getRate((uint32_t)timeStamp, input, 7, mOutsideLimits).second);
    } else {
      return pileUpCorrection(mScalers.getRateGivenT(timeStamp * 1.e-3, input, 7, mOutsideLimits).second);
    }
  } else {
    LOG(error) << "Inputs not available";
    return -1.;
  }
}
double CTPRateFetcher::fetchCTPratesInputsNoPuCorr(uint64_t timeStamp, int input)
{
  std::vector<ctp::CTPScalerRecordO2>& recs = mScalers.getScalerRecordO2();
  if (recs[0].scalersInps.size() == 48) {
    if (mOrbit) {
      return mScalers.getRate((uint32_t)timeStamp, input, 7, mOutsideLimits).second;
    } else {
      return mScalers.getRateGivenT(timeStamp * 1.e-3, input, 7, mOutsideLimits).second; // qc flag implemented only for time
    }
  } else {
    LOG(error) << "Inputs not available";
    return -1.;
  }
}

double CTPRateFetcher::pileUpCorrection(double triggerRate)
{
  if (mLHCIFdata.getFillNumber() == 0) {
    LOG(fatal) << "No filling" << std::endl;
  }
  auto bfilling = mLHCIFdata.getBunchFilling();
  std::vector<int> bcs = bfilling.getFilledBCs();
  double nbc = bcs.size();
  double nTriggersPerFilledBC = triggerRate / nbc / constants::lhc::LHCRevFreq;
  double mu = -std::log(1 - nTriggersPerFilledBC);
  return mu * nbc * constants::lhc::LHCRevFreq;
}

void CTPRateFetcher::setupRun(int runNumber, o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, bool initScalers)
{
  if (runNumber == mRunNumber) {
    return;
  }
  mRunNumber = runNumber;
  LOG(info) << "Setting up CTP scalers for run " << mRunNumber << " and timestamp : " << timeStamp;
  auto ptrLHCIFdata = ccdb->getSpecific<parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", timeStamp);
  if (ptrLHCIFdata == nullptr) {
    LOG(error) << "GRPLHCIFData not in database, timestamp:" << timeStamp;
    return;
  }
  mLHCIFdata = *ptrLHCIFdata;
  std::map<std::string, std::string> metadata;
  metadata["runNumber"] = std::to_string(mRunNumber);
  auto ptrConfig = ccdb->getSpecific<ctp::CTPConfiguration>("CTP/Config/Config", timeStamp, metadata);
  if (ptrConfig == nullptr) {
    LOG(error) << "CTPRunConfig not in database, timestamp:" << timeStamp;
    return;
  }
  mConfig = *ptrConfig;
  if (initScalers) {
    auto ptrScalers = ccdb->getSpecific<ctp::CTPRunScalers>("CTP/Calib/Scalers", timeStamp, metadata);
    if (ptrScalers) {
      mScalers = *ptrScalers;
      mScalers.convertRawToO2();
    } else {
      LOG(error) << "CTPRunScalers not in database, timestamp:" << timeStamp;
    }
  }
}
