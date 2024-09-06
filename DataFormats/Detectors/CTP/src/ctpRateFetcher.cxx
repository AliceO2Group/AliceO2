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

#include "DataFormatsCTP/ctpRateFetcher.h"

#include <map>
#include <vector>

#include "CommonConstants/LHCConstants.h"
#include "DataFormatsCTP/Configuration.h"
#include "DataFormatsCTP/Scalers.h"
#include "CCDB/BasicCCDBManager.h"

using namespace o2::ctp;
double ctpRateFetcher::fetch(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, std::string sourceName)
{
  auto triggerRate = fetchNoPuCorr(ccdb, timeStamp, runNumber, sourceName);
  if (triggerRate >= 0) {
    return pileUpCorrection(triggerRate);
  }
  return -1;
}
double ctpRateFetcher::fetchNoPuCorr(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, std::string sourceName)
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
      double_t ret = fetchCTPratesClassesNoPuCorr(timeStamp, "CMTVX-B-NOPF");
      if (ret == -2.) {
        LOG(info) << "Trying different class";
        ret = fetchCTPratesClassesNoPuCorr(timeStamp, "CMTVX-NONE");
        if (ret < 0) {
          LOG(fatal) << "None of the classes used for lumi found";
        }
      }
      return ret;
    }
  }
  LOG(error) << "CTP rate for " << sourceName << " not available";
  return -1.;
}
void ctpRateFetcher::updateScalers(ctp::CTPRunScalers* scalers)
{
  mScalers = scalers;
  mScalers->convertRawToO2();
}
//
double ctpRateFetcher::fetchCTPratesClasses(uint64_t timeStamp, const std::string& className, int inputType)
{
  auto triggerRate = fetchCTPratesClassesNoPuCorr(timeStamp, className, inputType);
  if (triggerRate >= 0) {
    return pileUpCorrection(triggerRate);
  }
  return -1;
}
double ctpRateFetcher::fetchCTPratesClassesNoPuCorr(uint64_t timeStamp, const std::string& className, int inputType)
{
  std::vector<ctp::CTPClass> ctpcls = mConfig->getCTPClasses();
  std::vector<int> clslist = mConfig->getTriggerClassList();
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
  auto rate{mScalers->getRateGivenT(timeStamp * 1.e-3, classIndex, inputType)};
  return rate.second;
}
double ctpRateFetcher::fetchCTPratesInputs(uint64_t timeStamp, int input)
{
  std::vector<ctp::CTPScalerRecordO2> recs = mScalers->getScalerRecordO2();
  if (recs[0].scalersInps.size() == 48) {
    return pileUpCorrection(mScalers->getRateGivenT(timeStamp * 1.e-3, input, 7).second);
  } else {
    LOG(error) << "Inputs not available";
    return -1.;
  }
}
double ctpRateFetcher::fetchCTPratesInputsNoPuCorr(uint64_t timeStamp, int input)
{
  std::vector<ctp::CTPScalerRecordO2> recs = mScalers->getScalerRecordO2();
  if (recs[0].scalersInps.size() == 48) {
    return mScalers->getRateGivenT(timeStamp * 1.e-3, input, 7).second;
  } else {
    LOG(error) << "Inputs not available";
    return -1.;
  }
}

double ctpRateFetcher::pileUpCorrection(double triggerRate)
{
  if (mLHCIFdata == nullptr) {
    LOG(fatal) << "No filling" << std::endl;
  }
  auto bfilling = mLHCIFdata->getBunchFilling();
  std::vector<int> bcs = bfilling.getFilledBCs();
  double nbc = bcs.size();
  double nTriggersPerFilledBC = triggerRate / nbc / constants::lhc::LHCRevFreq;
  double mu = -std::log(1 - nTriggersPerFilledBC);
  return mu * nbc * constants::lhc::LHCRevFreq;
}

void ctpRateFetcher::setupRun(int runNumber, o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, bool initScalers)
{
  if (runNumber == mRunNumber) {
    return;
  }
  mRunNumber = runNumber;
  LOG(debug) << "Setting up CTP scalers for run " << mRunNumber;
  std::map<string, string> metadata;
  mLHCIFdata = ccdb->getSpecific<parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", timeStamp, metadata);
  if (mLHCIFdata == nullptr) {
    LOG(fatal) << "GRPLHCIFData not in database, timestamp:" << timeStamp;
  }
  metadata["runNumber"] = std::to_string(mRunNumber);
  mConfig = ccdb->getSpecific<ctp::CTPConfiguration>("CTP/Config/Config", timeStamp, metadata);
  if (mConfig == nullptr) {
    LOG(fatal) << "CTPRunConfig not in database, timestamp:" << timeStamp;
  }
  if (initScalers) {
    mScalers = ccdb->getSpecific<ctp::CTPRunScalers>("CTP/Calib/Scalers", timeStamp, metadata);
    if (mScalers == nullptr) {
      LOG(fatal) << "CTPRunScalers not in database, timestamp:" << timeStamp;
    }
    mScalers->convertRawToO2();
  }
}
