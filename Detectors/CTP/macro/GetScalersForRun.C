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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <CCDB/BasicCCDBManager.h>
#include <DataFormatsCTP/Configuration.h>
#include "ctpRateFetcher.h"
#endif
using namespace o2::ctp;

void GetScalersForRun(int runNumber = 0, int fillN = 0, bool test = 1)
{
  if (test == 0) {
    return;
  }
  std::string mCCDBPathCTPScalers = "CTP/Calib/Scalers";
  std::string mCCDBPathCTPConfig = "CTP/Config/Config";
  // o2::ccdb::CcdbApi api;
  // api.init("http://alice-ccdb.cern.ch"); // alice-ccdb.cern.ch
  // api.init("http://ccdb-test.cern.ch:8080");
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  // ccdbMgr.setURL("http://ccdb-test.cern.ch:8080");
  auto soreor = ccdbMgr.getRunDuration(runNumber);
  uint64_t timeStamp = (soreor.second - soreor.first) / 2 + soreor.first;
  std::cout << "Timestamp:" << timeStamp << std::endl;
  //
  std::string sfill = std::to_string(fillN);
  std::map<string, string> metadata;
  metadata["fillNumber"] = sfill;
  auto lhcifdata = ccdbMgr.getSpecific<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", timeStamp);
  auto bfilling = lhcifdata->getBunchFilling();
  std::vector<int> bcs = bfilling.getFilledBCs();
  std::cout << "Number of interacting bc:" << bcs.size() << std::endl;
  //
  std::string srun = std::to_string(runNumber);
  metadata.clear(); // can be empty
  metadata["runNumber"] = srun;
  ccdbMgr.setURL("http://ccdb-test.cern.ch:8080");
  auto ctpscalers = ccdbMgr.getSpecific<CTPRunScalers>(mCCDBPathCTPScalers, timeStamp, metadata);
  if (ctpscalers == nullptr) {
    LOG(info) << "CTPRunScalers not in database, timestamp:" << timeStamp;
  }
  auto ctpcfg = ccdbMgr.getSpecific<CTPConfiguration>(mCCDBPathCTPConfig, timeStamp, metadata);
  if (ctpcfg == nullptr) {
    LOG(info) << "CTPRunConfig not in database, timestamp:" << timeStamp;
  }
  std::cout << "all good" << std::endl;
  ctpscalers->convertRawToO2();
  std::vector<CTPClass> ctpcls = ctpcfg->getCTPClasses();
  // std::vector<int> clslist = ctpcfg->getTriggerClassList();
  std::vector<uint32_t> clslist = ctpscalers->getClassIndexes();
  std::map<int, int> clsIndexToScaler;
  std::cout << "Classes:";
  int i = 0;
  for (auto const& cls : clslist) {
    std::cout << cls << " ";
    clsIndexToScaler[cls] = i;
    i++;
  }
  std::cout << std::endl;
  int tsc = 255;
  int tce = 255;
  int vch = 255;
  int iznc = 255;
  for (auto const& cls : ctpcls) {
    if (cls.name.find("CMTVXTSC-B-NOPF-CRU") != std::string::npos) {
      tsc = cls.getIndex();
      std::cout << cls.name << ":" << tsc << std::endl;
    }
    if (cls.name.find("CMTVXTCE-B-NOPF-CRU") != std::string::npos) {
      tce = cls.getIndex();
      std::cout << cls.name << ":" << tce << std::endl;
    }
    if (cls.name.find("CMTVXVCH-B-NOPF-CRU") != std::string::npos) {
      vch = cls.getIndex();
      std::cout << cls.name << ":" << vch << std::endl;
    }
    // if (cls.name.find("C1ZNC-B-NOPF-CRU") != std::string::npos) {
    if (cls.name.find("C1ZNC-B-NOPF") != std::string::npos) {
      iznc = cls.getIndex();
      std::cout << cls.name << ":" << iznc << std::endl;
    }
  }
  std::vector<CTPScalerRecordO2> recs = ctpscalers->getScalerRecordO2();
  if (recs[0].scalersInps.size() == 48) {
    std::cout << "ZNC:";
    int inp = 26;
    double_t nbc = bcs.size();
    double_t frev = 11245;
    double_t sigmaratio = 28.;
    double_t time0 = recs[0].epochTime;
    double_t timeL = recs[recs.size() - 1].epochTime;
    double_t Trun = timeL - time0;
    double_t integral = recs[recs.size() - 1].scalersInps[inp - 1] - recs[0].scalersInps[inp - 1];
    double_t rate = integral / Trun;
    double_t rat = integral / Trun / nbc / frev;
    double_t mu = -TMath::Log(1 - rat);
    double_t pp = 1 - mu / (TMath::Exp(mu) - 1);
    double_t ratepp = mu * nbc * frev;
    double_t integralpp = ratepp * Trun;
    std::cout << "Rate:" << rate / sigmaratio << " Integral:" << integral << " mu:" << mu << " Pileup prob:" << pp;
    std::cout << " Integralpp:" << integralpp << " Ratepp:" << ratepp / sigmaratio << std::endl;
    // ctpscalers->printInputRateAndIntegral(26);
  } else {
    std::cout << "Inputs not available" << std::endl;
  }
  //
  if (tsc != 255) {
    std::cout << "TSC:";
    ctpscalers->printClassBRateAndIntegral(clsIndexToScaler[tsc] + 1);
  }
  if (tce != 255) {
    std::cout << "TCE:";
    ctpscalers->printClassBRateAndIntegral(clsIndexToScaler[tce] + 1);
  }
  // std::cout << "TCE input:" << ctpscalers->printInputRateAndIntegral(5) << std::endl;;
  if (vch != 255) {
    std::cout << "VCH:";
    ctpscalers->printClassBRateAndIntegral(clsIndexToScaler[vch] + 1);
  }
  if (iznc != 255) {
    std::cout << "ZNC class:";
    // uint64_t integral = recs[recs.size() - 1].scalers[iznc].l1After - recs[0].scalers[iznc].l1After;
    auto zncrate = ctpscalers->getRateGivenT(0, iznc, 6);
    std::cout << "ZNC class rate:" << zncrate.first / 28. << std::endl;
  } else {
    std::cout << "ZNC class not available" << std::endl;
  }
  // ctpRateFetcher ctprate;
}
