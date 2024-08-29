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

#ifndef COMMON_CCDB_CTPRATEFETCHER_H_
#define COMMON_CCDB_CTPRATEFETCHER_H_

#include <string>

#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsParameters/GRPLHCIFData.h"

namespace o2
{
namespace ctp
{
class CTPRunScalers;
class CTPConfiguration;

class ctpRateFetcher
{
 public:
  ctpRateFetcher() = default;
  double fetch(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, std::string sourceName);
  double fetchNoPuCorr(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, std::string sourceName);

  void setManualCleanup(bool manualCleanup = true) { mManualCleanup = manualCleanup; }

 private:
  double fetchCTPratesInputs(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, int input);
  double fetchCTPratesClasses(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, const std::string& className, int inputType = 1);
  double fetchCTPratesInputsNoPuCorr(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, int input);
  double fetchCTPratesClassesNoPuCorr(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, const std::string& className, int inputType = 1);
  
  double pileUpCorrection(double rate);
  void setupRun(int runNumber, o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp);
  bool mManualCleanup = false;
  int mRunNumber = -1;
  ctp::CTPConfiguration* mConfig = nullptr;
  ctp::CTPRunScalers* mScalers = nullptr;
  o2::parameters::GRPLHCIFData* mLHCIFdata = nullptr;
};
} // namespace ctp
} // namespace o2

#endif // COMMON_CCDB_CTPRATEFETCHER_H_
