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
#include "DataFormatsCTP/Configuration.h"
#include "DataFormatsCTP/Scalers.h"

namespace o2
{
namespace ctp
{

class CTPRateFetcher
{
 public:
  CTPRateFetcher() = default;
  double fetch(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, const std::string sourceName);
  double fetchNoPuCorr(o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, int runNumber, const std::string sourceName);
  void setupRun(int runNumber, o2::ccdb::BasicCCDBManager* ccdb, uint64_t timeStamp, bool initScalers);
  void updateScalers(ctp::CTPRunScalers& scalers);

 private:
  double fetchCTPratesInputs(uint64_t timeStamp, int input);
  double fetchCTPratesClasses(uint64_t timeStamp, const std::string& className, int inputType = 1);
  double fetchCTPratesInputsNoPuCorr(uint64_t timeStamp, int input);
  double fetchCTPratesClassesNoPuCorr(uint64_t timeStamp, const std::string& className, int inputType = 1);

  double pileUpCorrection(double rate);
  int mRunNumber = -1;
  o2::ctp::CTPConfiguration mConfig{};
  o2::ctp::CTPRunScalers mScalers{};
  o2::parameters::GRPLHCIFData mLHCIFdata{};
  ClassDefNV(CTPRateFetcher, 1);
};
} // namespace ctp
} // namespace o2

#endif // COMMON_CCDB_CTPRATEFETCHER_H_
