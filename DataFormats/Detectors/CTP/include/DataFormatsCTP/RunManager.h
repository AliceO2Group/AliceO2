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

/// \file RunManager.h
/// \brief Managing runs for config and scalers
/// \author Roman Lietava
#ifndef _CTP_RUNMANAGER_H_
#define _CTP_RUNMANAGER_H_
#include "DataFormatsCTP/Configuration.h"
namespace o2
{
namespace ctp
{
struct CTPActiveRun {
  CTPActiveRun() = default;
  long timeStart;
  long timeStop;
  CTPConfiguration cfg;
  CTPRunScalers scalers;
};
class CTPRunManager
{
 public:
  CTPRunManager() = default;
  void init();
  int loadRun(const std::string& cfg);
  int startRun(const std::string& cfg);
  int stopRun(uint32_t irun, long timeStamp);
  int addScalers(uint32_t irun, std::time_t time);
  int processMessage(std::string& topic, const std::string& message);
  void printActiveRuns() const;
  int saveRunScalersToCCDB(int i);
  int saveRunConfigToCCDB(CTPConfiguration* cfg, long timeStart);
  static CTPConfiguration getConfigFromCCDB(long timestamp, std::string run);
  CTPRunScalers getScalersFromCCDB(long timestamp, std::string, bool& ok);
  int loadScalerNames();
  // void setCCDBPathConfig(std::string path) { mCCDBPathCTPConfig = path;};
  void setCCDBPathScalers(std::string path) { mCCDBPathCTPScalers = path; };
  static void setCCDBHost(std::string host) { mCCDBHost = host; };
  void printCounters();

 private:
  /// Database constants
  // std::string mCCDBHost = "http://ccdb-test.cern.ch:8080";
  static std::string mCCDBHost;
  std::string mCCDBPathCTPScalers = "CTP/Calib/Scalers";
  std::array<CTPActiveRun*, NRUNS> mActiveRuns;
  std::array<std::uint32_t, NRUNS> mActiveRunNumbers;
  std::array<uint32_t, CTPRunScalers::NCOUNTERS> mCounters;
  std::map<std::string, uint32_t> mScalerName2Position;
  std::map<uint32_t, CTPActiveRun*> mRunsLoaded;
  int mEOX = 0; // redundancy check
  int mNew = 1; // 1 - no CCDB: used for QC
  ClassDefNV(CTPRunManager, 5);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_RUNMANAGER_H_
