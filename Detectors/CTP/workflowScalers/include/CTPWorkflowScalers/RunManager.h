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
#include "CTPWorkflowScalers/ctpCCDBManager.h"
#include "DataFormatsCTP/Configuration.h"
#include "BookkeepingApi/BkpClientFactory.h"
#include "BookkeepingApi/BkpClient.h"
using namespace o2::bkp::api;
namespace o2
{
namespace ctp
{
typedef std::map<uint32_t, std::array<uint32_t, 6>> counters_t;
typedef std::map<uint32_t, std::array<uint64_t, 6>> counters64_t;
struct CTPActiveRun {
  CTPActiveRun() = default;
  long timeStart;
  long timeStop;
  CTPConfiguration cfg;
  CTPRunScalers scalers;
  void initBK();
  int send2BK(std::unique_ptr<BkpClient>& BKClient, size_t ts, bool start);
  //
  counters_t cnts0;     // first counters in run
  counters_t cntslast0; // last minus one read counters needed for overflow correction
  counters_t cntslast;  // last read counters
  counters64_t overflows;
  // QC
  int qcwpcount = 0;
};
class CTPRunManager : public ctpCCDBManager
{
 public:
  CTPRunManager() = default;
  void init();
  int loadRun(const std::string& cfg);
  int startRun(const std::string& cfg);
  int stopRun(uint32_t irun, long timeStamp);
  int addScalers(uint32_t irun, std::time_t time, bool start = 0);
  int processMessage(std::string& topic, const std::string& message);
  void printActiveRuns() const;
  int loadScalerNames();
  int getNRuns();
  void setBKHost(std::string host) { mBKHost = host; };
  void setQCWritePeriod(int period) { mQCWritePeriod = period;};
  uint64_t checkOverflow(uint32_t lcnt0, uint32_t lcnt1, uint64_t lcntcor);
  void printCounters();

 private:
  /// Database constants
  std::string mBKHost = "";
  std::array<CTPActiveRun*, NRUNS> mActiveRuns;
  std::array<std::uint32_t, NRUNS> mActiveRunNumbers;
  std::array<uint32_t, CTPRunScalers::NCOUNTERS> mCounters;
  std::map<std::string, uint32_t> mScalerName2Position;
  std::map<uint32_t, CTPActiveRun*> mRunsLoaded;
  std::unique_ptr<BkpClient> mBKClient;
  int mEOX = 0; // redundancy check
  int mNew = 1; // 1 - no CCDB: used for QC
  int mQCWritePeriod = 3; // Time in 10secs between two writes to QCCD 
  ClassDefNV(CTPRunManager, 7);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_RUNMANAGER_H_
