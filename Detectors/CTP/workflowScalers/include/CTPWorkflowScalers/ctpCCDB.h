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
#ifndef _CTP_CTPCCDB_H_
#define _CTP_CTPCCDB_H_
#include "DataFormatsCTP/Configuration.h"

namespace o2
{
namespace ctp
{
class ctpCCDB
{
 public:
  ctpCCDB() = default;
  int saveRunScalersToCCDB(CTPRunScalers& scalers, long timeStart, long timeStop);
  int saveRunConfigToCCDB(CTPConfiguration* cfg, long timeStart);
  static CTPConfiguration getConfigFromCCDB(long timestamp, std::string run, bool& ok);
  static CTPConfiguration getConfigFromCCDB(long timestamp, std::string run);
  CTPRunScalers getScalersFromCCDB(long timestamp, std::string, bool& ok);
  void setCCDBPathConfig(std::string path) { mCCDBPathCTPConfig = path; };
  void setCCDBPathScalers(std::string path) { mCCDBPathCTPScalers = path; };
  static void setCCDBHost(std::string host) { mCCDBHost = host; };

 protected:
  /// Database constants
  // std::string mCCDBHost = "http://ccdb-test.cern.ch:8080";
  static std::string mCCDBHost;
  std::string mCCDBPathCTPScalers = "CTP/Calib/Scalers";
  std::string mCCDBPathCTPConfig = "CTP/Config/Config";

  ClassDefNV(ctpCCDB, 0);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_CTPCCDB_H_
