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
class ctpCCDBManager
{
 public:
  ctpCCDBManager() = default;
  int saveRunScalersToCCDB(CTPRunScalers& scalers, long timeStart, long timeStop);
  int saveRunScalersToQCDB(CTPRunScalers& scalers, long timeStart, long timeStop);
  int saveRunConfigToCCDB(CTPConfiguration* cfg, long timeStart);
  int saveSoxOrbit(uint32_t runNumber, uint32_t soxOrbit, long timeStart);
  int saveOrbitReset(long timeStamp);
  int saveCtpCfg(uint32_t runNumber, long timeStamp);
  static CTPConfiguration getConfigFromCCDB(long timestamp, std::string run, bool& ok);
  CTPConfiguration getConfigFromCCDB(long timestamp, std::string run);
  CTPRunScalers getScalersFromCCDB(long timestamp, std::string run, bool& ok);
  static CTPRunScalers getScalersFromCCDB(long timestamp, std::string, std::string path, bool& ok);
  static void setCCDBHost(std::string host) { mCCDBHost = host; };
  static void setQCDBHost(std::string host) { mQCDBHost = host; };
  void setCtpCfgDir(std::string& ctpcfgdir) { mCtpCfgDir = ctpcfgdir; };

 protected:
  /// Database constants
  // std::string mCCDBHost = "http://ccdb-test.cern.ch:8080";
  // std::string mQCDBHost = "http://ali-qcdb.cern.ch:8083";
  static std::string mCCDBHost;
  static std::string mQCDBHost;
  const std::string mCCDBPathCTPScalers = "CTP/Calib/Scalers";
  // std::string mCCDBPathCTPConfig = "CTP/Config/Config";  - in Configuration.h
  const std::string mQCDBPathCTPScalers = "qc/CTP/Scalers";
  const std::string mCCDBPathSoxOrbit = "CTP/Calib/FirstRunOrbit";
  const std::string mCCDBPathOrbitReset = "CTP/Calib/OrbitReset";
  const std::string mCCDBPathCtpCfg = "CTP/Config/CtpCfg";
  std::string mCtpCfgDir;

  ClassDefNV(ctpCCDBManager, 2);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_CTPCCDB_H_
