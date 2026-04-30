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

#ifndef O2_CHECK_RESID_CONFIG_H
#define O2_CHECK_RESID_CONFIG_H
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::checkresid
{
struct CheckResidConfig : o2::conf::ConfigurableParamHelper<CheckResidConfig> {
  int minPVContributors = 10;
  int minTPCCl = 60;
  int minITSCl = 7;
  float minPt = 0.4f;
  float maxPt = 50.f;
  float maxTgl = 2.f;
  float rCompIBOB = 12.f;

  bool pvcontribOnly = true;
  bool addPVAsCluster = true;
  bool useStableRef = true;
  bool doIBOB = true;
  bool doResid = true;

  bool refitPV = true;
  float refitPVMV = false;
  float refitPVIniScale = 100.f;

  std::string outname{"checkResid"};
  // histogram settings
  int nBinsRes = 100;
  int nBinsPhi = 30;
  int nBinsZ = 20;
  int nBinsPt = 15;
  int nBinsTgl = 20;
  int minHistoStat2Fit = 1000;
  float maxPull = 4;
  float zranges[8] = {10.f, 15.f, 15.f, 15.f, 40.f, 40.f, 74.f, 74.f};
  float maxDYZ[8] = {0.03, 0.015, 0.01, 0.01, 0.08, 0.08, 0.12, 0.1};
  float maxDPar[5] = {0.15, 0.15, 0.015, 0.015, 1.};
  // drawing settings
  float resMMLrY[8] = {0.003, 0.003, 0.003, 0.003, 0.005, 0.005, 0.005, 0.005};
  float resMMLrZ[8] = {0.002, 0.0015, 0.0015, 0.0015, 0.005, 0.005, 0.005, 0.005};
  float resMMPar[5] = {0.03, 0.01, 0.005, 0.001, 0.5};
  //
  // string with existing histomanagers files to draw (comma or semicolon separated) and optional legends
  std::string ext_hm_list{};
  std::string ext_leg_list{};

  O2ParamDef(CheckResidConfig, "checkresid");
};
} // namespace o2::checkresid

#endif
