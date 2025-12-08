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
  float maxPt = 100.f;
  float rCompIBOB = 12.f;

  bool pvcontribOnly = true;
  bool addPVAsCluster = true;
  bool refitPV = true;
  bool useStableRef = true;
  bool doIBOB = true;
  bool doResid = true;

  O2ParamDef(CheckResidConfig, "checkresid");
};
} // namespace o2::checkresid

#endif
