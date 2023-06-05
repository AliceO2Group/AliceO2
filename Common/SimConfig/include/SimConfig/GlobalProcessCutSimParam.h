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

#ifndef DETECTORS_GCONFIG_INCLUDE_SIMSETUP_GLOBALPROCESSCUTSIMPARAM_H_
#define DETECTORS_GCONFIG_INCLUDE_SIMSETUP_GLOBALPROCESSCUTSIMPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{

struct GlobalProcessCutSimParam : public o2::conf::ConfigurableParamHelper<GlobalProcessCutSimParam> {
  int PAIR = 1;
  int COMP = 1;
  int PHOT = 1;
  int PFIS = 0;
  int DRAY = 0;
  int ANNI = 1;
  int BREM = 1;
  int HADR = 1;
  int MUNU = 1;
  int DCAY = 1;
  int LOSS = 2;
  int MULS = 1;
  int CKOV = 1;

  double CUTGAM = 1.0E-3; // GeV --> 1 MeV
  double CUTELE = 1.0E-3; // GeV --> 1 MeV
  double CUTNEU = 1.0E-3; // GeV --> 1 MeV
  double CUTHAD = 1.0E-3; // GeV --> 1 MeV
  double CUTMUO = 1.0E-3; // GeV --> 1 MeV
  double BCUTE = 1.0E-3;  // GeV --> 1 MeV
  double BCUTM = 1.0E-3;  // GeV --> 1 MeV
  double DCUTE = 1.0E-3;  // GeV --> 1 MeV
  double DCUTM = 1.0E-3;  // GeV --> 1 MeV
  double PPCUTM = 1.0E-3; // GeV --> 1 MeV
  double TOFMAX = 0.1;    // seconds

  // boilerplate stuff + make principal key "GlobalSimProcs"
  O2ParamDef(GlobalProcessCutSimParam, "GlobalSimProcs");
};

} // namespace o2

#endif /* DETECTORS_GCONFIG_INCLUDE_SIMSETUP_GLOBALPROCESSCUTSIMPARAM_H_ */
