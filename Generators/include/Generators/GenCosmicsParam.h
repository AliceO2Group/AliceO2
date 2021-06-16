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

#ifndef ALICEO2_GENCOSMICSPARAM_H
#define ALICEO2_GENCOSMICSPARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

//  @file   GenCosmicsParam
//  @author ruben.shahoyan@cern.ch
//  @brief  Parameters for cosmics generation

namespace o2
{
namespace eventgen
{
struct GenCosmicsParam : public o2::conf::ConfigurableParamHelper<GenCosmicsParam> {
  enum GenParamType : int { ParamMI,
                            ParamACORDE,
                            ParamTPC }; // source parameterizations
  enum AccType : int { ITS0,
                       ITS1,
                       ITS2,
                       ITS3,
                       ITS4,
                       ITS5,
                       ITS6,
                       TPC,
                       Custom };
  GenParamType param = ParamTPC;
  AccType accept = TPC;
  int nPart = 1;            ///< number of particles per event
  int maxTrials = 10000000; ///< number of failed trials to abandon generation
  float maxAngle = 45.;     ///< max angle wrt azimuth to generate (in degrees)
  float origin = 550.;      ///< create particle at this radius
  float pmin = 0.5;         ///< min total momentum
  float pmax = 100;         ///< max total momentum
  float customAccX = 250;   ///< require particle to pass within this |X| at Y=0 if AccType=custom is selected
  float customAccZ = 250;   ///< require particle to pass within this |Z| at Y=0 if AccType=custom is selected

  // boilerplate stuff + make principal key
  O2ParamDef(GenCosmicsParam, "cosmics");
};

} // namespace eventgen
} // namespace o2

#endif
