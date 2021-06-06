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

/// \author R. Pezzi - March 2020

#ifndef ALICEO2_MFT_BASEPARAM_H_
#define ALICEO2_MFT_BASEPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mft
{

// **
// ** Parameters for MFT base configuration
// **
struct MFTBaseParam : public o2::conf::ConfigurableParamHelper<MFTBaseParam> {
  // Geometry Builder parameters
  // Within MFT acceptance
  bool buildHeatExchanger = true;
  bool buildFlex = true;

  // Out of acceptance
  bool buildCone = true;
  bool buildBarrel = true;
  bool buildPatchPanel = true;
  bool buildPCBSupports = true;
  bool buildPCBs = true;
  bool buildPSU = true;
  bool buildReadoutCables = true;
  bool buildServices = true;

  // General configurations
  bool minimal = false; // Disables all elements out of MFT acceptance

  // General misalignment input parameters
  bool misalignHalf = false;
  bool misalignDisk = false;
  bool misalignLadder = false;
  bool misalignSensor = true;

  double xHalf = 0.0;
  double yHalf = 0.0;
  double zHalf = 0.0;
  double psiHalf = 0.0;
  double thetaHalf = 0.0;
  double phiHalf = 0.0;

  double xDisk = 0.0;
  double yDisk = 0.0;
  double zDisk = 0.0;
  double psiDisk = 0.0;
  double thetaDisk = 0.0;
  double phiDisk = 0.0;

  double xLadder = 0.0;
  double yLadder = 0.0;
  double zLadder = 0.0;
  double psiLadder = 0.0;
  double thetaLadder = 0.0;
  double phiLadder = 0.0;

  double xSensor = 0.0;
  double ySensor = 0.0;
  double zSensor = 0.0;
  double psiSensor = 0.0;
  double thetaSensor = 0.0;
  double phiSensor = 0.0;

  O2ParamDef(MFTBaseParam, "MFTBase");
};

} // end namespace mft
} // end namespace o2

#endif // ALICEO2_MFT_BASEPARAM_H_
