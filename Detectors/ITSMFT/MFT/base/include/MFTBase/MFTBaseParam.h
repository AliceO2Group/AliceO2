// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  O2ParamDef(MFTBaseParam, "MFTBase");
};

} // end namespace mft
} // end namespace o2

#endif // ALICEO2_MFT_BASEPARAM_H_
