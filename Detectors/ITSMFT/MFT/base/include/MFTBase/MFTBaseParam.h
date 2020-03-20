// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R. Pezzi - Março 2020

#ifndef ALICEO2_MFT_BASEPARAM_H_
#define ALICEO2_MFT_BASEPARAM_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

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
  bool buildCone = false;
  bool buildBarrel = false;
  bool buildPatchPanel = false;
  bool buildPCBSupports = false;
  bool buildPCBs = false;
  bool buildPSU = false;

  // General configurations
  bool buildFullMFT = false;
  // bool geometryDebug = false; // A debug option would be usefull for suppressing textual output, such as those from the HeatExchanger

  O2ParamDef(MFTBaseParam, "MFTBase");
};

} // end namespace mft
} // end namespace o2

#endif // ALICEO2_MFT_BASEPARAM_H_
