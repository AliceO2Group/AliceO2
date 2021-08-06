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

/// \file IDCGroupingParameter.h
/// \brief Definition of the parameter for the grouping of the IDCs
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_SPACECHARGEPARAMETER_H_
#define ALICEO2_TPC_SPACECHARGEPARAMETER_H_

#include <array>
#include "CommonUtils/ConfigurableParamHelper.h"
#include "Rtypes.h" // for ClassDefNV

namespace o2
{
namespace tpc
{

/// struct for setting the parameters for the grouping of IDCs
struct ParameterSpaceCharge : public o2::conf::ConfigurableParamHelper<ParameterSpaceCharge> {
  unsigned short NRVertices = 129;   /// NRVertices number of vertices in z direction
  unsigned short NZVertices = 129;   /// NZVertices number of vertices in r direction
  unsigned short NPhiVertices = 180; /// NRPhiVertices number of vertices in phi direction

  O2ParamDef(ParameterSpaceCharge, "TPCSpaceChargeParam");
};

} // namespace tpc
} // namespace o2

#endif
