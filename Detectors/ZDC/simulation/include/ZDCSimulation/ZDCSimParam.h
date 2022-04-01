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

#ifndef O2_ZDC_SIMPARAMS_H_
#define O2_ZDC_SIMPARAMS_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace zdc
{
// parameters of ZDC digitization / transport simulation

struct ZDCSimParam : public o2::conf::ConfigurableParamHelper<ZDCSimParam> {

  bool continuous = true; ///< flag for continuous simulation
  int nBCAheadCont = 1;   ///< number of BC to read ahead of trigger in continuous mode
  int nBCAheadTrig = 3;   ///< number of BC to read ahead of trigger in triggered mode
  bool recordSpatialResponse = false; ///< whether to record 2D spatial response showering images in proton/neutron detector
  bool useZDCFastSim = false;         ///< whether to use fastsim module on event
  std::string ZDCFastSimClassifierPath = "";   ///< path to model file that classify if data are viable for model
  std::string ZDCFastSimClassifierScales = ""; ///< path to scales file for classifier
  std::string ZDCFastSimModelPath = "";        ///< path to model file
  std::string ZDCFastSimModelScales = "";      ///< path to scales file for model

  O2ParamDef(ZDCSimParam, "ZDCSimParam");
};
} // namespace zdc
} // namespace o2

#endif
