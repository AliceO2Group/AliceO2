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

  bool continuous = true;                      ///< flag for continuous simulation
  /// Build the +-113 m beam line, its magnets and the ZN/ZP calorimeters that sit
  /// there. This is what the ZDC costs in transport time; turning it off leaves the
  /// ZEM calorimeters at z ~ 7.6 m in the geometry.
  bool buildBeamLine = true;
  bool buildZEM = true;                        ///< build the ZEM calorimeters
  int nBCAheadCont = 1;                        ///< number of BC to read ahead of trigger in continuous mode
  int nBCAheadTrig = 3;                        ///< number of BC to read ahead of trigger in triggered mode
  bool recordSpatialResponse = false;          ///< whether to record 2D spatial response showering images in proton/neutron detector
  bool useZDCFastSim = false;                  ///< replace ZDC geometry and transport by the fast simulation model (spectator p and n only)
  std::string fastSimCCDBPath = "ZDC/Config/FastSimModel"; ///< CCDB path of the fast simulation model (o2::zdc::FastSimModel)
  std::string fastSimLocalFile = "";                       ///< if not empty: local ROOT file with the model, used instead of the CCDB

  O2ParamDef(ZDCSimParam, "ZDCSimParam");
};
} // namespace zdc
} // namespace o2

#endif
