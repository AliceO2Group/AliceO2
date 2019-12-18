// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ZDC_SIMPARAMS_H_
#define O2_ZDC_SIMPARAMS_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

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

  O2ParamDef(ZDCSimParam, "ZDCSimParam");
};
} // namespace zdc
} // namespace o2

#endif
