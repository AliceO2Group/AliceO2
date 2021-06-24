// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ZDC_ENERGYPARAM_H
#define O2_ZDC_ENERGYPARAM_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>

/// \file ZDCEnergyParam.h
/// \brief ZDC Energy calibration
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct ZDCEnergyParam {
  float energy_calib[NChannels] = {0}; // Energy calibration coefficients
  void setEnergyCalib(uint32_t ich, float val);
  float getEnergyCalib(uint32_t ich) const;
  void print();
  ClassDefNV(ZDCEnergyParam, 1);
};
} // namespace zdc
} // namespace o2

#endif
