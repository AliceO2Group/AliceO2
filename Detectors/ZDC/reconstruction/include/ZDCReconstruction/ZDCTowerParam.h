// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ZDC_TOWERPARAM_H
#define O2_ZDC_TOWERPARAM_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>

/// \file ZDCTowerParam.h
/// \brief ZDC Tower calibration
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct ZDCTowerParam {
  float tower_calib[NChannels] = {0}; // Tower calibration coefficients
  void setTowerCalib(uint32_t ich, float val);
  float getTowerCalib(uint32_t ich) const;
  void print();
  ClassDefNV(ZDCTowerParam, 1);
};
} // namespace zdc
} // namespace o2

#endif
