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

#ifndef O2_ZDC_TDCPARAM_H_
#define O2_ZDC_TDCPARAM_H_

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>

/// \file ZDCTDCParam.h
/// \brief Parameters to correct TDCs (produced by QA)
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct ZDCTDCParam {
  //std::array<float, NTDCChannels> tdcShift{}
  float tdc_shift[NTDCChannels] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Correction of TDC position (ns)
  void setShift(uint32_t ich, float val);
  float getShift(uint32_t ich) const;
  void print();
  ClassDefNV(ZDCTDCParam, 1);
};
} // namespace zdc
} // namespace o2

#endif
