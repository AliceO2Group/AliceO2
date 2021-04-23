// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
/// \brief Parameters to correct TDCs
/// \author P. Cortese

namespace o2
{
namespace zdc
{
// parameters of ZDC reconstruction

struct ZDCTDCParam {
  float tdc_shift[NTDCChannels] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Correction of TDC position (ns)
 public:
  void setShift(uint32_t ich, float val);
  float getShift(uint32_t ich) const;
  void print();

  ClassDefNV(ZDCTDCParam, 1);
};
} // namespace zdc
} // namespace o2

#endif
