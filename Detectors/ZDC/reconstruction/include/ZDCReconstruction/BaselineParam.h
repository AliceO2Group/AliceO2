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

#ifndef ZDC_BASELINECALIB_PARAM_H
#define ZDC_BASELINECALIB_PARAM_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <vector>

/// \file BaselineParam.h
/// \brief Baseline calibration data
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct BaselineParam {

  BaselineParam();
  std::array<bool, NChannels> modified{};
  float baseline[NChannels] = {}; // configuration per channel
  void setCalib(uint32_t ich, float val, bool ismodified = true);
  float getCalib(uint32_t ich) const;
  void print(bool printall = true) const;

  ClassDefNV(BaselineParam, 1);
};

} // namespace zdc
} // namespace o2

#endif
