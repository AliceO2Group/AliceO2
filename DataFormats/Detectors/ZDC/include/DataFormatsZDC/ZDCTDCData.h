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

#ifndef ZDC_TDC_DATA_H
#define ZDC_TDC_DATA_H

#include "ZDCBase/Constants.h"
#include <array>
#include <Rtypes.h>

/// \file ZDCTDCData.h
/// \brief Container class to store a TDC hit in a ZDC channel
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct ZDCTDCData {

  int8_t id = IdDummy; // channel ID
  int16_t val = 0;     // tdc value
  int16_t amp = 0;     // tdc amplitude

  ZDCTDCData() = default;
  ZDCTDCData(int8_t ida, int16_t vala, int16_t ampa)
  {
    id = ida;
    val = vala;
    amp = ampa;
  }

  inline float amplitude() const
  {
    return FTDCAmp * amp;
  }
  inline float value() const
  {
    return FTDCVal * val;
  }
  inline uint8_t ch() const
  {
    return id;
  }

  void print() const;

  ClassDefNV(ZDCTDCData, 1);
};
} // namespace zdc
} // namespace o2

#endif
