// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _ZDC_TDC_DATA_H
#define _ZDC_TDC_DATA_H

#include "ZDCBase/Constants.h"
#include <array>
#include <Rtypes.h>

/// \file TDCData.h
/// \brief Container class to store a TDC hit in a ZDC channel
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct TDCData {

  int8_t id = IdDummy; // channel ID
  int16_t val;         // tdc value
  int16_t amp;         // tdc amplitude

  TDCData() = default;
  TDCData(int8_t ida, int16_t vala, int16_t ampa)
  {
    id = ida;
    val = vala;
    amp = ampa;
  }

  void print() const;

  ClassDefNV(TDCData, 1);
};
} // namespace zdc
} // namespace o2

#endif
