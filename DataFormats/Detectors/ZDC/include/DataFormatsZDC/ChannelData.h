// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _ZDC_CHANNEL_DATA_H_
#define _ZDC_CHANNEL_DATA_H_

#include "ZDCBase/Constants.h"
#include <array>
#include <Rtypes.h>

/// \file ChannelData.h
/// \brief Container class to store NTimeBinsPerBC ADC values of single ZDC channel
/// \author ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{

struct ChannelData {

  int8_t id = IdDummy; // channel ID
  std::array<int16_t, NTimeBinsPerBC> data = {0};

  ChannelData() = default;
  ChannelData(int8_t ida, const std::array<float, NTimeBinsPerBC>& src)
  {
    id = ida;
    for (int i = NTimeBinsPerBC; i--;) {
      data[i] = int16_t(src[i]);
    }
  }

  void print() const;

  ClassDefNV(ChannelData, 1);
};
} // namespace zdc
} // namespace o2

#endif
