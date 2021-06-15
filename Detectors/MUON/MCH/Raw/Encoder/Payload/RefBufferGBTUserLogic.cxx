// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See http://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "RefBuffers.h"
#include <array>
#include "MCHRawCommon/DataFormats.h"

extern std::array<const uint8_t, 96> REF_BUFFER_GBT_USERLOGIC_CHARGESUM;
template <>
gsl::span<const std::byte> REF_BUFFER_GBT<o2::mch::raw::UserLogicFormat, o2::mch::raw::ChargeSumMode>()
{
  return gsl::span<const std::byte>(reinterpret_cast<const std::byte*>(&REF_BUFFER_GBT_USERLOGIC_CHARGESUM[0]), REF_BUFFER_GBT_USERLOGIC_CHARGESUM.size());
}
std::array<const uint8_t, 96> REF_BUFFER_GBT_USERLOGIC_CHARGESUM = {
  // clang-format off
0x13, 0x01, 0xF0, 0x40, 0x55, 0x55, 0x01, 0x58, 0x0C, 0x12, 0x00, 0xA0, 
0x50, 0x03, 0x00, 0x58, 0x01, 0x30, 0xA0, 0x00, 0x00, 0x5B, 0x02, 0x58, 
0x04, 0xC0, 0x2F, 0xD4, 0x00, 0x01, 0x00, 0x58, 0x0C, 0x80, 0x02, 0x00, 
0x00, 0x00, 0x00, 0x58, 0x13, 0x01, 0xF0, 0x40, 0x55, 0x55, 0x61, 0x58, 
0x19, 0x12, 0x60, 0xAD, 0x50, 0x03, 0x60, 0x58, 0x01, 0x30, 0xD0, 0x00, 
0x00, 0x09, 0x62, 0x58, 0x04, 0x5C, 0x28, 0xD4, 0x00, 0x01, 0x60, 0x58, 
0x0C, 0x14, 0x02, 0x40, 0x82, 0x04, 0x60, 0x58, 0xF7, 0x0B, 0x35, 0x40, 
0x00, 0x0C, 0x60, 0x58, 0xA3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x58

  // clang-format on
};
