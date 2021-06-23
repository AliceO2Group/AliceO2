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

#include "RefBuffers.h"
#include <array>
#include "MCHRawCommon/DataFormats.h"

extern std::array<const uint8_t, 496> REF_BUFFER_CRU_USERLOGIC_CHARGESUM;
template <>
gsl::span<const std::byte> REF_BUFFER_CRU<o2::mch::raw::UserLogicFormat, o2::mch::raw::ChargeSumMode>()
{
  return gsl::span<const std::byte>(reinterpret_cast<const std::byte*>(&REF_BUFFER_CRU_USERLOGIC_CHARGESUM[0]), REF_BUFFER_CRU_USERLOGIC_CHARGESUM.size());
}
std::array<const uint8_t, 496> REF_BUFFER_CRU_USERLOGIC_CHARGESUM = {
  // clang-format off
0x04, 0x40, 0x00, 0x00, 0x1E, 0x01, 0x00, 0x00, 0x80, 0x00, 0x80, 0x00, 
0x0F, 0x00, 0x0F, 0x00, 0x39, 0x30, 0x00, 0x00, 0x39, 0x30, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA6, 0x02, 0xA6, 0x02, 
0x03, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x13, 0x01, 0xF0, 0x40, 0x55, 0x55, 0xA1, 0x00, 
0x03, 0x12, 0x00, 0xE3, 0x46, 0x00, 0xA0, 0x00, 0x01, 0x60, 0xD0, 0x00, 
0x00, 0x58, 0xA2, 0x00, 0x04, 0x40, 0xBB, 0x11, 0x00, 0x01, 0xA0, 0x00, 
0x18, 0x14, 0x02, 0x40, 0x90, 0x04, 0xA0, 0x00, 0x70, 0x6F, 0x04, 0x40, 
0x00, 0x18, 0xA0, 0x00, 0xA3, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA0, 0x00, 
0xED, 0xDE, 0xED, 0xFE, 0xED, 0xDE, 0xED, 0xFE, 0x04, 0x40, 0x00, 0x00, 
0x1E, 0x01, 0x00, 0x00, 0x40, 0x00, 0x40, 0x00, 0x0F, 0x01, 0x0F, 0x00, 
0x39, 0x30, 0x00, 0x00, 0x39, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0xA6, 0x02, 0xA6, 0x02, 0x03, 0x08, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x04, 0x40, 0x00, 0x00, 0x1B, 0x01, 0x00, 0x00, 0xF0, 0x00, 0xF0, 0x00, 
0x0F, 0x00, 0x0D, 0x10, 0x39, 0x30, 0x00, 0x00, 0x39, 0x30, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA6, 0x02, 0xA6, 0x02, 
0x03, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x13, 0x01, 0xF0, 0x40, 0x55, 0x55, 0x81, 0x38, 
0x1A, 0x12, 0x80, 0xE0, 0x46, 0x00, 0x80, 0x38, 0x01, 0x60, 0xA0, 0x00, 
0x00, 0x4D, 0x82, 0x38, 0x04, 0x60, 0xB8, 0x11, 0x00, 0x01, 0x80, 0x38, 
0x18, 0x50, 0x00, 0x80, 0x90, 0x04, 0x80, 0x38, 0x28, 0x6E, 0x04, 0x40, 
0x00, 0x18, 0x80, 0x38, 0x1E, 0x00, 0x50, 0x21, 0x01, 0x38, 0x82, 0x38, 
0x1B, 0x01, 0x10, 0x00, 0x06, 0x28, 0x80, 0x38, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x80, 0x38, 0xED, 0xDE, 0xED, 0xFE, 0xED, 0xDE, 0xED, 0xFE, 
0x13, 0x01, 0xF0, 0x40, 0x55, 0x55, 0x01, 0x04, 0x03, 0x12, 0x40, 0xF6, 
0x46, 0x00, 0x00, 0x04, 0x01, 0x60, 0x40, 0x1A, 0x00, 0x54, 0x02, 0x04, 
0x04, 0xD0, 0xBD, 0x11, 0x00, 0x01, 0x00, 0x04, 0x18, 0xB8, 0x06, 0x00, 
0x96, 0x04, 0x00, 0x04, 0x84, 0x6F, 0x04, 0x40, 0x00, 0x18, 0x00, 0x04, 
0xB8, 0x01, 0xF0, 0x20, 0x01, 0x94, 0x03, 0x04, 0x1B, 0x01, 0x10, 0x00, 
0x06, 0xC2, 0x01, 0x04, 0x00, 0x00, 0x48, 0x00, 0xE9, 0x1B, 0x01, 0x04, 
0x00, 0x04, 0x80, 0x01, 0x73, 0x00, 0x00, 0x04, 0x48, 0x12, 0x50, 0xEA, 
0x46, 0x00, 0x00, 0x04, 0x01, 0x60, 0x40, 0x1A, 0x00, 0x00, 0x00, 0x04, 
0x04, 0x40, 0x00, 0x00, 0x1B, 0x01, 0x00, 0x00, 0x40, 0x00, 0x40, 0x00, 
0x0F, 0x01, 0x0D, 0x10, 0x39, 0x30, 0x00, 0x00, 0x39, 0x30, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA6, 0x02, 0xA6, 0x02, 
0x03, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00
  // clang-format on
};
