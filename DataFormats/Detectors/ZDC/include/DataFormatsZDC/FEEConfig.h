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
//
// DataFormats/Detectors/ZDC/include/DataFormatsZDC/RawEventData.h

#include "ZDCBase/Constants.h"

#ifndef ALICEO2_ZDC_FEECONFIG_H
#define ALICEO2_ZDC_FEECONFIG_H

/// \file FEEConfig.h
/// \brief ZDC FEE configuration
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct FEEFillingMap {
  uint64_t filling[56];
};

struct FEEConfigMap {
  uint32_t address[5 * NChPerModule + 3] = {0, 1, 2, 3,
                                            4, 5, 6, 7,
                                            8, 9, 10, 11,
                                            12, 13, 14, 15,
                                            16, 17, 18, 19,
                                            76, 77, 78};
  uint64_t delay_sample[NChPerModule] = {6, 6, 6, 6};                                   // 4 bits
  uint64_t delay_coarse[NChPerModule] = {200, 200, 200, 200};                           // 8 bits
  uint64_t threshold_level[NChPerModule] = {10, 10, 10, 10};                            // 12 bits
  uint64_t difference_delta[NChPerModule] = {4, 4, 4, 4};                               // 3 bits
  uint64_t masking_difference[NChPerModule] = {0x00ff00, 0x00ff00, 0x00ff00, 0x00ff00}; // 24 bits
  uint64_t masking_alicet = 0x00000010;                                                 // 32 bits
  uint64_t masking_autot = 0xf;                                                         // 4 bits
  uint64_t masking_readout = 0xf;                                                       // 4 bits
};

} // namespace zdc
} // namespace o2

#endif
