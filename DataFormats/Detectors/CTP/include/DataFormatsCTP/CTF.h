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

/// \file   CTF.h
/// \author ruben.shahoyan@cern.ch
/// \brief  Definitions for CTP CTF data

#ifndef O2_CTP_CTF_H
#define O2_CTP_CTF_H

#include <vector>
#include <Rtypes.h>
#include "DetectorsCommonDataFormats/EncodedBlocks.h"

namespace o2
{
namespace ctp
{

/// Header for a single CTF
struct CTFHeader : public o2::ctf::CTFDictHeader {
  uint64_t lumiCounts = 0;    /// FT0 Luminosity counts moving average over lumiNHBFs orbits
  uint64_t lumiCountsFV0 = 0; /// FV0 Luminosity counts moving average over lumiNHBFs orbits
  uint32_t lumiNHBFs = 0;  /// Number of HBFs over which lumi is integrated
  uint32_t lumiOrbit = 0;  /// 1st orbit of TF where lumi was updated, can be compared with firstOrbit
  uint32_t nTriggers = 0;  /// number of triggers
  uint32_t firstOrbit = 0; /// orbit of 1st trigger
  uint16_t firstBC = 0;    /// bc of 1st trigger

  ClassDefNV(CTFHeader, 3);
};

/// wrapper for the Entropy-encoded trigger inputs and classes of the TF
struct CTF : public o2::ctf::EncodedBlocks<CTFHeader, 4, uint32_t> {

  static constexpr size_t N = getNBlocks();
  enum Slots { BLC_bcIncTrig,
               BLC_orbitIncTrig,
               BLC_bytesInput, // bytes of the CTPInputMask bitset (6 bytes from lowest to highest)
               BLC_bytesClass  // bytes of the CTPClassMask bitset (8 bytes from lowest to highest)
  };
  ClassDefNV(CTF, 2);
};

} // namespace ctp
} // namespace o2

#endif
