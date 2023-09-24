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
/// \author fnoferin@cern.ch
/// \brief  Definitions for TOF CTF data

#ifndef O2_TOF_CTF_H
#define O2_TOF_CTF_H

#include <vector>
#include <Rtypes.h>
#include "DetectorsCommonDataFormats/EncodedBlocks.h"

namespace o2
{
namespace tof
{

class ROFRecord;
class CompClusterExt;

/// Header for a single CTF
struct CTFHeader : public o2::ctf::CTFDictHeader {
  uint32_t nROFs = 0;         /// number of ROFrame in TF
  uint32_t nDigits = 0;       /// number of digits in TF
  uint32_t nPatternBytes = 0; /// number of bytes for explict patterns
  uint32_t firstOrbit = 0;    /// 1st orbit of TF
  uint16_t firstBC = 0;       /// 1st BC of TF

  ClassDefNV(CTFHeader, 2);
};

/// Compressed but not yet entropy-encoded infos
struct CompressedInfos {

  CTFHeader header;

  /*
    ROF = 1/3 orbit = 1188 BC
    1 TF = 128 * 3 = 364 ROF
    TIMEFRAME = 2^8 TDC
    1 BC = 2^10 TDC = 4 TIMEFRAME
    ROF = 4752 TIMEFRAME < 2^13 TIMEFRAME

    timeFrame = deltaBC/64;
    timeTDC = (deltaBC%64)*1024+ dig.getTDC();
    */

  // ROF header data
  std::vector<int16_t> bcIncROF;     /// increment of ROF BC wrt BC of previous ROF
  std::vector<int32_t> orbitIncROF;  /// increment of ROF orbit wrt orbit of previous ROF
  std::vector<uint32_t> ndigROF;     /// number of digits in ROF
  std::vector<uint32_t> ndiaROF;     /// number of diagnostic/pattern words in ROF
  std::vector<uint32_t> ndiaCrate;   /// number of diagnostic/pattern words per crate in ROF

  // Hit data
  std::vector<int16_t> timeFrameInc;  /// time increment with respect of previous digit in TimeFrame units
  std::vector<uint16_t> timeTDCInc;   /// time increment with respect of previous digit in TDC channel (about 24.4 ps) within timeframe
  std::vector<uint16_t> stripID;      /// increment of stripID wrt that of prev. strip
  std::vector<uint8_t> chanInStrip;   /// channel in strip 0-95 (ordered in time)
  std::vector<uint16_t> tot;          /// Time-Over-Threshold in TOF channel (about 48.8 ps)
  std::vector<uint8_t> pattMap;       /// explict patterns container

  CompressedInfos() = default;

  void clear();

  ClassDefNV(CompressedInfos, 3);
};

/// wrapper for the Entropy-encoded clusters of the TF
struct CTF : public o2::ctf::EncodedBlocks<CTFHeader, 11, uint32_t> {

  static constexpr size_t N = getNBlocks();
  enum Slots { BLCbcIncROF,
               BLCorbitIncROF,
               BLCndigROF,
               BLCndiaROF,
               BLCndiaCrate,
               BLCtimeFrameInc,
               BLCtimeTDCInc,
               BLCstripID,
               BLCchanInStrip,
               BLCtot,
               BLCpattMap };

  ClassDefNV(CTF, 1);
};

} // namespace tof
} // namespace o2

#endif
