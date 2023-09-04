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
/// \brief  Definitions for ITS/MFT CTF data

#ifndef O2_ITSMFT_CTF_H
#define O2_ITSMFT_CTF_H

#include <vector>
#include <Rtypes.h>
#include "DetectorsCommonDataFormats/EncodedBlocks.h"

namespace o2
{
namespace itsmft
{

class ROFRecord;
class CompClusterExt;

/// Header for a single CTF
struct CTFHeader : public o2::ctf::CTFDictHeader {
  uint32_t nROFs = 0;         /// number of ROFrame in TF
  uint32_t nClusters = 0;     /// number of clusters in TF
  uint32_t nChips = 0;        /// number of fired chips in TF : this is for the version with chipInc stored once per new chip
  uint32_t nPatternBytes = 0; /// number of bytes for explict patterns
  uint32_t firstOrbit = 0;    /// 1st orbit of TF
  uint16_t firstBC = 0;       /// 1st BC of TF
  ClassDefNV(CTFHeader, 2);
};

/// Compressed but not yet entropy-encoded clusters
struct CompressedClusters {

  CTFHeader header;

  // ROF header data
  std::vector<uint16_t> firstChipROF; /// 1st chip ID in the ROF
  std::vector<uint16_t> bcIncROF;     /// increment of ROF BC wrt BC of previous ROF
  std::vector<uint32_t> orbitIncROF;  /// increment of ROF orbit wrt orbit of previous ROF
  std::vector<uint32_t> nclusROF;     /// number of clusters in ROF

  // Chip data
  std::vector<uint16_t> chipInc; /// increment of chipID wrt that of prev. chip
  std::vector<uint16_t> chipMul; /// clusters in chip
  std::vector<uint16_t> row;     /// row of fired pixel
  std::vector<int16_t> colInc;   /// increment of pixel column wrt that of prev. pixel (sometimes can be slightly negative)
  std::vector<uint16_t> pattID;  /// cluster pattern ID
  std::vector<uint8_t> pattMap;  /// explict patterns container

  CompressedClusters() = default;

  void clear();

  ClassDefNV(CompressedClusters, 1);
};

/// wrapper for the Entropy-encoded clusters of the TF
struct CTF : public o2::ctf::EncodedBlocks<CTFHeader, 10, uint32_t> {

  static constexpr size_t N = getNBlocks();
  enum Slots { BLCfirstChipROF,
               BLCbcIncROF,
               BLCorbitIncROF,
               BLCnclusROF,
               BLCchipInc,
               BLCchipMul,
               BLCrow,
               BLCcolInc,
               BLCpattID,
               BLCpattMap };
  ClassDefNV(CTF, 1);
};

} // namespace itsmft
} // namespace o2

#endif
