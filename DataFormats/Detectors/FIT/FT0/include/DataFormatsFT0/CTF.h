// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CTF.h
/// \author ruben.shahoyan@cern.ch
/// \brief  Definitions for FT0 CTF data

#ifndef O2_FT0_CTF_H
#define O2_FT0_CTF_H

#include <vector>
#include <Rtypes.h>
#include "DetectorsCommonDataFormats/EncodedBlocks.h"

namespace o2
{
namespace ft0
{

/// Header for a single CTF
struct CTFHeader {
  uint32_t nTriggers = 0;  /// number of truggers in TF
  uint32_t firstOrbit = 0; /// 1st orbit of TF
  uint16_t firstBC = 0;    /// 1st BC of TF

  ClassDefNV(CTFHeader, 1);
};

/// Intermediate, compressed but not yet entropy-encoded digits
struct CompressedDigits {

  CTFHeader header;

  // trigger data
  std::vector<uint8_t> trigger;   // trigger bits
  std::vector<uint16_t> bcInc;    // increment in BC if the same orbit, otherwise abs bc
  std::vector<uint32_t> orbitInc; // increment in orbit
  std::vector<uint8_t> nChan;     // number of fired channels

  // channel data
  std::vector<uint8_t> idChan;   // channels ID: 1st on absolute, then increment
  std::vector<int16_t> cfdTime;  // CFD time
  std::vector<uint32_t> qtcAmpl; // Amplitude
  std::vector<uint8_t> qtc;      // QTC

  CompressedDigits() = default;

  void clear();

  ClassDefNV(CompressedDigits, 1);
};

/// wrapper for the Entropy-encoded clusters of the TF
struct CTF : public o2::ctf::EncodedBlocks<CTFHeader, 8, uint32_t> {

  static constexpr size_t N = getNBlocks();
  enum Slots {
    BLC_trigger,  // trigger bits
    BLC_bcInc,    // increment in BC
    BLC_orbitInc, // increment in orbit
    BLC_nChan,    // number of fired channels
    BLC_idChan,   // channels ID: 1st on absolute, then increment
    BLC_qtc,      // QTC
    BLC_cfdTime,  // CFD time
    BLC_qtcAmpl
  }; // Amplitude

  ClassDefNV(CTF, 1);
};

} // namespace ft0
} // namespace o2

#endif
