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
/// \brief  Definitions for MCH CTF data

#ifndef O2_MCH_CTF_H
#define O2_MCH_CTF_H

#include <vector>
#include <Rtypes.h>
#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include <iosfwd>

namespace o2
{
namespace mch
{

/// Header for a single CTF
struct CTFHeader {
  uint32_t nROFs = 0;      /// number of ROFrames in TF
  uint32_t nDigits = 0;    /// number of digits in TF
  uint32_t firstOrbit = 0; /// 1st orbit of TF
  uint16_t firstBC = 0;    /// 1st BC of TF

  ClassDefNV(CTFHeader, 1);
};

/// wrapper for the Entropy-encoded clusters of the TF
struct CTF : public o2::ctf::EncodedBlocks<CTFHeader, 9, uint32_t> {

  static constexpr size_t N = getNBlocks();
  enum Slots { BLC_bcIncROF,
               BLC_orbitIncROF,
               BLC_nDigitsROF,
               BLC_tfTime,
               BLC_nSamples,
               BLC_isSaturated,
               BLC_detID,
               BLC_padID,
               BLC_ADC };
  ClassDefNV(CTF, 1);
};

std::ostream& operator<<(std::ostream&, const CTFHeader&);

} // namespace mch
} // namespace o2

#endif
