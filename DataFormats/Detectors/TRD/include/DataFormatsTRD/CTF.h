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
/// \brief  Definitions for TRD CTF data

#ifndef O2_TRD_CTF_H
#define O2_TRD_CTF_H

#include <vector>
#include <Rtypes.h>
#include "DetectorsCommonDataFormats/EncodedBlocks.h"

namespace o2
{
namespace trd
{

/// Header for a single CTF
struct CTFHeader {
  uint32_t nTriggers = 0;  /// number of triggers
  uint32_t nTracklets = 0; /// number of tracklets
  uint32_t nDigits = 0;    /// number of digits
  uint32_t firstOrbit = 0; /// orbit of 1st trigger
  uint16_t firstBC = 0;    /// bc of 1st trigger
  uint16_t format = 0;     /// format word to be added to tracklet

  ClassDefNV(CTFHeader, 1);
};

/// wrapper for the Entropy-encoded triggers and cells of the TF
struct CTF : public o2::ctf::EncodedBlocks<CTFHeader, 15, uint32_t> {

  static constexpr size_t N = getNBlocks();
  enum Slots { BLC_bcIncTrig,
               BLC_orbitIncTrig,
               BLC_entriesTrk,
               BLC_entriesDig,
               BLC_HCIDTrk, // tracklers sorted in HCID -> 1st entry of trigger keeps abs HCID, then increments
               BLC_padrowTrk,
               BLC_colTrk,
               BLC_posTrk,
               BLC_slopeTrk,
               BLC_pidTrk,
               BLC_CIDDig, // digits sorted in CID -> 1st entry of trigger keeps abs CID, then increments
               BLC_ROBDig,
               BLC_MCMDig,
               BLC_chanDig,
               BLC_ADCDig
  };
  ClassDefNV(CTF, 1);
};

} // namespace trd
} // namespace o2

#endif
