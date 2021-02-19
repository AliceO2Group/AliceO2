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
/// \brief  Definitions for ZDC CTF data

#ifndef O2_ZDC_CTF_H
#define O2_ZDC_CTF_H

#include <vector>
#include <Rtypes.h>
#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/PedestalData.h"

namespace o2
{
namespace zdc
{

/// Header for a single CTF
struct CTFHeader {
  uint32_t nTriggers = 0;     /// number of triggers
  uint32_t nChannels = 0;     /// number of referred channels
  uint32_t nPedestals = 0;    /// number of pedestal objects
  uint32_t firstOrbit = 0;    /// orbit of 1st trigger
  uint32_t firstOrbitPed = 0; /// orbit of 1st pedestal
  uint16_t firstBC = 0;       /// bc of 1st trigger

  ClassDefNV(CTFHeader, 1);
};

/// wrapper for the Entropy-encoded triggers and cells of the TF
struct CTF : public o2::ctf::EncodedBlocks<CTFHeader, 11, uint32_t> {

  static constexpr size_t N = getNBlocks();
  enum Slots { BLC_bcIncTrig,
               BLC_orbitIncTrig,
               BLC_moduleTrig,
               BLC_channelsHL, // 32-bit channels pattern word split to 2 16-bit words stored as H, then L
               BLC_triggersHL, // 32-bit trigger word split to 2 16-bit words stored as H, then L
               BLC_extTriggers,
               BLC_nchanTrig,
               //
               BLC_chanID,
               BLC_chanData,
               //
               BLC_orbitIncPed,
               BLC_pedData
  };
  ClassDefNV(CTF, 1);
};

} // namespace zdc
} // namespace o2

#endif
