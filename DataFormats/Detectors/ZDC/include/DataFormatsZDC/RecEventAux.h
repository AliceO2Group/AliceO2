// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _ZDC_RECEVENT_AUX_H
#define _ZDC_RECEVENT_AUX_H

#include "CommonDataFormat/InteractionRecord.h"
#include "MathUtils/Cartesian.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>
#include <map>

/// \file RecEvent.h
/// \brief Class to describe reconstructed ZDC event (single BC with signal in one of detectors) during the reconstruction stage
/// \author cortese@to.infn.it, ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{

struct RecEventAux {
  o2::InteractionRecord ir;
  uint32_t flags;                                        /// reconstruction flags
  constexpr static float fAmp = 1. / 8.;                 /// Multiplication factor in conversion from integer
  constexpr static float fVal = 1. / TSNS;               /// Multiplication factor in conversion from integer
  std::map<uint8_t, float> ezdc;                         /// signal in ZDCs
  int16_t tdcVal[NTDCChannels][MaxTDCValues];            /// TdcChannels
  int16_t tdcAmp[NTDCChannels][MaxTDCValues];            /// TdcAmplitudes
  int ntdc[NTDCChannels] = {0};                          /// Number of hits in TDC
  std::array<bool, NTDCChannels> pattern;                /// Pattern of TDC
  uint16_t fired[NTDCChannels] = {0};                    /// Position at which the trigger algorithm is fired
  float inter[NTDCChannels][NTimeBinsPerBC * TSN] = {0}; /// Interpolated samples
  uint32_t ref[NChannels] = {O2_ZDC_REF_INIT};           /// Cache of references

  // Functions
  void print() const;
  float EZDC(uint8_t ich)
  {
    std::map<uint8_t, float>::iterator it = ezdc.find(ich);
    if (it != ezdc.end()) {
      return it->second;
    } else {
      return -std::numeric_limits<float>::infinity();
    }
  }

  float EZNAC() { return EZDC(IdZNAC); }
  float EZNA1() { return EZDC(IdZNA1); }
  float EZNA2() { return EZDC(IdZNA2); }
  float EZNA3() { return EZDC(IdZNA3); }
  float EZNA4() { return EZDC(IdZNA4); }
  float EZNASum() { return EZDC(IdZNASum); }

  float EZPAC() { return EZDC(IdZPAC); }
  float EZPA1() { return EZDC(IdZPA1); }
  float EZPA2() { return EZDC(IdZPA2); }
  float EZPA3() { return EZDC(IdZPA3); }
  float EZPA4() { return EZDC(IdZPA4); }
  float EZPASum() { return EZDC(IdZPASum); }

  float EZEM1() { return EZDC(IdZEM1); }
  float EZEM2() { return EZDC(IdZEM2); }

  float EZNCC() { return EZDC(IdZNCC); }
  float EZNC1() { return EZDC(IdZNC1); }
  float EZNC2() { return EZDC(IdZNC2); }
  float EZNC3() { return EZDC(IdZNC3); }
  float EZNC4() { return EZDC(IdZNC4); }
  float EZNCSum() { return EZDC(IdZNCSum); }

  float EZPCC() { return EZDC(IdZPCC); }
  float EZPC1() { return EZDC(IdZPC1); }
  float EZPC2() { return EZDC(IdZPC2); }
  float EZPC3() { return EZDC(IdZPC3); }
  float EZPC4() { return EZDC(IdZPC4); }
  float EZPCSum() { return EZDC(IdZPCSum); }
  ClassDefNV(RecEventAux, 1);
};

} // namespace zdc

/// Defining RecEventAux explicitly as messageable
///
/// It does not fulfill is_messageable because the underlying ROOT
/// classes of Point2D are note trivially copyable.
namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::zdc::RecEventAux> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif
