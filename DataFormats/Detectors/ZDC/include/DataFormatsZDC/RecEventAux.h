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

#ifndef ZDC_RECEVENT_AUX_H
#define ZDC_RECEVENT_AUX_H

#include "CommonDataFormat/InteractionRecord.h"
#include "MathUtils/Cartesian.h"
#include "DataFormatsZDC/RecEventFlat.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>
#include <vector>
#include <map>

/// \file RecEvent.h
/// \brief Class to describe reconstructed ZDC event (single BC with signal in one of detectors) during the reconstruction stage
/// \author cortese@to.infn.it, ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{

struct RecEventAux : public RecEventFlat {
  uint32_t flags;
#ifdef O2_ZDC_TDC_C_ARRAY
  int16_t tdcVal[NTDCChannels][MaxTDCValues]; /// TdcValues (encoded)
  int16_t tdcAmp[NTDCChannels][MaxTDCValues]; /// TdcAmplitudes (encoded)
#endif
  int ntdc[NTDCChannels] = {0};                              /// Number of hits in TDC
  std::array<bool, NTDCChannels> pattern;                    /// Pattern of TDC
  uint16_t fired[NTDCChannels] = {0};                        /// Position at which the trigger algorithm is fired
  bool chfired[NChannels] = {0};                             /// Fired TDC condition related to channel
  uint32_t ref[NChannels];                                   /// Cache of references
  std::array<bool, NChannels> err;                           /// Generic error condition
  std::array<int16_t, NTimeBinsPerBC> data[NChannels] = {0}; /// Samples (raw or filtered)

  // Functions
  RecEventAux()
  {
    for (int32_t i = 0; i < NChannels; i++) {
      ref[i] = ZDCRefInitVal;
    }
  }

  void print() const;
  ClassDefNV(RecEventAux, 1);
};

} // namespace zdc
} // namespace o2

#endif
