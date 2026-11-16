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

#ifndef O2_TRK_ALMIRAPARAM_H
#define O2_TRK_ALMIRAPARAM_H

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "CommonConstants/LHCConstants.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "TRKBase/Specs.h"

namespace o2
{
namespace trk
{

struct AlmiraParam : public o2::conf::ConfigurableParamHelper<AlmiraParam> {
  // This should be part of geometryTGeo
  static constexpr size_t kNLayers = constants::VD::petal::nLayers + constants::ML::nLayers + constants::OT::nLayers + constants::MLOTDisks::nLayers;
  static constexpr size_t getNLayers() { return kNLayers; }

  static constexpr int kDefROFLengthInBC = 16; ///< default ROF length in BC (16 clock cycles of 1 BC for VD)

  int roFrameLengthInBCPerLayer[kNLayers] = {0};  ///< ROF length in BC per layer
  float strobeDelayPerLayer[kNLayers] = {0};      ///< strobe delay in ns per layer
  float strobeLengthContPerLayer[kNLayers] = {0}; ///< strobe length in ns per layer
  int roFrameBiasInBCPerLayer[kNLayers] = {0};    ///< ROF start bias in BC per layer
  int roFrameDelayInBCPerLayer[kNLayers] = {0};   ///< extra ROF delay in BC per layer
  float clockPeriodPerLayer[kNLayers] = {0};      ///< ALMIRA clock period in ns per layer (timestamp fine granularity), 0 = default (1 BC VD, 4 BC ML/OT/disks)
  int nTimestampBits = 12;                        ///< bits of the digit timestamp: [frame (coarse) | clock cycle in ROF (fine)]
  int nFineBits = 4;                              ///< bits of the fine part (clock cycle in ROF); the coarse part (frame: free-running ROF counter, rolling over at 2^(nTimestampBits - nFineBits)) has nTimestampBits - nFineBits

  float getClockPeriod(int layer) const
  {
    if (clockPeriodPerLayer[layer] > 0) {
      return clockPeriodPerLayer[layer];
    }
    return (layer < constants::VD::petal::nLayers ? 1 : 4) * o2::constants::lhc::LHCBunchSpacingNS; // the clock is synchronous with the LHC: 1 BC (VD), 4 BC (ML/OT/disks)
  }

  /// number of clock cycles in a ROF; if the ROF is not a multiple of the clock period, the last cycle is shorter
  /// (e.g. ROF 18 BC, clock 4 BC -> 4.5 -> 5 cycles). Tolerance for the float rounding: 18 BC / 24.95 ns must give 18, not 19
  static int computeNClockCycles(int rofLengthInBC, float clockPeriod)
  {
    return std::ceil(rofLengthInBC * o2::constants::lhc::LHCBunchSpacingNS / clockPeriod - 1e-3); // - 1e-3: tolerance for float rounding
  }
  int getNClockCyclesPerROF(int layer) const { return computeNClockCycles(getROFLengthInBC(layer), getClockPeriod(layer)); }

  /// timestamp [frame | clock cycle in ROF] of a signal with time toa (ns) w.r.t. the start of the ROF with index rofIndex
  /// (ROFs counted continuously by the digitizer from the first sampled IR). The frame is a free-running ROF counter which
  /// rolls over at 2^(nTimestampBits - nFineBits). toa < 0 means that the signal started in the previous ROF and is still
  /// above threshold in this one: the timestamp refers to the ROF where the signal started.
  /// TODO: once the edge detection is implemented, check this
  uint16_t encodeTimestamp(float toa, long rofIndex, int rofLengthInBC, float clockPeriod, int nClockCycles) const
  {
    const float rofNS = rofLengthInBC * o2::constants::lhc::LHCBunchSpacingNS;
    const int shift = std::floor(toa / rofNS); // 0, or -1 if the signal started in the previous ROF
    const float tInROF = toa - shift * rofNS;  // in [0, rofNS)
    const int clockCycle = std::min<int>(tInROF / clockPeriod, nClockCycles - 1);
    const long frame = (rofIndex + shift) & ((1L << (nTimestampBits - nFineBits)) - 1); // rollover (ROF -1 gets the last frame)
    return uint16_t(frame << nFineBits | clockCycle);
  }

  int getFrame(uint16_t ts) const { return ts >> nFineBits; }
  int getClockCycle(uint16_t ts) const { return ts & ((1 << nFineBits) - 1); }

  int getROFLengthInBC(int layer) const
  {
    if (roFrameLengthInBCPerLayer[layer] > 0) {
      return roFrameLengthInBCPerLayer[layer];
    } else {
      return kDefROFLengthInBC;
    }
  }
  float getStrobeDelay(int layer) const { return strobeDelayPerLayer[layer]; }
  float getStrobeLengthCont(int layer) const { return strobeLengthContPerLayer[layer]; }
  int getROFBiasInBC(int layer) const { return roFrameBiasInBCPerLayer[layer]; }
  int getROFDelayInBC(int layer) const { return roFrameDelayInBCPerLayer[layer]; }

  O2ParamDef(AlmiraParam, "TRKAlmiraParam");
};

} // namespace trk

namespace framework
{
template <typename T>
struct is_messageable;

template <>
struct is_messageable<o2::trk::AlmiraParam> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif
