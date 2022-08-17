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

/// \file ParameterElectronics.h
/// \brief Definition of the parameter class for the detector electronics
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_ParameterElectronics_H_
#define ALICEO2_TPC_ParameterElectronics_H_

#include <array>
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "CommonConstants/LHCConstants.h"

namespace o2
{
namespace tpc
{

enum class DigitzationMode : char {
  FullMode = 0,         ///< Apply noise, pedestal and saturation
  ZeroSuppression = 1,  ///< Apply noise, pedestal and saturation and then from that subtract the pedestal
  SubtractPedestal = 2, ///< Apply noise, pedestal and saturation and then from that subtract the pedestal
  NoSaturation = 3,     ///< Apply only noise and pedestal
  PropagateADC = 4      ///< Just propagate the bare ADC value
};

struct ParameterElectronics : public o2::conf::ConfigurableParamHelper<ParameterElectronics> {
  static constexpr int TIMEBININBC = 8;

  int NShapedPoints = 8;                                                        ///< Number of ADC samples which are taken into account for a given, shaped signal (should fit
                                                                                /// into SSE registers)
  float PeakingTime = 160e-3f;                                                  ///< Peaking time of the SAMPA [us]
  float ChipGain = 20.f;                                                        ///< Gain of the SAMPA [mV/fC] - may be either 20 or 30
  float ADCdynamicRange = 2200.f;                                               ///< Dynamic range of the ADC [mV]
  float ADCsaturation = 1024.f;                                                 ///< ADC saturation [ADC counts]
  float ZbinWidth = TIMEBININBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; ///< Width of a z bin [us]
  float ElectronCharge = 1.602e-19f;                                            ///< Electron charge [C]
  float adcToT = 1.f / 1024.f;                                                  ///< relation between time over threshold and ADC value
  bool doIonTail = false;                                                       ///< add ion tail in simulation
  bool doSaturationTail = false;                                                ///< add saturation tail in simulation
  bool doNoiseEmptyPads = false;                                                ///< add noise in pads without signal in simulation
  DigitzationMode DigiMode = DigitzationMode::ZeroSuppression;                  ///< Digitization mode [full / ... ]

  /// Average time from the start of the signal shaping to the COG of the sampled distribution
  ///
  /// Since the shaping of the electronic starts when the signals arrive, the cluster finder
  /// reconstructs signals later in time. Due to the asymmetry of the signal, the average
  /// shaping time is somewhat larger then the PeakingTime
  float getAverageShapingTime() const { return PeakingTime + 35e-3f; }

  O2ParamDef(ParameterElectronics, "TPCEleParam");
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_ParameterElectronics_H_
