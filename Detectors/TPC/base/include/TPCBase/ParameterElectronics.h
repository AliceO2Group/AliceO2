// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace tpc
{

enum class DigitzationMode : char {
  FullMode = 0,         ///< Apply noise, pedestal and saturation
  SubtractPedestal = 1, ///< Apply noise, pedestal and saturation and then from that subtract the pedestal
  NoSaturation = 2,     ///< Apply only noise and pedestal
  PropagateADC = 3      ///< Just propagate the bare ADC value
};

struct ParameterElectronics : public o2::conf::ConfigurableParamHelper<ParameterElectronics> {

  int NShapedPoints = 8;                                        ///< Number of ADC samples which are taken into account for a given, shaped signal (should fit
                                                                /// into SSE registers)
  float PeakingTime = 160e-3f;                                  ///< Peaking time of the SAMPA [us]
  float ChipGain = 20.f;                                        ///< Gain of the SAMPA [mV/fC] - may be either 20 or 30
  float ADCdynamicRange = 2200.f;                               ///< Dynamic range of the ADC [mV]
  float ADCsaturation = 1024.f;                                 ///< ADC saturation [ADC counts]
  float ZbinWidth = 0.2;                                        ///< Width of a z bin [us]
  float ElectronCharge = 1.602e-19f;                            ///< Electron charge [C]
  DigitzationMode DigiMode = DigitzationMode::SubtractPedestal; ///< Digitization mode [full / ... ]

  O2ParamDef(ParameterElectronics, "TPCEleParam");
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_ParameterElectronics_H_
