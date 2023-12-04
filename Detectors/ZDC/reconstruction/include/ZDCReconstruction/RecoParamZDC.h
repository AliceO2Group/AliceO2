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

#ifndef O2_ZDC_RECOPARAMZDC_H
#define O2_ZDC_RECOPARAMZDC_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "ZDCBase/Constants.h"

/// \file RecoParamZDC.h
/// \brief ZDC reconstruction parameters
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct RecoParamZDC : public o2::conf::ConfigurableParamHelper<RecoParamZDC> {
  // Trigger
  int32_t tsh[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // Trigger shift
  int32_t tth[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // Trigger threshold
  bool bitset[NTDCChannels] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};           // Set bits in coincidence
  void setBit(uint32_t ibit, bool val = true);
  uint8_t triggerCondition = 0x0; // Trigger condition: 0x1 single, 0x3 double and 0x7 triple

  // Signal processing
  int low_pass_filter = -1;               // Low pass filtering
  int full_interpolation = -1;            // Full interpolation of waveform
  int full_interpolation_min_length = -1; // Minimum length to perform full interpolation
  int corr_signal = -1;                   // TDC signal correction
  int corr_background = -1;               // TDC pile-up correction

  int debug_output = -1; // Debug output

  // TDC
  int32_t tmod[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};     // Position of TDC channel in raw data
  int32_t tch[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};      // Position of TDC channel in raw data
  float tdc_shift[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};  // Correction of TDC position (0-25 ns, units of ~10 ps)
  float tdc_calib[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};  // Correction of TDC amplitude
  float tdc_search[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // Search zone for a TDC signal ideally 2.5 ns (units of ~10 ps)

  float tdc_offset[NTDCChannels] = {-FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty}; // Compensation of TDC amplitude offset

  // Enable extended search at beginning of first bunch
  bool setExtendedSearch = false;
  bool doExtendedSearch = false;

  // Store events with in-event pile-up
  bool setStoreEvPileup = false;
  bool doStoreEvPileup = false;

  // Charge integration
  int32_t amod[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // Position of ADC channel in raw data
  int32_t ach[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};  // Position of ADC  channel in raw data
  // Beginning and end of integration range: signal
  int32_t beg_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  int32_t end_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  // Beginning and end of integration range: pedestal
  int32_t beg_ped_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  int32_t end_ped_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  // Pedestal thresholds for pile-up detection
  float ped_thr_hi[NChannels] = {ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange};
  float ped_thr_lo[NChannels] = {ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange};

  float energy_calib[NChannels] = {0}; // Energy calibration coefficients
  // Compensation of ADC offset
  float adc_offset[NChannels] = {-FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty, -FInfty};
  float tower_calib[NChannels] = {0}; // Tower calibration coefficients

  void print();

  O2ParamDef(RecoParamZDC, "RecoParamZDC");
};
} // namespace zdc
} // namespace o2

#endif
