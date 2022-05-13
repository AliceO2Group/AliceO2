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
  Int_t tsh[NTDCChannels] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};   // Trigger shift
  Int_t tth[NTDCChannels] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8};   // Trigger threshold
  bool bitset[NTDCChannels] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Set bits in coincidence
  void setBit(uint32_t ibit, bool val = true);

  // Signal processing
  int low_pass_filter = -1;    // Low pass filtering
  int full_interpolation = -1; // Full interpolation of waveform
  int corr_signal = -1;        // TDC signal correction
  int corr_background = -1;    // TDC pile-up correction

  int debug_output = -1; // Debug output

  // TDC
  Int_t tmod[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};       // Position of TDC channel in raw data
  Int_t tch[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};        // Position of TDC channel in raw data
  float tdc_shift[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};  // Correction of TDC position (0-25 ns, units of ~10 ps)
  float tdc_calib[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};  // Correction of TDC amplitude
  float tdc_search[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // Search zone for a TDC signal ideally 2.5 ns (units of ~10 ps)

  // Charge integration
  Int_t amod[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // Position of ADC channel in raw data
  Int_t ach[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};  // Position of ADC  channel in raw data
  // Beginning and end of integration range: signal
  Int_t beg_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  Int_t end_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  // Beginning and end of integration range: pedestal
  Int_t beg_ped_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  Int_t end_ped_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  // Pedestal thresholds for pile-up detection
  float ped_thr_hi[NChannels] = {ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange};
  float ped_thr_lo[NChannels] = {ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange};

  // Energy calibration
  float energy_calib[NChannels] = {0}; // Energy calibration coefficients
  float tower_calib[NChannels] = {0};  // Tower calibration coefficients

  void print();

  O2ParamDef(RecoParamZDC, "RecoParamZDC");
};
} // namespace zdc
} // namespace o2

#endif
