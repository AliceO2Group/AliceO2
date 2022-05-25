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

#include "Framework/Logger.h"
#include "ZDCReconstruction/RecoParamZDC.h"

O2ParamImpl(o2::zdc::RecoParamZDC);

void o2::zdc::RecoParamZDC::setBit(uint32_t ibit, bool val)
{
  if (ibit >= 0 && ibit < NTDCChannels) {
    bitset[ibit] = val;
  } else {
    LOG(fatal) << __func__ << " bit " << ibit << " not in allowed range";
  }
}

void o2::zdc::RecoParamZDC::print()
{
  bool printed = false;
  if (low_pass_filter >= 0 || full_interpolation >= 0 || corr_signal >= 0 || corr_background >= 0) {
    if (!printed) {
      LOG(info) << "RecoParamZDC::print()";
      printed = true;
    }
    if (low_pass_filter >= 0) {
      printf(" LowPassFilter=%d", low_pass_filter);
    }
    if (full_interpolation >= 0) {
      printf(" FullInterpolation=%d", full_interpolation);
    }
    if (corr_signal >= 0) {
      printf(" CorrSignal=%d", corr_signal);
    }
    if (corr_background >= 0) {
      printf(" CorrBackground=%d", corr_background);
    }
    printf("\n");
  }
  if (!printed) {
    printed = true;
    LOG(info) << "RecoParamZDC::print()";
  }
  /*
    Int_t tsh[NTDCChannels] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};   // Trigger shift
    Int_t tth[NTDCChannels] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8};   // Trigger threshold
    bool bitset[NTDCChannels] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Set bits in coincidence
    void setBit(uint32_t ibit, bool val = true);

    // Signal processing
    int low_pass_filter = -1;    // Low pass filtering
    int full_interpolation = -1; // Full interpolation of waveform
    int corr_signal = -1;        // TDC signal correction
    int corr_background = -1;    // TDC pile-up correction

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
  */
  bool modified = false;
  for (int i = 0; i < o2::zdc::NChannels; i++) {
    if (energy_calib[i] != 0) {
      modified = true;
      break;
    }
  }
  if (modified) {
    if (!printed) {
      printed = true;
      LOG(info) << "RecoParamZDC::print()";
    }
    printf("energ_calib: ");
    for (int i = 0; i < o2::zdc::NChannels; i++) {
      if (energy_calib[i] != 0) {
        printf(" %s=%f", o2::zdc::ChannelNames[i].data(), energy_calib[i]);
      }
    }
    printf("\n");
  }
  modified = false;
  for (int i = 0; i < o2::zdc::NChannels; i++) {
    if (tower_calib[i] != 0) {
      modified = true;
      break;
    }
  }
  if (modified) {
    if (!printed) {
      printed = true;
      LOG(info) << "RecoParamZDC::print()";
    }
    printf("tower_calib: ");
    for (int i = 0; i < o2::zdc::NChannels; i++) {
      if (tower_calib[i] != 0) {
        printf(" %s=%f", o2::zdc::ChannelNames[i].data(), tower_calib[i]);
      }
    }
    printf("\n");
  }
}
