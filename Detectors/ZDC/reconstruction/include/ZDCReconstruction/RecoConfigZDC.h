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

#ifndef O2_ZDC_RECOCONFIGZDC_H
#define O2_ZDC_RECOCONFIGZDC_H

#include <array>
#include <Rtypes.h>
#include "ZDCBase/Constants.h"

/// \file RecoConfigZDC.h
/// \brief ZDC reconstruction parameters
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct RecoConfigZDC {
  // Trigger
  int32_t tsh[NTDCChannels] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};             // Trigger shift
  int32_t tth[NTDCChannels] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8};             // Trigger threshold
  std::array<bool, NTDCChannels> bitset = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Set bits in coincidence
  void setBit(uint32_t ibit, bool val = true);
  uint8_t triggerCondition = 0x3; /// Trigger condition: 0x1 single, 0x3 double and 0x7 triple
  uint8_t getTriggerCondition() { return triggerCondition; }
  void setTripleTrigger() { triggerCondition = 0x7; }
  void setDoubleTrigger() { triggerCondition = 0x3; }
  void setSingleTrigger() { triggerCondition = 0x1; }

  // Signal processing
  bool low_pass_filter = true;     // Low pass filtering
  bool full_interpolation = false; // Full interpolation of waveform
  bool corr_signal = true;         // TDC signal correction
  bool corr_background = true;     // TDC pile-up correction

  // TDC
  int tdc_search[NTDCChannels] = {250, 250, 250, 250, 250, 250, 250, 250, 250, 250}; // Search zone for a TDC signal ideally 2.5 ns (units of ~10 ps)
  void setSearch(uint32_t ich, int val);
  int getSearch(uint32_t ich) const;
  bool extendedSearch = false; // Extend search at beginning of window (needs orbit pedestal info)
  bool storeEvPileup = false;  // Store TDC hits with in-event pile-up

  // Charge integration
  // Beginning and end of integration range: signal
  int32_t beg_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  int32_t end_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  // Beginning and end of integration range: pedestal
  int32_t beg_ped_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  int32_t end_ped_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  void setIntegration(uint32_t ich, int beg, int end, int beg_ped, int end_ped);
  // Pedestal thresholds for pile-up detection
  // Default value ADCRange will never allow to revert to orbit pedestal and will never identify pile-up
  // Values <=0 will identify all events as pile-up and use always orbit pedestal
  float ped_thr_hi[NChannels] = {ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange};
  float ped_thr_lo[NChannels] = {ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange, ADCRange};
  void setPedThreshold(int32_t ich, float high, float low);

  void print() const;

  ClassDefNV(RecoConfigZDC, 4);
};
} // namespace zdc
} // namespace o2

#endif
