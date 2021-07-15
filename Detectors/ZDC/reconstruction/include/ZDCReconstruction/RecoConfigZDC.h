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
  Int_t tsh[NTDCChannels] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};               // Trigger shift
  Int_t tth[NTDCChannels] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8};               // Trigger threshold
  std::array<bool, NTDCChannels> bitset = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Set bits in coincidence
  void setBit(uint32_t ibit, bool val = true);

  // TDC
  int tdc_search[NTDCChannels] = {250, 250, 250, 250, 250, 250, 250, 250, 250, 250}; // Search zone for a TDC signal ideally 2.5 ns (units of ~10 ps)
  void setSearch(uint32_t ich, int val);
  int getSearch(uint32_t ich) const;

  // Charge integration
  // Beginning and end of integration range: signal
  Int_t beg_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  Int_t end_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  // Beginning and end of integration range: pedestal
  Int_t beg_ped_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  Int_t end_ped_int[NChannels] = {DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange, DummyIntRange};
  void setIntegration(uint32_t ich, int beg, int end, int beg_ped, int end_ped);

  void print();

  ClassDefNV(RecoConfigZDC, 1);
};
} // namespace zdc
} // namespace o2

#endif
