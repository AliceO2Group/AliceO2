// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ZDC_RECOPARAMZDC_H_
#define O2_ZDC_RECOPARAMZDC_H_

#include <array>
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
// parameters of ZDC reconstruction

struct RecoParamZDC : public o2::conf::ConfigurableParamHelper<RecoParamZDC> {
  Int_t tsh[NTDCChannels] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};                                                                                // Trigger shift
  Int_t tth[NTDCChannels] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8};                                                                                // Trigger threshold
  Int_t tmod[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};                                                                     // Position of TDC channel in raw data
  Int_t tch[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};                                                                      // Position of TDC channel in raw data
  Int_t amod[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};        // Position of ADC channel in raw data
  Int_t ach[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};         // Position of ADC  channel in raw data
  float tdc_shift[NTDCChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};                                                                // Correction of TDC position (0-25 ns, units of ~10 ps)
  float tdc_search[NTDCChannels] = {250, 250, 250, 250, 250, 250, 250, 250, 250, 250};                                                     // Search zone for a TDC signal ideally 2.5 ns (units of ~10 ps)
  Int_t beg_int[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};     // Start integration - signal
  Int_t end_int[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};     // End integration - signal
  Int_t beg_ped_int[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // Start integration - pedestal
  Int_t end_ped_int[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // End integration - pedestal
  std::array<bool, NTDCChannels> bitset = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};                                                                  // Set bits in coincidence
 public:
  void setBit(uint32_t ibit, bool val = true);
  O2ParamDef(RecoParamZDC, "RecoParamZDC");
};
} // namespace zdc
} // namespace o2

#endif
