// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ZDC_INTEGRATIONPARAM_H_
#define O2_ZDC_INTEGRATIONPARAM_H_

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>

/// \file ZDCIntegrationParam.h
/// \brief Parameters to configure integration
/// \author P. Cortese

namespace o2
{
namespace zdc
{
// parameters of ZDC reconstruction

struct ZDCIntegrationParam {
 public:
  Int_t beg_int[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};     // Start integration - signal
  Int_t end_int[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};     // End integration - signal
  Int_t beg_ped_int[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // Start integration - pedestal
  Int_t end_ped_int[NChannels] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // End integration - pedestal
  void setIntegration(uint32_t ich, int beg, int end, int beg_ped, int end_ped);
  void print();

  ClassDefNV(ZDCIntegrationParam, 1);
};
} // namespace zdc
} // namespace o2

#endif
