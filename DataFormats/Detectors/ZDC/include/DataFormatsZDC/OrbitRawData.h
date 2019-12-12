// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _ZDC_ORBITRAWDATA_H_
#define _ZDC_ORBITRAWDATA_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>

/// \file OrbitRawData.h
/// \brief Class to describe ZDC scalers and pedestals per orbit, received from the FE
/// \author cortese@to.infn.it, ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{

struct OrbitRawData {
  o2::InteractionRecord ir;
  std::array<uint16_t, MaxTriggerChannels> scalersFE; // FE scalers per triggering channel for given orbit
  std::array<int32_t, NChannels> pedestals;           // pedestal (* nSamples) per channel for given orbit
  uint16_t nSamples;                                  // N samples conributing to pedestal

  float getPedestal(int ch) const { return nSamples ? float(pedestals[ch]) / nSamples : 0.f; }

  void print() const;

  ClassDefNV(OrbitRawData, 1);
};
} // namespace zdc
} // namespace o2

#endif
