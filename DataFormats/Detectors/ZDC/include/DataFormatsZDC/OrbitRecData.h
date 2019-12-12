// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _ZDC_ORBITRECDATA_H_
#define _ZDC_ORBITRECDATA_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>

/// \file OrbitRecData.h
/// \brief Class to describe ZDC scalers reconstructed from the channels data
/// \author cortese@to.infn.it, ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{

struct OrbitRecData {
  static constexpr int NScalerCombination = 7;
  o2::InteractionRecord ir;
  std::array<uint16_t, NChannels> scalers;              // computed scalers per channel for given orbit
  std::array<uint16_t, NScalerCombination> scalersComb; // combined channels scalers

  void print() const;

  ClassDefNV(OrbitRecData, 1);
};
} // namespace zdc
} // namespace o2

#endif
