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
