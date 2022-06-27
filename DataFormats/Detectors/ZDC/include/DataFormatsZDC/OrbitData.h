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

#ifndef _ZDC_ORBIT_DATA_H_
#define _ZDC_ORBIT_DATA_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "ZDCBase/Constants.h"
#include <array>
#include <Rtypes.h>

/// \file OrbitData.h
/// \brief Class to describe pedestal data accumulated over the orbit
/// \author ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{

struct OrbitData {
  o2::InteractionRecord ir;
  std::array<int16_t, NChannels> data{};
  std::array<uint16_t, NChannels> scaler{};

  void print() const;

  ClassDefNV(OrbitData, 1);
};
} // namespace zdc
} // namespace o2

#endif
