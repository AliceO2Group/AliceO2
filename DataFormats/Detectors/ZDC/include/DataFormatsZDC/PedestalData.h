// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _ZDC_PEDESTAL_DATA_H_
#define _ZDC_PEDESTAL_DATA_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "ZDCBase/Constants.h"
#include <array>
#include <Rtypes.h>

/// \file Pedestal.h
/// \brief Class to describe pedestal data accumulated over the orbit
/// \author ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{

struct PedestalData {
  o2::InteractionRecord ir;
  std::array<int16_t, NChannels> data{};
  float asFloat(int i) const { return data[i] / 8.; }
  void print() const;

  ClassDefNV(PedestalData, 2);
};
} // namespace zdc
} // namespace o2

#endif
