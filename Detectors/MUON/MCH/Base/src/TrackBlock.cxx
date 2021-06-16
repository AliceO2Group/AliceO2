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

/// \file TrackBlock.cxx
/// \brief Implementation of the MCH track parameters minimal structure
///
/// \author Philippe Pillot, Subatech

#include "MCHBase/TrackBlock.h"

namespace o2
{
namespace mch
{

std::ostream& operator<<(std::ostream& stream, const TrackParamStruct& trackParam)
{
  auto oldflags = stream.flags();
  stream << "{x = " << trackParam.x << ", y = " << trackParam.y << ", z = " << trackParam.z
         << ", px = " << trackParam.px << ", py = " << trackParam.py << ", pz = " << trackParam.pz
         << ", sign = " << trackParam.sign << "}";
  stream.flags(oldflags);
  return stream;
}

} // namespace mch
} // namespace o2
