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

/// \file   FVDBaseParam.g
/// \brief  General constants in FVD
///
/// \author Maciej Slupecki, University of Jyvaskyla, Finland

#ifndef ALICEO2_FVD_FVDBASEPARAM_
#define ALICEO2_FVD_FVDBASEPARAM_

#include "FVDBase/GeometryTGeo.h"

namespace o2
{
namespace fvd
{

struct FVDBaseParam {
  static constexpr int nFvdChannels = GeometryTGeo::getNumberOfReadoutChannels();
  static constexpr int nFvdChannelsPlusRef = nFvdChannels + 1;
};

} // namespace fvd
} // namespace o2
#endif
