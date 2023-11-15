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

/// \file TPCZSLinkMapping.cxx
/// \author Felix Weiglhofer

#include "TPCZSLinkMapping.h"
#include "TPCBase/Mapper.h"

#include <algorithm>
#include <cassert>

using namespace GPUCA_NAMESPACE::gpu;

TPCZSLinkMapping::TPCZSLinkMapping(o2::tpc::Mapper& mapper)
{
#ifdef GPUCA_TPC_GEOMETRY_O2
  const auto& fecToGlobalPad = mapper.getMapFECIDGlobalPad();
  assert(fecToGlobalPad.size() == TPC_FEC_IDS_IN_SECTOR);

  const auto& globalPadToPadPos = mapper.getMapGlobalPadToPadPos();
  assert(globalPadToPadPos.size() == TPC_PADS_IN_SECTOR);

  for (size_t i = 0; i < TPC_FEC_IDS_IN_SECTOR; i++) {
    FECIDToPadPos[i] = globalPadToPadPos[fecToGlobalPad[i]];
  }
#endif
}
