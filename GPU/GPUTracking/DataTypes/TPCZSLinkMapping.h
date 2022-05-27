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

/// \file TPCZSLinkMapping.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_TPC_ZS_LINK_MAPPING_H
#define O2_GPU_TPC_ZS_LINK_MAPPING_H

#include "clusterFinderDefs.h"
#include "TPCBase/PadPos.h"

namespace o2::tpc
{
class Mapper;
}

namespace GPUCA_NAMESPACE::gpu
{

struct TPCZSLinkMapping {
#ifndef GPUCA_GPUCODE
  TPCZSLinkMapping() = default;
  TPCZSLinkMapping(o2::tpc::Mapper& mapper);
#endif

  o2::tpc::PadPos FECIDToPadPos[TPC_FEC_IDS_IN_SECTOR];
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
