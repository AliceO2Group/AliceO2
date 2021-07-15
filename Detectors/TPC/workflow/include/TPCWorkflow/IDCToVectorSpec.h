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

/// @file   IDCToVectorSpec.h
/// @author Jens Wiechula
/// @brief  Processor to convert IDCs to a vector for each pad in a CRU

#ifndef TPC_IDCToVectorSpec_H_
#define TPC_IDCToVectorSpec_H_

#include "Framework/DataProcessorSpec.h"
#include <string_view>

namespace o2::tpc
{

/// create a processor spec
/// convert IDC raw values to a vector for each pad in a CRU
o2::framework::DataProcessorSpec getIDCToVectorSpec(const std::string inputSpec, std::vector<uint32_t> const& crus);

} // end namespace o2::tpc

#endif //TPC_IDCToVectorSpec_H_
