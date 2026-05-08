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

/// @file   CMVToVectorSpec.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  Processor to convert CMVs to a vector in a CRU

#ifndef TPC_CMVToVectorSpec_H_
#define TPC_CMVToVectorSpec_H_

#include "Framework/DataProcessorSpec.h"

namespace o2::tpc
{

/// create a processor spec
/// convert CMV raw values to a vector in a CRU
o2::framework::DataProcessorSpec getCMVToVectorSpec(const std::string inputSpec, std::vector<uint32_t> const& crus);

} // end namespace o2::tpc

#endif // TPC_CMVToVectorSpec_H_
