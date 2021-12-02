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

/// @file   definitions.h
/// @author Michael Lettrich
/// @since  2021-10-22
/// @brief  Global Definitions used throughout rANS library

#ifndef INCLUDE_RANS_DEFINITIONS_H_
#define INCLUDE_RANS_DEFINITIONS_H_

#include <cstdint>
#include <vector>

namespace o2
{
namespace rans
{
using symbol_t = int32_t;
using count_t = uint32_t;
using histogram_t = std::vector<count_t>;
} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_DEFINITIONS_H_ */
