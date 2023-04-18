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

/// \file   MIDWorkflow/FilteringBCSpec.h
/// \brief  MID filtering spec
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   16 March 2022

#ifndef O2_MID_FILTERINGBCSPEC_H
#define O2_MID_FILTERINGBCSPEC_H

#include "Framework/DataProcessorSpec.h"

#include <string_view>

namespace o2
{
namespace mid
{
framework::DataProcessorSpec getFilteringBCSpec(bool useMC = true, std::string_view inDesc = "FDATA");
}
} // namespace o2

#endif // O2_MID_FILTERINGBCSPEC_H
