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

/// \file   MIDWorkflow/RawCheckerSpec.h
/// \brief  Data processor spec for MID raw checker device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   06 April 2020

#ifndef O2_MID_RAWCHECKERSPEC_H
#define O2_MID_RAWCHECKERSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"

#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/ElectronicsDelay.h"

namespace o2
{
namespace mid
{
framework::DataProcessorSpec getRawCheckerSpec(const std::vector<uint16_t>& feeIds, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay, bool perGBT = false);
} // namespace mid
} // namespace o2

#endif //O2_MID_RAWCHECKERSPEC_H
