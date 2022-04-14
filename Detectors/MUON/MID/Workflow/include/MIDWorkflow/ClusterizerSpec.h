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

/// \file   MIDWorkflow/ClusterizerSpec.h
/// \brief  Data processor specs for MID clustering device
/// \author Gabriele G. Fronze <gfronze at cern.ch>
/// \date   9 July 2018

#ifndef O2_MID_CLUSTERIZERSPEC_H
#define O2_MID_CLUSTERIZERSPEC_H

#include "Framework/DataProcessorSpec.h"
#include <string_view>

namespace o2
{
namespace mid
{
framework::DataProcessorSpec getClusterizerSpec(bool isMC, std::string_view inDataDesc, std::string_view inRofDesc, std::string_view inLabelsDesc);
}
} // namespace o2

#endif // O2_MID_CLUSTERIZERSPEC_H
