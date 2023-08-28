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

#ifndef O2_ITS_WORKFLOW_DCS_DATA_GENERATOR_SPEC_H
#define O2_ITS_WORKFLOW_DCS_DATA_GENERATOR_SPEC_H

#include "DetectorsDCS/DCSDataPointHint.h"
#include "Framework/DataProcessorSpec.h"
#include <variant>
#include <string>
#include <vector>
#include <cstdint>

using namespace o2::framework;

namespace o2::its
{
o2::framework::DataProcessorSpec getITSDCSDataGeneratorSpec(const char* detName = "ITS");
} // namespace o2::its

#endif
