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

/// @file   IRFramesReaderSpec.h

#ifndef O2_IRFRAMES_READER
#define O2_IRFRAMES_READER

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace globaltracking
{

/// create a processor spec
/// read IRFrames data from a root file
framework::DataProcessorSpec getIRFrameReaderSpec(o2::header::DataOrigin origin, uint32_t subSpec, const std::string& devName, const std::string& defFileName);

} // namespace globaltracking
} // namespace o2

#endif
