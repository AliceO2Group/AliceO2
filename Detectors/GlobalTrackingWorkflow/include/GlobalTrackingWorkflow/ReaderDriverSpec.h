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

/// @file   ReaderDriverSpec.h

#ifndef O2_READER_DRIVER_
#define O2_READER_DRIVER_

#include "Framework/DataProcessorSpec.h"
#include <string>

namespace o2
{
namespace globaltracking
{

/// create a processor spec
/// pushes an empty output to provide timing to downstream devices
framework::DataProcessorSpec getReaderDriverSpec(const std::string& metricChannel = "", size_t minSHM = 0);

} // namespace globaltracking
} // namespace o2

#endif
