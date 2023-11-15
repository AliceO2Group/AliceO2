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

/// @file   StrangenessTrackingReaderSpec.h

#ifndef O2_STRANGENESS_TRACKING_READERSPEC
#define O2_STRANGENESS_TRACKING_READERSPEC

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace strangeness_tracking
{

/// create a processor spec
/// read secondary vertex data from a root file
o2::framework::DataProcessorSpec getStrangenessTrackingReaderSpec(bool useMC);

} // namespace strangeness_tracking
} // namespace o2

#endif
