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

/// \file StrangenessTrackingWriterSpec.h
/// \brief
///

#ifndef O2_STRANGENESSTRACKINGWRITER
#define O2_STRANGENESSTRACKINGWRITER

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace strangeness_tracking
{

/// create a processor spec
/// write ITS tracks to ROOT file
o2::framework::DataProcessorSpec getStrangenessTrackingWriterSpec();

} // namespace strangeness_tracking
} // namespace o2

#endif /* O2_STRANGENESSTRACKINGWRITER */
