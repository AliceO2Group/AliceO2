// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackWriterSpec.h

#ifndef O2_ITS_TRACKWRITER
#define O2_ITS_TRACKWRITER

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace its
{

/// create a processor spec
/// write ITS tracks to ROOT file
o2::framework::DataProcessorSpec getTrackWriterSpec(bool useMC);

} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKWRITER */
