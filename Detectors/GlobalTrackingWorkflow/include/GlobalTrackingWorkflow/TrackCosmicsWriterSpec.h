// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackCosmicsWriterSpec.h

#ifndef O2_TRACK_COSMICS_WRITER
#define O2_TRACK_COSMICS_WRITER

#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

/// create a processor spec
/// write cosmics tracks to a root file
framework::DataProcessorSpec getTrackCosmicsWriterSpec(bool useMC);

} // namespace globaltracking
} // namespace o2

#endif /* O2_TRACK_COSMICS_WRITER */
