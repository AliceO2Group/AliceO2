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

/// \file TrackFinderOriginalSpec.h
/// \brief Definition of a data processor to read clusters, reconstruct tracks and send them
///
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_TRACKFINDERORIGINALSPEC_H_
#define O2_MCH_TRACKFINDERORIGINALSPEC_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace mch
{

o2::framework::DataProcessorSpec getTrackFinderOriginalSpec(const char* specName = "mch-track-finder-original",
                                                            bool computeTime = true, bool digits = false,
                                                            bool disableCCDBMagField = false);

} // end namespace mch
} // end namespace o2

#endif // O2_MCH_TRACKFINDERORIGINALSPEC_H_
