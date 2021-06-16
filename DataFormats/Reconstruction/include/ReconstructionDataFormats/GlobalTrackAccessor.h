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

/// @file  GlobalTrackAccessor.h
/// \brief Accessor for TrackParCov derived objects from multiple containers
/// \author ruben.shahoyan@cern.ch

#ifndef O2_GLOBAL_TRACK_ACCESSOR
#define O2_GLOBAL_TRACK_ACCESSOR

#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/AbstractRefAccessor.h"

namespace o2
{
namespace dataformats
{
using GlobalTrackAccessor = AbstractRefAccessor<o2::track::TrackParCov, GlobalTrackID::NSources>;
}
} // namespace o2

#endif
