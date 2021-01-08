// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFinderOriginalSpec.h
/// \brief Definition of a data processor to read clusters, reconstruct tracks and send them
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_TRACKFINDERORIGINALSPEC_H_
#define ALICEO2_MCH_TRACKFINDERORIGINALSPEC_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace mch
{

o2::framework::DataProcessorSpec getTrackFinderOriginalSpec();

} // end namespace mch
} // end namespace o2

#endif // ALICEO2_MCH_TRACKFINDERORIGINALSPEC_H_
