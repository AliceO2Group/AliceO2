// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Track.h
/// \brief Base track model for the Barrel, params only, w/o covariance
/// \author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch

#ifndef ALICEO2_BASE_TRACK
#define ALICEO2_BASE_TRACK

#include "ReconstructionDataFormats/TrackParametrization.h"
#include "ReconstructionDataFormats/TrackParametrizationWithError.h"

namespace o2
{
namespace track
{

using TrackParF = TrackParametrization<float>;
using TrackParD = TrackParametrization<double>;
using TrackPar = TrackParF;

using TrackParCovF = TrackParametrizationWithError<float>;
using TrackParCovD = TrackParametrizationWithError<double>;
using TrackParCov = TrackParCovF;

} // namespace track
} // namespace o2

#endif
