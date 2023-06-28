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

#ifndef O2_TRD_TRACKBASEDCALIBSPEC_H
#define O2_TRD_TRACKBASEDCALIBSPEC_H

/// \file   TrackBasedCalibSpec.h
/// \brief Steers the creation of calibration input based on tracks
/// \author Ole Schmidt

// input TRD tracks, TRD tracklets, TRD calibrated tracklets
// output AngularResidHistos object

#include "Framework/DataProcessorSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{
/// create a processor spec
framework::DataProcessorSpec getTRDTrackBasedCalibSpec(o2::dataformats::GlobalTrackID::mask_t src, bool vdexb, bool gain);

} // namespace trd
} // namespace o2

#endif // O2_TRD_TRACKBASEDCALIBSPEC_H
