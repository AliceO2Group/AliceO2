// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackerParam.h
/// \brief Configurable parameters for MCH tracking
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_TRACKERPARAM_H_
#define ALICEO2_MCH_TRACKERPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mch
{

/// Configurable parameters for MCH tracking
struct TrackerParam : public o2::conf::ConfigurableParamHelper<TrackerParam> {

  double chamberResolutionX = 0.2; ///< chamber resolution (cm) in x used as cluster resolution during tracking
  double chamberResolutionY = 0.2; ///< chamber resolution (cm) in y used as cluster resolution during tracking

  double sigmaCutForTracking = 5.;    ///< to select clusters (local chi2) and tracks (global chi2) during tracking
  double sigmaCutForImprovement = 4.; ///< to select clusters (local chi2) and tracks (global chi2) during improvement

  double nonBendingVertexDispersion = 70.; ///< vertex dispersion (cm) in non bending plane
  double bendingVertexDispersion = 70.;    ///< vertex dispersion (cm) in bending plane

  /// if true, at least one cluster in the station is requested to validate the track
  bool requestStation[5] = {true, true, true, true, true};

  bool moreCandidates = false; ///< find more track candidates starting from 1 cluster in each of station (1..) 4 and 5
  bool refineTracks = true;    ///< refine the tracks in the end using cluster resolution

  O2ParamDef(TrackerParam, "MCHTracking");
};

} // namespace mch
} // end namespace o2

#endif // ALICEO2_MCH_TRACKERPARAM_H_
