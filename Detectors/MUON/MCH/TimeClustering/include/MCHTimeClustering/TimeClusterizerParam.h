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

#ifndef O2_MCH_TIMECLUSTERING_TIMECLUSTERIZER_PARAM_H_
#define O2_MCH_TIMECLUSTERING_TIMECLUSTERIZER_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::mch
{

/**
 * @class TimeClusterizerParam
 * @brief Configurable parameters for the time clustering
 */
struct TimeClusterizerParam : public o2::conf::ConfigurableParamHelper<TimeClusterizerParam> {

  bool onlyTrackable = true; ///< only output ROFs that match the trackable condition @see MCHROFFiltering/TrackableFilter

  int maxClusterWidth = 1000 / 25;  ///< maximum time width of time clusters, in BC units
  int peakSearchNbins = 5;          ///< number of time bins for the peak search algorithm (must be an odd number >= 3)
  int minDigitsPerROF = 0;          ///< minimum number of digits per ROF (below that threshold ROF is discarded)
  bool peakSearchSignalOnly = true; ///< only use signal-like hits in peak search
  bool irFramesOnly = false;        ///< only output ROFs that overlap one of the IRFrames (provided externally, e.g. by ITS) @see MCHROFFiltering/IRFrameFilter

  O2ParamDef(TimeClusterizerParam, "MCHTimeClusterizer");
};

} // namespace o2::mch

#endif
