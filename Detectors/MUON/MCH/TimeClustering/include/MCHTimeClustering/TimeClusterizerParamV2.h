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

#ifndef O2_MCH_TIMECLUSTERING_TIMECLUSTERIZER_PARAM_V2_H_
#define O2_MCH_TIMECLUSTERING_TIMECLUSTERIZER_PARAM_V2_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::mch
{

/**
 * @class TimeClusterizerParamV2
 * @brief Configurable parameters for the time clustering
 */
struct TimeClusterizerParamV2 : public o2::conf::ConfigurableParamHelper<TimeClusterizerParamV2> {

  bool onlyTrackable = true; ///< only output ROFs that match the trackable condition @see MCHROFFiltering/TrackableFilter

  int maxClusterWidth = 1000 / 25; ///< initial width of time clusters, in BC units (default 1us)
  int peakSearchWindow = 20;        ///< width of the peak search window, in BC units
  int peakSearchNbins = 5;          ///< number of time bins for the peak search algorithm (must be an odd number >= 3)
  int peakSearchNDigitsMin = 10;    ///< minimum number of digits for peak candidates
  bool peakSearchSignalOnly = true; ///< only use signal-like hits in peak search
  bool mergeROFs = true;            ///< whether to merge consecutive ROFs
  int mergeGapMax = 11;             ///< maximum allowed gap between ROFs, above which no merging is done
  int minDigitsPerROF = 0;          ///< minimum number of digits per ROF (below that threshold ROF is discarded)
  bool irFramesOnly = false;        ///< only output ROFs that overlap one of the IRFrames (provided externally, e.g. by ITS) @see MCHROFFiltering/IRFrameFilter

  float rofRejectionFraction = 0; ///< fraction of output (i.e. time-clusterized) ROFs to discard. If 0 (default) keep them all. WARNING: use a non zero value only at Pt2 for sync reco, if needed.

  O2ParamDef(TimeClusterizerParamV2, "MCHTimeClusterizerV2");
};

} // namespace o2::mch

#endif
