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

#ifndef O2_MCH_EVALUATION_COMPARE_TRACKS_H__
#define O2_MCH_EVALUATION_COMPARE_TRACKS_H__

#include "MCHEvaluation/ExtendedTrack.h"
#include <TMatrixDfwd.h>
#include <array>
#include <list>
#include <vector>

class TH1;

namespace o2::mch::eval
{
int compareEvents(std::list<ExtendedTrack>& tracks1, std::list<ExtendedTrack>& tracks2,
                  double precision, bool printDiff, bool printAll,
                  std::vector<TH1*>& trackResidualsAtFirstCluster,
                  std::vector<TH1*>& clusterClusterResiduals);

bool areCompatible(const TrackParam& param1, const TrackParam& param2, double precision);

bool areCompatible(const TMatrixD& cov1, const TMatrixD& cov2, double precision);

void selectTracks(std::list<ExtendedTrack>& tracks);
} // namespace o2::mch::eval

#endif
