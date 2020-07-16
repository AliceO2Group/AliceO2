// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Utilities for manipulating track parameters
///
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#ifndef O2_ANALYSIS_TRACKUTILITIES_H_
#define O2_ANALYSIS_TRACKUTILITIES_H_

template <typename T>
auto getTrackParCov(T track)
{
  std::array<float, 5> arraypar = {track.y(), track.z(), track.snp(),
                                   track.tgl(), track.signed1Pt()};
  std::array<float, 15> covpar = {track.cYY(), track.cZY(), track.cZZ(),
                                  track.cSnpY(), track.cSnpZ(),
                                  track.cSnpSnp(), track.cTglY(), track.cTglZ(),
                                  track.cTglSnp(), track.cTglTgl(),
                                  track.c1PtY(), track.c1PtZ(), track.c1PtSnp(),
                                  track.c1PtTgl(), track.c1Pt21Pt2()};
  return o2::track::TrackParCov(track.x(), track.alpha(), std::move(arraypar), std::move(covpar));
}

#endif // O2_ANALYSIS_TRACKUTILITIES_H_
