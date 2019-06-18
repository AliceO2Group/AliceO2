// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/TestingSimTools/src/TrackGenerator.cxx
/// \brief  Implementation of a fast track generator for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 December 2017

#include "MIDTestingSimTools/TrackGenerator.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
std::vector<Track> TrackGenerator::generate()
{
  /// Generate tracks. The number of tracks follows a poissonian distribution
  std::poisson_distribution<> distTracks(mMeanTracksPerEvent);
  int nTracks = distTracks(mGenerator);
  return generate(nTracks);
}

//______________________________________________________________________________
std::vector<Track> TrackGenerator::generate(int nTracks)
{
  /// Generate N tracks
  /// @param nTracks Number of tracks to generate
  std::vector<Track> tracks;
  Track track;
  std::array<float, 3> pos;
  std::array<float, 2> dir;
  for (int itrack = 0; itrack < nTracks; ++itrack) {
    for (int ipos = 0; ipos < 3; ++ipos) {
      pos[ipos] =
        std::uniform_real_distribution<float>{ mPositionLimits[2 * ipos], mPositionLimits[2 * ipos + 1] }(mGenerator);
    }
    track.setPosition(pos[0], pos[1], pos[2]);

    std::array<float, 4> limits = getLimitsForAcceptance(pos);

    for (int idir = 0; idir < 2; ++idir) {
      dir[idir] = std::uniform_real_distribution<float>{ limits[2 * idir], limits[2 * idir + 1] }(mGenerator);
    }
    track.setDirection(dir[0], dir[1], 1.);
    tracks.push_back(track);
  }
  return tracks;
}

//______________________________________________________________________________
std::array<float, 4> TrackGenerator::getLimitsForAcceptance(std::array<float, 3> pos)
{
  /// Restricts the limits of the slopes
  /// so that the track enters in the spectrometer acceptance
  std::array<float, 4> limits = mSlopeLimits;
  float dZ = -1600. - pos[2];
  if (dZ == 0.) {
    dZ = 0.0001;
  }

  // These are the maximum x and y position of MT11 with the standard alignment
  std::array<float, 2> maxValues{ { 257., 306.7 } };
  for (int icoor = 0; icoor < 2; ++icoor) {
    int imin = 2 * icoor;
    int imax = 2 * icoor + 1;
    float minSlope = (pos[icoor] - maxValues[icoor]) / dZ;
    float maxSlope = (maxValues[icoor] - pos[icoor]) / dZ;
    if (minSlope > limits[imin]) {
      limits[imin] = minSlope;
    }
    if (maxSlope < limits[imax]) {
      limits[imax] = maxSlope;
    }
  }
  return limits;
}
} // namespace mid
} // namespace o2
