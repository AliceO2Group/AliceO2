// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDTestingSimTools/TrackGenerator.h
/// \brief  Fast track generator for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 December 2017

#ifndef O2_MID_TRACKGENERATOR_H
#define O2_MID_TRACKGENERATOR_H

#include <array>
#include <vector>
#include <random>
#include "DataFormatsMID/Track.h"

namespace o2
{
namespace mid
{
/// Class to generate tracks for MID
class TrackGenerator
{
 public:
  std::vector<Track> generate();
  std::vector<Track> generate(int nTracks);

  /// Sets the seed
  inline void setSeed(unsigned int seed) { mGenerator.seed(seed); }

  /// Sets the mean number of track per events
  void setMeanTracksPerEvent(int meanTracksPerEvent) { mMeanTracksPerEvent = meanTracksPerEvent; }
  /// Sets the limits of the track slope
  void setSlopeLimits(float slopeXmin, float slopeXmax, float slopeYmin, float slopeYmax)
  {
    mSlopeLimits = {{slopeXmin, slopeXmax, slopeYmin, slopeYmax}};
  }
  /// Sets the limits of the track origin
  void setPositionLimits(float xMin, float xMax, float yMin, float yMax, float zMin, float zMax)
  {
    mPositionLimits = {{xMin, xMax, yMin, yMax, zMin, zMax}};
  }

 private:
  std::array<float, 4> getLimitsForAcceptance(std::array<float, 3> pos);

  int mMeanTracksPerEvent = 1;                                         ///< Mean tracks per event
  std::array<float, 4> mSlopeLimits{{-0.2, 0.2, -0.5, 0.5}};           ///< Limits for track slope
  std::array<float, 6> mPositionLimits{{-2., 2, -2., 2., -5., 5.}};    ///< x,y,z position limits
  std::default_random_engine mGenerator{std::default_random_engine()}; ///< Random numbers generator
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_TRACKGENERATOR_H */
