// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/src/Track.cxx
/// \brief  Reconstructed MID track
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   04 September 2017

#include "DataFormatsMID/Track.h"

#include <iostream>

namespace o2
{
namespace mid
{

//______________________________________________________________________________
void Track::setCovarianceParameters(float xErr2, float yErr2, float slopeXErr2, float slopeYErr2, float covXSlopeX,
                                    float covYSlopeY)
{
  /// Sets the covariance parameters
  mCovarianceParameters[static_cast<int>(CovarianceParamIndex::VarX)] = xErr2;
  mCovarianceParameters[static_cast<int>(CovarianceParamIndex::VarY)] = yErr2;
  mCovarianceParameters[static_cast<int>(CovarianceParamIndex::VarSlopeX)] = slopeXErr2;
  mCovarianceParameters[static_cast<int>(CovarianceParamIndex::VarSlopeY)] = slopeYErr2;
  mCovarianceParameters[static_cast<int>(CovarianceParamIndex::CovXSlopeX)] = covXSlopeX;
  mCovarianceParameters[static_cast<int>(CovarianceParamIndex::CovYSlopeY)] = covYSlopeY;
}

//______________________________________________________________________________
void Track::setDirection(float xDir, float yDir, float zDir)
{
  /// Sets the track direction parameters
  mDirection = {xDir, yDir, zDir};
}

//______________________________________________________________________________
void Track::setPosition(float xPos, float yPos, float zPos)
{
  /// Sets the track starting position
  mPosition = {xPos, yPos, zPos};
}

//______________________________________________________________________________
int Track::getClusterMatched(int chamber) const
{
  /// Gets the matched clusters
  if (chamber > 3) {
    std::cerr << "Error: chamber must be in range [0, 4]\n";
    return 0;
  }
  return mClusterMatched[chamber];
}

//______________________________________________________________________________
void Track::setClusterMatched(int chamber, int id)
{
  /// Sets the matched clusters
  if (chamber > 3) {
    std::cerr << "Error: chamber must be in range [0, 4]\n";
    return;
  }
  mClusterMatched[chamber] = id;
}

//______________________________________________________________________________
bool Track::propagateToZ(float zPosition)
{
  /// Propagate track to the specified zPosition
  /// A linear extraplation is performed

  // Nothing to be done if we're already at zPosition
  // Notice that the z position is typically the z of the cluster,
  // which is provided in float precision as well.
  // So it is typically a well defined value, hence the strict equality
  if (mPosition[2] == zPosition) {
    return false;
  }

  // Propagate the track position parameters
  float dZ = zPosition - mPosition[2];
  float newX = mPosition[0] + mDirection[0] * dZ;
  float newY = mPosition[1] + mDirection[1] * dZ;
  setPosition(newX, newY, zPosition);

  // Propagate the covariance matrix
  std::array<float, 6> newCovParams;
  float dZ2 = dZ * dZ;
  for (int idx = 0; idx < 2; ++idx) {
    int slopeIdx = idx + 2;
    int covIdx = idx + 4;
    // s_x^2 -> s_x^2 + 2*cov(x,slopeX)*dZ + s_slopeX^2*dZ^2
    newCovParams[idx] =
      mCovarianceParameters[idx] + 2. * mCovarianceParameters[covIdx] * dZ + mCovarianceParameters[slopeIdx] * dZ2;
    // cov(x,slopeX) -> cov(x,slopeX) + s_slopeX^2*dZ
    newCovParams[covIdx] = mCovarianceParameters[covIdx] + mCovarianceParameters[slopeIdx] * dZ;
    // s_slopeX^2 -> s_slopeX^2
    newCovParams[slopeIdx] = mCovarianceParameters[slopeIdx];
  }

  mCovarianceParameters.swap(newCovParams);

  return true;
}

//______________________________________________________________________________
bool Track::isCompatible(const Track& track, float chi2Cut) const
{
  /// Check if tracks are compatible within uncertainties
  if (track.mPosition[2] != mPosition[2]) {
    Track copyTrack(track);
    copyTrack.propagateToZ(mPosition[2]);
    return isCompatible(copyTrack, chi2Cut);
  }

  // // method 1: chi2 calculation accounting for covariance between slope and position.
  // // This is the full calculation. However, if we have two parallel tracks with same x
  // // but different ( > Nsigmas) y position, the algorithm will return true since the
  // // difference in one of the parameters is compensated by the similarity of the others.
  // // This is probably not what we need.
  // double chi2 = 0.;
  // double pos1[2] = { getPosition().x(), getPosition().y() };
  // double dir1[2] = { getDirection().x(), getDirection().y() };
  // double pos2[2] = { track.getPosition().x(), track.getPosition().y() };
  // double dir2[2] = { track.getDirection().x(), track.getDirection().y() };
  //
  // for (int icoor = 0; icoor < 2; ++icoor) {
  //   double diffPos = pos1[icoor] - pos2[icoor];
  //   double diffSlope = dir1[icoor] - dir2[icoor];
  //   double varPos = mCovarianceParameters[icoor] + track.mCovarianceParameters[icoor];
  //   double varSlope = mCovarianceParameters[icoor + 2] + track.mCovarianceParameters[icoor + 2];
  //   double cov = mCovarianceParameters[icoor + 4] + track.mCovarianceParameters[icoor + 4];
  //   chi2 += (diffPos * diffPos * varSlope + diffSlope * diffSlope * varPos - 2. * diffPos * diffSlope * cov) /
  //           (varPos * varSlope - cov * cov);
  // }
  //
  // return (chi2 / 4.) < chi2Cut;

  // method 2: apply the cut on each parameter
  // This method avoids the issue of method 1
  double p1[4] = {mPosition[0], mPosition[1], mDirection[0], mDirection[1]};
  double p2[4] = {track.mPosition[0], track.mPosition[1], track.mDirection[0],
                  track.mDirection[1]};
  for (int ipar = 0; ipar < 4; ++ipar) {
    double diff = p1[ipar] - p2[ipar];
    if (diff * diff / (mCovarianceParameters[ipar] + track.mCovarianceParameters[ipar]) > chi2Cut) {
      return false;
    };
  }
  return true;
}

//______________________________________________________________________________
std::ostream& operator<<(std::ostream& stream, const Track& track)
{
  /// Overload ostream operator
  stream << "Position: (" << track.mPosition[0] << ", " << track.mPosition[1] << ", " << track.mPosition[2] << ")";
  stream << " Direction: (" << track.mDirection[0] << ", " << track.mDirection[1] << ", " << track.mDirection[2] << ")";
  stream << " Covariance (X, Y, SlopeX, SlopeY, X-SlopeX, Y-SlopeY): (";
  for (int ival = 0; ival < 6; ++ival) {
    stream << track.mCovarianceParameters[ival];
    stream << ((ival == 5) ? ")" : ", ");
  }
  return stream;
}

} // namespace mid
} // namespace o2
