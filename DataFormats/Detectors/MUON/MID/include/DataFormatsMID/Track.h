// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   DataFormatsMID/Track.h
/// \brief  Reconstructed MID track
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   04 September 2017
#ifndef O2_MID_TRACK_H
#define O2_MID_TRACK_H

#include <array>
#include <ostream>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/array.hpp>
#include "MathUtils/Cartesian3D.h"

namespace o2
{
namespace mid
{
/// This class defines the MID track
class Track
{
 public:
  Track();
  virtual ~Track() = default;

  Track(const Track&) = default;
  Track& operator=(const Track&) = default;
  Track(Track&&) = default;
  Track& operator=(Track&&) = default;

  /// Gets the track starting position
  const Point3D<float> getPosition() const { return mPosition; }
  void setPosition(float xPos, float yPos, float zPos);

  /// Gets the track direction parameters
  const Vector3D<float> getDirection() const { return mDirection; }
  void setDirection(float xDir, float yDir, float zDir);

  enum class CovarianceParamIndex : int {
    VarX,       ///< Variance on X position
    VarY,       ///< Variance on Y position
    VarSlopeX,  ///< Variance on X slope
    VarSlopeY,  ///< Variance on Y slope
    CovXSlopeX, ///< Covariance on X position and slope
    CovYSlopeY  ///< Covariance on Y position and slope
  };

  /// Gets the covariance parameters
  const std::array<float, 6> getCovarianceParameters() const { return mCovarianceParameters; }
  /// returns the covariance parameter covParam
  float getCovarianceParameter(CovarianceParamIndex covParam) const
  {
    return mCovarianceParameters[static_cast<int>(covParam)];
  }
  void setCovarianceParameters(float xErr2, float yErr2, float slopeXErr2, float slopeYErr2, float covXSlopeX,
                               float covYSlopeY);

  /// Sets the covariance matrix
  void setCovarianceParameters(const std::array<float, 6>& covParam) { mCovarianceParameters = covParam; }

  /// Checks if covariance is set
  bool isCovarianceSet() { return (mCovarianceParameters[0] != 0.); }

  bool propagateToZ(float zPosition);

  int getClusterMatched(int chamber) const;
  void setClusterMatched(int chamber, int id);

  /// Returns the chi2 of the track
  float getChi2() const { return mChi2; }
  /// Sets the chi2 of the track
  void setChi2(float chi2) { mChi2 = chi2; }
  /// Returns the number of degrees of freedom of the track
  int getNDF() const { return mNDF; }
  /// Sets the number of degrees of freedom of the track
  void setNDF(int ndf) { mNDF = ndf; }
  /// Returns the normalized chi2 of the track
  float getChi2OverNDF() const { return mChi2 / (float)mNDF; }

  bool isCompatible(const Track& track, float chi2Cut) const;

  friend std::ostream& operator<<(std::ostream& stream, const Track& track);

 private:
  Point3D<float> mPosition;                   ///< Position
  Vector3D<float> mDirection;                 ///< Direction
  std::array<float, 6> mCovarianceParameters; ///< Covariance parameters
  std::array<int, 4> mClusterMatched;         ///< Matched cluster index
  float mChi2;                                ///< Chi2 of track
  int mNDF;                                   ///< Number of degree of freedom for chi2

  friend class boost::serialization::access;

  /// Serializes the track
  template <class Archive>
  void save(Archive& ar, const unsigned int version) const
  {
    ar& mPosition.x() & mPosition.y() & mPosition.z();
    ar& mDirection.x() & mDirection.y() & mDirection.z();
    ar& mCovarianceParameters;
    ar& mClusterMatched;
  }

  /// Deserializes the track
  template <class Archive>
  void load(Archive& ar, const unsigned int version)
  {
    float xp, yp, zp, xd, yd, zd;
    ar& xp& yp& zp;
    mPosition.SetXYZ(xp, yp, zp);
    ar& xd& yd& zd;
    mDirection.SetXYZ(xd, yd, zd);
    ar& mCovarianceParameters;
    ar& mClusterMatched;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER()
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_TRACK_H */
