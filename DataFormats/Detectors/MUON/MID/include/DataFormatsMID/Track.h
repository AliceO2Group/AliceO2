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

/// \file   DataFormatsMID/Track.h
/// \brief  Reconstructed MID track
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   04 September 2017
#ifndef O2_MID_TRACK_H
#define O2_MID_TRACK_H

#include <array>
#include <ostream>

#include "Rtypes.h"

namespace o2
{
namespace mid
{
/// This class defines the MID track
class Track
{
 public:
  /// Gets the track x position
  float getPositionX() const { return mPosition[0]; }
  /// Gets the track y position
  float getPositionY() const { return mPosition[1]; }
  /// Gets the track z position
  float getPositionZ() const { return mPosition[2]; }

  /// Ses the track position
  /// \param xPos x position
  /// \param yPos y position
  /// \param zPos z position
  void setPosition(float xPos, float yPos, float zPos);

  /// Gets the track x direction
  float getDirectionX() const { return mDirection[0]; }
  /// Gets the track y direction
  float getDirectionY() const { return mDirection[1]; }
  /// Gets the track z direction
  float getDirectionZ() const { return mDirection[2]; }

  /// Ses the track direction
  /// \param xDir x direction
  /// \param yDir y direction
  /// \param zDir z direction
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
  const std::array<float, 6>& getCovarianceParameters() const { return mCovarianceParameters; }
  /// returns the covariance parameter covParam
  float getCovarianceParameter(CovarianceParamIndex covParam) const
  {
    return mCovarianceParameters[static_cast<int>(covParam)];
  }

  /// Sets the covariance parameters
  /// \param xErr2 Variance on x position
  /// \param yErr2 Variance on y position
  /// \param slopeXErr2 Variance on x slope
  /// \param slopeYErr2 Variance on y slope
  /// \param covXSlopeX Covariance on x position and slope
  /// \param covYSlopeY Covariance on y position and slope
  void setCovarianceParameters(float xErr2, float yErr2, float slopeXErr2, float slopeYErr2, float covXSlopeX, float covYSlopeY);

  /// Sets the covariance matrix
  /// \param covParam array with the covariance parameters
  void setCovarianceParameters(const std::array<float, 6>& covParam) { mCovarianceParameters = covParam; }

  /// Checks if covariance is set
  bool isCovarianceSet() { return (mCovarianceParameters[0] != 0.); }

  /// Propagates the track parameter to z position with a linear extrapolation
  /// \param zPosition Z position
  /// \return false if the track parameters were already at the required z
  bool propagateToZ(float zPosition);

  /// Gets the matched clusters without bound checking
  /// \param chamber Chamber ID (from 0 to 3)
  int getClusterMatchedUnchecked(int chamber) const { return mClusterMatched[chamber]; }
  /// Gets the matched clusters
  /// \param chamber Chamber ID (from 0 to 3)
  int getClusterMatched(int chamber) const;

  /// Sets the matched clusters without bound checking
  /// \param chamber Chamber ID (from 0 to 3)
  /// \param id Cluster ID
  void setClusterMatchedUnchecked(int chamber, int id) { mClusterMatched[chamber] = id; }

  /// Sets the matched clusters
  /// \param chamber Chamber ID (from 0 to 3)
  /// \param id Cluster ID
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
  float getChi2OverNDF() const { return mChi2 / static_cast<float>(mNDF); }

  /// Check if tracks are compatible within uncertainties
  /// \param track Track for which compatibility is checked
  /// \param chi2Cut Chi2 cut for the comparison
  bool isCompatible(const Track& track, float chi2Cut) const;

  /// Sets the fired chamber
  /// \param chamber Chamber ID (from 0 to 3)
  /// \param cathode Cathode (0 or 1)
  void setFiredChamber(int chamber, int cathode) { mEfficiencyWord |= 1 << (4 * cathode + chamber); }

  /// Is fired chamber
  /// \param chamber Chamber ID (from 0 to 3)
  /// \param cathode Cathode (0 or 1)
  /// \return true if the chamber was fired
  bool isFiredChamber(int chamber, int cathode) const { return mEfficiencyWord & (1 << (4 * cathode + chamber)); }

  /// Gets hit map
  uint8_t getHitMap() const { return mEfficiencyWord & 0xFF; }

  /// Gets the word allowing to compute the chamber efficiency
  uint32_t getEfficiencyWord() const { return mEfficiencyWord; }

  /// Sets the fired FEE ID
  void setFiredFEEId(int uniqueFeeId) { setEfficiencyWord(8, 0x7FFF, uniqueFeeId); }

  /// Gets the fired FEE ID
  int getFiredFEEId() const { return (mEfficiencyWord >> 8) & 0x7FFF; }

  /// Gets the fired local board for efficiency calculation
  [[deprecated]] int getFiredLocalBoard() const { return (mEfficiencyWord >> 8) & 0xFF; }

  /// Gets the fired Detection Element ID
  int getFiredDEId() const { return (mEfficiencyWord >> 16) & 0x7F; }

  /// Gets the fired Detection Element ID
  [[deprecated("Use getFiredDEId instead")]] int getFiredDeId() const { return getFiredDEId(); }

  /// Gets the fired column ID
  int getFiredColumnId() const { return (mEfficiencyWord >> 12) & 0x7; }

  /// Gets the fired line ID
  int getFiredLineId() const { return (mEfficiencyWord >> 8) & 0x3; }

  /// Sets the flag for efficiency calculation
  /// \param effFlag efficiency flag
  void setEfficiencyFlag(int effFlag) { setEfficiencyWord(24, 0xF, effFlag); }

  /// Gets the flag for efficiency calculation
  /// \return
  /// \li \c 0 track cannot be used for efficiency calculation
  /// \li \c 1 track can be used to estimate chamber efficiency
  /// \li \c 2 track can be used to estimate detection element efficiency
  /// \li \c 3 track can be used to estimate local board efficiency
  int getEfficiencyFlag() const { return (mEfficiencyWord >> 24) & 0xF; }

  /// Overload ostream operator for MID track
  friend std::ostream& operator<<(std::ostream& stream, const Track& track);

 private:
  /// Set portions of the efficiency word
  /// \param pos Position in the word
  /// \param mask Maximum size of the bits
  /// \param value Value to be set
  void setEfficiencyWord(int pos, int mask, int value);
  std::array<float, 3> mPosition = {};             ///< Position
  std::array<float, 3> mDirection = {};            ///< Direction
  std::array<float, 6> mCovarianceParameters = {}; ///< Covariance parameters
  std::array<int, 4> mClusterMatched = {};         ///< Matched cluster index
  float mChi2 = 0.;                                ///< Chi2 of track
  int mNDF = 0;                                    ///< Number of chi2 degrees of freedom
  uint32_t mEfficiencyWord = 0;                    ///< Efficiency word

  ClassDefNV(Track, 1);
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_TRACK_H */
