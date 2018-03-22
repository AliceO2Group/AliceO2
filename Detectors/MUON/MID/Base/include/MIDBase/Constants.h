// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDBase/Constants.h
/// \brief  Useful constants for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   08 March 2018
#ifndef O2_MID_CONSTANTS_H
#define O2_MID_CONSTANTS_H

#include <array>

namespace o2
{
namespace mid
{
class Constants
{
 public:
  Constants() = default;
  virtual ~Constants() = default;

  Constants(const Constants&) = delete;
  Constants& operator=(const Constants&) = delete;
  Constants(Constants&&) = delete;
  Constants& operator=(Constants&&) = delete;

  /// The RPC is a short one
  /// @param deId The detection element ID
  inline static bool isShortRPC(int deId) { return (deId % 9 == 4); }

  /// Returns the height of the local board in the chamber
  /// @param chamber The chamber ID (0-3)
  inline static double getLocalBoardHeight(int chamber) { return mLocalBoardHeight * mScaleFactors[chamber]; }

  /// Returns the height of the local board in the chamber
  /// @param chamber The chamber ID (0-3)
  inline static double getLocalBoardWidth(int chamber) { return mLocalBoardWidth * mScaleFactors[chamber]; }

  /// Returns the position of the center of the RPC in the chamber
  /// @param chamber The chamber ID (0-3)
  /// @param rpc The RPC ID (0-9)
  inline static double getRPCCenterPosX(int chamber, int rpc)
  {
    return isShortRPC(rpc) ? mRPCShortCenterPos * mScaleFactors[chamber] : mRPCCenterPos * mScaleFactors[chamber];
  }

  /// Returns the half width of the RPC in the chamber
  /// @param chamber The chamber ID (0-3)
  /// @param rpc The RPC ID (0-9)
  inline static double getRPCHalfWidth(int chamber, int rpc)
  {
    return isShortRPC(rpc) ? mRPCShortHalfWidth * mScaleFactors[chamber] : mRPCHalfWidth * mScaleFactors[chamber];
  }

  /// Returns the half height of the RPC in the chamber
  /// @param chamber The chamber ID (0-3)
  inline static double getRPCHalfHeight(int chamber) { return 2. * mLocalBoardHeight * mScaleFactors[chamber]; }

  /// Returns the unit strip pitch size in the chamber
  /// @param chamber The chamber ID (0-3)
  inline static double getStripUnitPitchSize(int chamber) { return mStripUnitPitchSize * mScaleFactors[chamber]; }

  /// Get chamber index from detection element ID
  inline static int getChamber(int deId) { return (deId % 36) / 9; }

  /// Gets detection element Id
  /// @param isRight RPC is in right side
  /// @param chamber The chamber ID (0-3)
  /// @param rpc RPC ID (0-8)
  inline static int getDEId(bool isRight, int chamber, int rpc)
  {
    int deOffset = (isRight) ? 0 : mNDetectionElementsPerSide;
    return deOffset + mNRPCLines * chamber + rpc;
  }

  static constexpr int mNChambers = 4;                  ///< Number of chambers
  static constexpr int mNDetectionElements = 72;        ///< Number of RPCs
  static constexpr int mNLocalBoards = 234;             ///< Number of local boards per chamber
  static constexpr int mNStripsBP = 16;                 ///< Number of strips in the Bending Plane
  static constexpr int mNDetectionElementsPerSide = 36; ///< Number of detection elements per side
  static constexpr int mNRPCLines = 9;                  ///< Number of RPC lines

  // Standard sizes/position in the first chamber.
  // The values for the other chambers can be obtained multiplying by the scaling factors below
  static constexpr double mLocalBoardHeight = 17.; ///< Local board height in the first chamber
  static constexpr double mLocalBoardWidth = 34.;  ///< Local board width in the first chamber
  static constexpr double mRPCCenterPos = 129.5;   ///< Position of most RPCs in the right side of the first chamber
  static constexpr double mRPCHalfWidth = 127.5;   ///< Half width of most RPCs in the first chamber
  static constexpr double mRPCShortCenterPos =
    155.; ///< Position of the short RPC in the right side of the first chamber
  static constexpr double mRPCShortHalfWidth = 102.;    ///< Half width of the short RPC in the first chamber
  static constexpr double mStripUnitPitchSize = 1.0625; ///< Unit pitch size of the strip in the first chamber

  static constexpr double mRPCZShift =
    3.6; ///< Default shift of the RPC z position with respect to the average chamber position

  static constexpr double mBeamAngle = -0.794; ///< Angle between beam position and horizontal

  static constexpr std::array<const double, 4> mScaleFactors{
    { 1., 1.01060, 1.06236, 1.07296 }
  }; ///< Array of scale factors for projective geometry
  static constexpr std::array<const double, 4> mDefaultChamberZ{
    { -1603.5, -1620.5, -1703.5, -1720.5 }
  }; ///< Array of default z position of the chamber
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_MAPPING_H */
