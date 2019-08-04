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
#include <string>

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
  inline static double getLocalBoardHeight(int chamber) { return sLocalBoardHeight * sScaleFactors[chamber]; }

  /// Returns the height of the local board in the chamber
  /// @param chamber The chamber ID (0-3)
  inline static double getLocalBoardWidth(int chamber) { return sLocalBoardWidth * sScaleFactors[chamber]; }

  /// Returns the position of the center of the RPC in the chamber
  /// @param chamber The chamber ID (0-3)
  /// @param rpc The RPC ID (0-9)
  inline static double getRPCCenterPosX(int chamber, int rpc)
  {
    return isShortRPC(rpc) ? sRPCShortCenterPos * sScaleFactors[chamber] : sRPCCenterPos * sScaleFactors[chamber];
  }

  /// Returns the half length of the RPC in the chamber
  /// @param chamber The chamber ID (0-3)
  /// @param rpc The RPC ID (0-9)
  inline static double getRPCHalfLength(int chamber, int rpc)
  {
    return isShortRPC(rpc) ? sRPCShortHalfLength * sScaleFactors[chamber] : sRPCHalfLength * sScaleFactors[chamber];
  }

  /// Returns the half height of the RPC in the chamber
  /// @param chamber The chamber ID (0-3)
  inline static double getRPCHalfHeight(int chamber) { return 2. * sLocalBoardHeight * sScaleFactors[chamber]; }

  /// Returns the unit strip pitch size in the chamber
  /// @param chamber The chamber ID (0-3)
  inline static double getStripUnitPitchSize(int chamber) { return sStripUnitPitchSize * sScaleFactors[chamber]; }

  /// Gets the chamber index from detection element ID
  /// @param deId The detection element ID
  inline static int getChamber(int deId) { return (deId % 36) / 9; }

  /// Assert detection element id
  static void assertDEId(int deId);

  /// Gets detection element Id
  /// @param isRight RPC is in right side
  /// @param chamber The chamber ID (0-3)
  /// @param rpc RPC ID (0-8)
  inline static int getDEId(bool isRight, int chamber, int rpc)
  {
    int deOffset = (isRight) ? 0 : sNDetectionElementsPerSide;
    return deOffset + sNRPCLines * chamber + rpc;
  }

  static std::string getDEName(int deId);

  static constexpr int sNChambers = 4;                  ///< Number of chambers
  static constexpr int sNDetectionElements = 72;        ///< Number of RPCs
  static constexpr int sNLocalBoards = 234;             ///< Number of local boards per chamber
  static constexpr int sNStripsBP = 16;                 ///< Number of strips in the Bending Plane
  static constexpr int sNDetectionElementsPerSide = 36; ///< Number of detection elements per side
  static constexpr int sNRPCLines = 9;                  ///< Number of RPC lines

  // Standard sizes/position in the first chamber.
  // The values for the other chambers can be obtained multiplying by the scaling factors below
  static constexpr double sLocalBoardHeight = 17.; ///< Local board height in the first chamber
  static constexpr double sLocalBoardWidth = 34.;  ///< Local board width in the first chamber
  static constexpr double sRPCCenterPos = 129.5;   ///< Position of most RPCs in the right side of the first chamber
  static constexpr double sRPCHalfLength = 127.5;  ///< Half length of most RPCs in the first chamber
  static constexpr double sRPCShortCenterPos =
    155.;                                               ///< Position of the short RPC in the right side of the first chamber
  static constexpr double sRPCShortHalfLength = 102.;   ///< Half length of the short RPC in the first chamber
  static constexpr double sStripUnitPitchSize = 1.0625; ///< Unit pitch size of the strip in the first chamber

  static constexpr double sRPCZShift =
    3.6; ///< Default shift of the RPC z position with respect to the average chamber position

  static constexpr double sBeamAngle = -0.794; ///< Angle between beam position and horizontal

  static constexpr std::array<const double, 4> sScaleFactors{
    {1., 1.01060, 1.06236, 1.07296}}; ///< Array of scale factors for projective geometry
  static constexpr std::array<const double, 4> sDefaultChamberZ{
    {-1603.5, -1620.5, -1703.5, -1720.5}}; ///< Array of default z position of the chamber
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_MAPPING_H */
