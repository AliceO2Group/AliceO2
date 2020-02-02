// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDBase/GeometryParameters.h
/// \brief  Useful geometrical parameters for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   08 March 2018
#ifndef O2_MID_GEOMETRYPARAMETERS_H
#define O2_MID_GEOMETRYPARAMETERS_H

#include <array>
#include <string>

namespace o2
{
namespace mid
{
namespace geoparams
{
constexpr double BeamAngle = -0.794; ///< Angle between beam position and horizontal

// Standard sizes/position in the first chamber.
// The values for the other chambers can be obtained multiplying by the scaling factors below
constexpr double LocalBoardHeight = 17.;      ///< Local board height in the first chamber
constexpr double LocalBoardWidth = 34.;       ///< Local board width in the first chamber
constexpr double RPCCenterPos = 129.5;        ///< Position of most RPCs in the right side of the first chamber
constexpr double RPCHalfLength = 127.5;       ///< Half length of most RPCs in the first chamber
constexpr double RPCShortCenterPos = 155.;    ///< Position of the short RPC in the right side of the first chamber
constexpr double RPCShortHalfLength = 102.;   ///< Half length of the short RPC in the first chamber
constexpr double RPCZShift = 3.6;             ///< Default shift of the RPC z position with respect to the average chamber position
constexpr double StripUnitPitchSize = 1.0625; ///< Unit pitch size of the strip in the first chamber

constexpr std::array<const double, 4> ChamberScaleFactors{{1., 1.01060, 1.06236, 1.07296}};  ///< Array of scale factors for projective geometry
constexpr std::array<const double, 4> DefaultChamberZ{{-1603.5, -1620.5, -1703.5, -1720.5}}; ///< Array of default z position of the chamber

enum class RPCtype { Long,
                     BottomCut,
                     TopCut,
                     Short };

/// The RPC is a short one
/// @param deId The detection element ID
inline bool isShortRPC(int deId) { return (deId % 9 == 4); }

/// Returns the height of the local board in the chamber
/// @param chamber The chamber ID (0-3)
inline double getLocalBoardHeight(int chamber) { return LocalBoardHeight * ChamberScaleFactors[chamber]; }

/// Returns the height of the local board in the chamber
/// @param chamber The chamber ID (0-3)
inline double getLocalBoardWidth(int chamber) { return LocalBoardWidth * ChamberScaleFactors[chamber]; }

/// Returns the position of the center of the RPC in the chamber
/// @param chamber The chamber ID (0-3)
/// @param rpc The RPC ID (0-9)
inline double getRPCCenterPosX(int chamber, int rpc)
{
  return isShortRPC(rpc) ? RPCShortCenterPos * ChamberScaleFactors[chamber] : RPCCenterPos * ChamberScaleFactors[chamber];
}

/// Returns the half length of the RPC in the chamber
/// @param chamber The chamber ID (0-3)
/// @param rpc The RPC ID (0-9)
inline double getRPCHalfLength(int chamber, int rpc)
{
  return isShortRPC(rpc) ? RPCShortHalfLength * ChamberScaleFactors[chamber] : RPCHalfLength * ChamberScaleFactors[chamber];
}

/// Returns the half height of the RPC in the chamber
/// @param chamber The chamber ID (0-3)
inline double getRPCHalfHeight(int chamber) { return 2. * LocalBoardHeight * ChamberScaleFactors[chamber]; }

/// Returns the unit strip pitch size in the chamber
/// @param chamber The chamber ID (0-3)
inline double getStripUnitPitchSize(int chamber) { return StripUnitPitchSize * ChamberScaleFactors[chamber]; }

RPCtype getRPCType(int deId);
std::string getRPCVolumeName(RPCtype type, int iChamber);
std::string getChamberVolumeName(int chamber);
} // namespace geoparams
} // namespace mid
} // namespace o2

#endif /* O2_MID_GEOMETRYPARAMETERS_H */
