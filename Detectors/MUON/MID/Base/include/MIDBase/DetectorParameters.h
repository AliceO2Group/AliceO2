// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDBase/DetectorParameters.h
/// \brief  Useful detector parameters for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   08 March 2018
#ifndef O2_MID_DETECTORPARAMETERS_H
#define O2_MID_DETECTORPARAMETERS_H

#include <array>
#include <string>

namespace o2
{
namespace mid
{
/// MID detector parameters
namespace detparams
{
constexpr int NChambers = 4;                  ///< Number of chambers
constexpr int NDetectionElements = 72;        ///< Number of RPCs
constexpr int NDetectionElementsPerSide = 36; ///< Number of detection elements per side
constexpr int NLocalBoards = 234;             ///< Number of local boards per chamber
constexpr int NRPCLines = 9;                  ///< Number of RPC lines
constexpr int NStripsBP = 16;                 ///< Number of strips in the Bending Plane

/// Gets the chamber index from detection element ID
/// @param deId The detection element ID
inline int getChamber(int deId) { return (deId % NDetectionElementsPerSide) / NRPCLines; }

/// Gets the chamber index from detection element ID
/// @param deId The detection element ID
inline int getRPCLine(int deId) { return deId % NRPCLines; }

void assertDEId(int deId);

int getDEId(bool isRight, int chamber, int rpc);
std::string getDEName(int deId);
} // namespace detparams
} // namespace mid
} // namespace o2

#endif /* O2_MID_DETECTORPARAMETERS_H */
