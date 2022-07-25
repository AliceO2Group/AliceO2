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

/// @file   GeomConstants.h
/// @brief  Some ALICE geometry constants of common interest

/// @author ruben.shahoyan@cern.ch

#ifndef ALICEO2_GEOMCONSTANTS_H_
#define ALICEO2_GEOMCONSTANTS_H_

namespace o2
{
namespace constants
{
namespace geom
{
constexpr float XBeamPipeOuterRef = 1.6; ///< inner radius of the beam pipe
constexpr float XTPCInnerRef = 83.0;     ///< reference radius at which TPC provides the tracks
constexpr float XTPCOuterRef = 255.0;    ///< reference radius to propagate outer TPC track
} // namespace geom
} // namespace constants
} // namespace o2

#endif
