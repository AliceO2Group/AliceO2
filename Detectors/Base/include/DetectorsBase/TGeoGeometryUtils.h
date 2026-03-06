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

/// \file TGeoGeometryUtils.h
/// \author Sandro Wenzel (CERN)
/// \brief Collection of utility functions for TGeo

#ifndef ALICEO2_BASE_TGEOGEOMETRYUTILS_H_
#define ALICEO2_BASE_TGEOGEOMETRYUTILS_H_

class TGeoShape;
class TGeoTessellated;

namespace o2
{
namespace base
{

/// A few utility functions to operate on TGeo geometries (transformations, printing, ...)
class TGeoGeometryUtils
{
 public:
  ///< Transform any (primitive) TGeoShape to a tessellated representation
  static TGeoTessellated* TGeoShapeToTGeoTessellated(TGeoShape const*);
};

} // namespace base
} // namespace o2

#endif
