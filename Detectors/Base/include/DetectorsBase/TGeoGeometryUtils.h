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

  ///< Create a bounded stand-in for the half-space { x : (x - p) . n <= 0 }, which is what
  ///< TGeoHalfSpace describes. Registers a cube of half-size `reach` under `name` and its
  ///< placement under "<name>_tr". The stand-in agrees with the half-space everywhere within
  ///< a distance `reach` of `p`, so `reach` must exceed the extent of the solid it is
  ///< subtracted from. Unlike TGeoHalfSpace, the result can be exported to GDML and
  ///< converted to native Geant4 geometry.
  ///<
  ///< Write the term in a composite expression **in parentheses**, as "-(<name>:<name>_tr)".
  ///< A trailing "shape:matrix" is not safe: TGeoManager::Parse takes the last top-level ":"
  ///< of an expression that already contains a top-level ")" to be a transformation of the
  ///< whole expression, warns "no geometrical transformation allowed at this level" and then
  ///< drops it - leaving an unplaced cube at the origin that swallows the parent solid.
  static void makeHalfSpaceBox(const char* name, const double p[3], const double n[3], double reach);
};

} // namespace base
} // namespace o2

#endif
