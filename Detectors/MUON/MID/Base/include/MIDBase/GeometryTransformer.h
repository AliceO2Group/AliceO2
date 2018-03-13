// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDBase/GeometryTransformer.h
/// \brief  Geometry transformer for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   07 July 2017

#ifndef O2_MID_GEOMETRYTRANSFORMER_H
#define O2_MID_GEOMETRYTRANSFORMER_H

#include <array>
#include "MathUtils/Cartesian3D.h"

namespace o2
{
namespace mid
{
/// Class to convert the local MID RPC coordinates into global ones
/// and viceversa
class GeometryTransformer
{
 public:
  GeometryTransformer();
  virtual ~GeometryTransformer() = default;

  GeometryTransformer(const GeometryTransformer&) = delete;
  GeometryTransformer& operator=(const GeometryTransformer&) = delete;
  GeometryTransformer(GeometryTransformer&&) = delete;
  GeometryTransformer& operator=(GeometryTransformer&&) = delete;

  Point3D<float> localToGlobal(int deId, const Point3D<float>& position) const;
  Point3D<float> globalToLocal(int deId, const Point3D<float>& position) const;
  Point3D<float> localToGlobal(int deId, float xPos, float yPos) const;
  Point3D<float> globalToLocal(int deId, float xPos, float yPos, float zPos) const;
  Vector3D<float> localToGlobal(int deId, const Vector3D<float>& direction) const;
  Vector3D<float> globalToLocal(int deId, const Vector3D<float>& direction) const;

 private:
  void init();
  std::array<ROOT::Math::Transform3D, 72> mTransformations; ///< Array of transformation matrices
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_GEOMETRYTRANSFORMER_H */
