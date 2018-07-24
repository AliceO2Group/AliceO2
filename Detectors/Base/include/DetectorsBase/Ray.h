// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Ray.cxx
/// \brief Call for the ray between start-end points for material budget estimate

#ifndef ALICEO2_RAY_H
#define ALICEO2_RAY_H

#include "AliTPCCommonRtypes.h"
#include "AliTPCCommonDef.h"
#include "MathUtils/Cartesian3D.h"
#include "DetectorsBase/MatLayerCyl.h"
#include "MathUtils/Utils.h"
#include <array>

/**********************************************************************
 *                                                                    *
 * Ray parameterized via its endpoints as                             *
 * Vi = Vi0 + t*(Vi1-Vi0), with Vi (i=0,1,2) for global X,Y,Z         *
 * and 0 < t < 1                                                      *
 *                                                                    *
 **********************************************************************/
namespace o2
{
namespace Base
{

class Ray
{

 public:
  using CrossPar = std::pair<float, float>;

  static constexpr float InvalidT = -1e9;
  static constexpr float Tiny = 1e-9;

  Ray();
  ~Ray() CON_DEFAULT;
  Ray(const Point3D<float> point0, const Point3D<float> point1);
  Ray(float x0, float y0, float z0, float x1, float y1, float z1);

  int crossLayer(const MatLayerCyl& lr);
  bool crossCircleR(float r2, CrossPar& cross) const;

  float crossRadial(const MatLayerCyl& lr, int sliceID) const;
  float crossRadial(float cs, float sn) const;
  float crossZ(float z) const;

  const CrossPar& getCrossParams(int i) const { return mCrossParams[i]; }

  void getMinMaxR2(float& rmin2, float& rmax2) const;

  float getDist() const { return mDistXYZ; }
  float getDist(float deltaT) const { return mDistXYZ * (deltaT > 0 ? deltaT : -deltaT); }

  Point3D<float> getPos(float t) const
  {
    return Point3D<float>(mP0.X() + t * mDx, mP0.Y() + t * mDy, mP0.Z() + t * mDz);
  }

  float getPhi(float t) const
  {
    float p = std::atan2(mP0.Y() + t * mDy, mP0.X() + t * mDx);
    o2::utils::BringTo02Pi(p);
    return p;
  }

  float getZ(float t) const { return mP0.Z() + t * mDz; }

  bool validateZRange(CrossPar& cpar, const MatLayerCyl& lr) const;

 private:
  Point3D<float> mP0;                   ///< entrance point
  float mDx;                            ///< X distance
  float mDy;                            ///< Y distance
  float mDz;                            ///< Z distance
  float mDistXY2;                       ///< dist^2 between points in XY plane
  float mDistXY2i;                      ///< inverse dist^2 between points in XY plane
  float mDistXYZ;                       ///< distance between 2 points
  float mXDxPlusYDy;                    ///< aux x0*DX+y0*DY
  float mXDxPlusYDyRed;                 ///< aux (x0*DX+y0*DY)/mDistXY2
  float mXDxPlusYDy2;                   ///< aux (x0*DX+y0*DY)^2
  float mR02;                           ///< radius^2 of mP0
  float mR12;                           ///< radius^2 of mP1
  std::array<CrossPar, 2> mCrossParams; ///< parameters of crossing the layer

  ClassDefNV(Ray, 1);
};

//______________________________________________________
inline bool Ray::crossCircleR(float r2, CrossPar& cross) const
{
  // calculate parameters t of intersection with circle of radius r^2
  // calculated as solution of equation
  // t^2*mDistXY2 +- sqrt( mXDxPlusYDy^2 - mDistXY2*(mR02 - r^2) )
  //
  float det = mXDxPlusYDy2 - mDistXY2 * (mR02 - r2);
  if (det < 0)
    return false; // no intersection
  float detRed = std::sqrt(det) * mDistXY2i;
  cross.first = mXDxPlusYDyRed + detRed;  // (-mXDxPlusYDy + det)*mDistXY2i;
  cross.second = mXDxPlusYDyRed - detRed; // (-mXDxPlusYDy - det)*mDistXY2i;
  return true;
}

inline float Ray::crossRadial(const MatLayerCyl& lr, int sliceID) const
{
  // calculate t of crossing with phimin of layer's slice sliceID
  return crossRadial(lr.getSliceCos(sliceID), lr.getSliceSin(sliceID));
}

//______________________________________________________
inline float Ray::crossRadial(float cs, float sn) const
{
  // calculate t of crossing with radial line with inclination cosine and sine
  float den = mDx * sn - mDy * cs;
  if (std::abs(den) < Tiny)
    return InvalidT;
  return (mP0.Y() * cs - mP0.X() * sn) / den;
}

//______________________________________________________
inline float Ray::crossZ(float z) const
{
  // calculate t of crossing XY plane at Z
  return std::abs(mDz) > Tiny ? (z - mP0.Z()) / mDz : InvalidT;
}

//______________________________________________________
inline bool Ray::validateZRange(CrossPar& cpar, const MatLayerCyl& lr) const
{
  // make sure that estimated crossing parameters are compatible
  // with Z coverage of the layer
  MatLayerCyl::RangeStatus zout0 = lr.isZOutside(getZ(cpar.first)), zout1 = lr.isZOutside(getZ(cpar.second));
  if (zout0 == zout1) { // either both points outside w/o crossing or boht inside
    return zout0 == MatLayerCyl::Within ? true : false;
  }
  // at least 1 point is outside, but there is a crossing
  if (zout0 != MatLayerCyl::Within) {
    cpar.first = crossZ(zout0 == MatLayerCyl::Below ? lr.getZMin() : lr.getZMax());
  }
  if (zout1 != MatLayerCyl::Within) {
    cpar.second = crossZ(zout1 == MatLayerCyl::Below ? lr.getZMin() : lr.getZMax());
  }
  return true;
}

//______________________________________________________
inline void Ray::getMinMaxR2(float& rmin2, float& rmax2) const
{
  // calculate min and max R2
  if (mR02 > mR12) {
    rmin2 = mR12;
    rmax2 = mR02;
  } else {
    rmin2 = mR02;
    rmax2 = mR12;
  }
  if (mXDxPlusYDyRed > 0.f && mXDxPlusYDyRed < 1.f) {
    // estimate point of closest approach to origin as the crossing of normal from the origin to input vector
    // use r^2(t) = mR02 + t^2 (mDx^2+mDy^2) + 2t*mXDxPlusYDy
    float xMin = mP0.X() + mXDxPlusYDyRed * mDx, yMin = mP0.Y() + mXDxPlusYDyRed * mDy;
    rmin2 = xMin * xMin + yMin * yMin;
  }
}

} // namespace Base
} // namespace o2

#endif
