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

#include "GPUCommonRtypes.h"
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "DetectorsBase/MatLayerCyl.h"
#include "MathUtils/Utils.h"

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version
#include "MathUtils/Cartesian3D.h"
#endif // !GPUCA_ALIGPUCODE

/**********************************************************************
 *                                                                    *
 * Ray parameterized via its endpoints as                             *
 * Vi = Vi0 + t*(Vi1-Vi0), with Vi (i=0,1,2) for global X,Y,Z         *
 * and 0 < t < 1                                                      *
 *                                                                    *
 **********************************************************************/
namespace o2
{
namespace base
{

class Ray
{

 public:
  using vecF3 = float[3];

  static constexpr float InvalidT = -1e9;
  static constexpr float Tiny = 1e-9;

  GPUd() Ray() : mP{0.f}, mD{0.f}, mDistXY2(0.f), mDistXY2i(0.f), mDistXYZ(0.f), mXDxPlusYDy(0.f), mXDxPlusYDyRed(0.f), mXDxPlusYDy2(0.f), mR02(0.f), mR12(0.f)
  {
  }
  GPUdDefault() ~Ray() CON_DEFAULT;

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version
  Ray(const Point3D<float> point0, const Point3D<float> point1);
#endif // !GPUCA_ALIGPUCODE
  GPUd() Ray(float x0, float y0, float z0, float x1, float y1, float z1);

  GPUd() int crossLayer(const MatLayerCyl& lr);
  GPUd() bool crossCircleR(float r2, float& cross1, float& cross2) const;

  GPUd() float crossRadial(const MatLayerCyl& lr, int sliceID) const;
  GPUd() float crossRadial(float cs, float sn) const;
  GPUd() float crossZ(float z) const;

  GPUd() void getCrossParams(int i, float& par1, float& par2) const
  {
    par1 = mCrossParams1[i];
    par2 = mCrossParams2[i];
  }

  GPUd() void getMinMaxR2(float& rmin2, float& rmax2) const;

  GPUd() float getDist() const { return mDistXYZ; }
  GPUd() float getDist(float deltaT) const { return mDistXYZ * (deltaT > 0 ? deltaT : -deltaT); }

  // for debud only
  float getPos(float t, int i) const { return mP[i] + t * mD[i]; }

  GPUd() float getPhi(float t) const
  {
    float p = o2::gpu::CAMath::ATan2(mP[1] + t * mD[1], mP[0] + t * mD[0]);
    o2::utils::BringTo02Pi(p);
    return p;
  }

  GPUd() float getZ(float t) const { return mP[2] + t * mD[2]; }

  GPUd() bool validateZRange(float& cpar1, float& cpar2, const MatLayerCyl& lr) const;

 private:
  vecF3 mP;               ///< entrance point
  vecF3 mD;               ///< X,Y,Zdistance
  float mDistXY2;         ///< dist^2 between points in XY plane
  float mDistXY2i;        ///< inverse dist^2 between points in XY plane
  float mDistXYZ;         ///< distance between 2 points
  float mXDxPlusYDy;      ///< aux x0*DX+y0*DY
  float mXDxPlusYDyRed;   ///< aux (x0*DX+y0*DY)/mDistXY2
  float mXDxPlusYDy2;     ///< aux (x0*DX+y0*DY)^2
  float mR02;             ///< radius^2 of mP
  float mR12;             ///< radius^2 of mP1
  float mCrossParams1[2]; ///< parameters of crossing the layer (first parameter)
  float mCrossParams2[2]; ///< parameters of crossing the layer (second parameter)

  ClassDefNV(Ray, 1);
};

//______________________________________________________
#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version

inline Ray::Ray(const Point3D<float> point0, const Point3D<float> point1)
  : mP{point0.X(), point0.Y(), point0.Z()}, mD{point1.X() - point0.X(), point1.Y() - point0.Y(), point1.Z() - point0.Z()}
{
  mDistXY2 = mD[0] * mD[0] + mD[1] * mD[1];
  mDistXY2i = mDistXY2 > 0 ? 1.f / mDistXY2 : 0.f;
  mDistXYZ = o2::gpu::CAMath::Sqrt(mDistXY2 + mD[2] * mD[2]);
  mXDxPlusYDy = point0.X() * mD[0] + point0.Y() * mD[1];
  mXDxPlusYDyRed = -mXDxPlusYDy * mDistXY2i;
  mXDxPlusYDy2 = mXDxPlusYDy * mXDxPlusYDy;
  mR02 = point0.Perp2();
  mR12 = point1.Perp2();
}
#endif // !GPUCA_ALIGPUCODE

//______________________________________________________
GPUdi() Ray::Ray(float x0, float y0, float z0, float x1, float y1, float z1)
  : mP{x0, y0, z0}, mD{x1 - x0, y1 - y0, z1 - z0}
{
  mDistXY2 = mD[0] * mD[0] + mD[1] * mD[1];
  mDistXY2i = mDistXY2 > 0 ? 1.f / mDistXY2 : 0.f;
  mDistXYZ = o2::gpu::CAMath::Sqrt(mDistXY2 + mD[2] * mD[2]);
  mXDxPlusYDy = x0 * mD[0] + y0 * mD[1];
  mXDxPlusYDyRed = -mXDxPlusYDy * mDistXY2i;
  mXDxPlusYDy2 = mXDxPlusYDy * mXDxPlusYDy;
  mR02 = x0 * x0 + y0 * y0;
  mR12 = x1 * x1 + y1 * y1;
}

//______________________________________________________
GPUdi() float Ray::crossRadial(float cs, float sn) const
{
  // calculate t of crossing with radial line with inclination cosine and sine
  float den = mD[0] * sn - mD[1] * cs;
  if (o2::gpu::CAMath::Abs(den) < Tiny) {
    return InvalidT;
  }
  return (mP[1] * cs - mP[0] * sn) / den;
}

//______________________________________________________
GPUdi() bool Ray::crossCircleR(float r2, float& cross1, float& cross2) const
{
  // calculate parameters t of intersection with circle of radius r^2
  // calculated as solution of equation
  // t^2*mDistXY2 +- sqrt( mXDxPlusYDy^2 - mDistXY2*(mR02 - r^2) )
  //
  float det = mXDxPlusYDy2 - mDistXY2 * (mR02 - r2);
  if (det < 0)
    return false; // no intersection
  float detRed = o2::gpu::CAMath::Sqrt(det) * mDistXY2i;
  cross1 = mXDxPlusYDyRed + detRed; // (-mXDxPlusYDy + det)*mDistXY2i;
  cross2 = mXDxPlusYDyRed - detRed; // (-mXDxPlusYDy - det)*mDistXY2i;
  return true;
}

//______________________________________________________
GPUdi() float Ray::crossRadial(const MatLayerCyl& lr, int sliceID) const
{
  // calculate t of crossing with phimin of layer's slice sliceID
  return crossRadial(lr.getSliceCos(sliceID), lr.getSliceSin(sliceID));
}

//______________________________________________________
GPUdi() float Ray::crossZ(float z) const
{
  // calculate t of crossing XY plane at Z
  return o2::gpu::CAMath::Abs(mD[2]) > Tiny ? (z - mP[2]) / mD[2] : InvalidT;
}

//______________________________________________________
GPUdi() bool Ray::validateZRange(float& cpar1, float& cpar2, const MatLayerCyl& lr) const
{
  // make sure that estimated crossing parameters are compatible
  // with Z coverage of the layer
  MatLayerCyl::RangeStatus zout0 = lr.isZOutside(getZ(cpar1)), zout1 = lr.isZOutside(getZ(cpar2));
  if (zout0 == zout1) { // either both points outside w/o crossing or boht inside
    return zout0 == MatLayerCyl::Within ? true : false;
  }
  // at least 1 point is outside, but there is a crossing
  if (zout0 != MatLayerCyl::Within) {
    cpar1 = crossZ(zout0 == MatLayerCyl::Below ? lr.getZMin() : lr.getZMax());
  }
  if (zout1 != MatLayerCyl::Within) {
    cpar2 = crossZ(zout1 == MatLayerCyl::Below ? lr.getZMin() : lr.getZMax());
  }
  return true;
}

//______________________________________________________
GPUdi() void Ray::getMinMaxR2(float& rmin2, float& rmax2) const
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
    // use r^2(t) = mR02 + t^2 (mD[0]^2+mD[1]^2) + 2t*mXDxPlusYDy
    float xMin = mP[0] + mXDxPlusYDyRed * mD[0], yMin = mP[1] + mXDxPlusYDyRed * mD[1];
    rmin2 = xMin * xMin + yMin * yMin;
  }
}

} // namespace base
} // namespace o2

#endif
