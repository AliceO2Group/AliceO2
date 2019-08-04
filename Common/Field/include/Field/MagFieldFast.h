// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MagFieldFast.h
/// \brief Definition of the fast magnetic field parametrization MagFieldFast
/// \author ruben.shahoyan@cern.ch
#ifndef ALICEO2_FIELD_MAGFIELDFAST_H_
#define ALICEO2_FIELD_MAGFIELDFAST_H_

#include <Rtypes.h>
#include <string>
#include "MathUtils/Cartesian3D.h"

namespace o2
{
namespace field
{
// Fast polynomial parametrization of Alice magnetic field, to be used for reconstruction.
// Solenoid part fitted by Shuto Yamasaki from AliMagWrapCheb in the |Z|<260Interface and R<500 cm
// Dipole part: to do
class MagFieldFast
{
 public:
  enum { kNSolRRanges = 5,
         kNSolZRanges = 22,
         kNQuadrants = 4,
         kNPolCoefs = 20 };
  enum EDim { kX,
              kY,
              kZ,
              kNDim };
  struct SolParam {
    float parBxyz[kNDim][kNPolCoefs];
  };

  MagFieldFast(const std::string inpFName = "");
  MagFieldFast(float factor, int nomField = 5, const std::string inpFmt = "$(O2_ROOT)/share/Common/maps/sol%dk.txt");
  MagFieldFast(const MagFieldFast& src) = default;
  ~MagFieldFast() = default;

  bool LoadData(const std::string inpFName);

  bool Field(const double xyz[3], double bxyz[3]) const;
  bool Field(const float xyz[3], float bxyz[3]) const;
  bool Field(const Point3D<float> xyz, float bxyz[3]) const;
  bool GetBcomp(EDim comp, const double xyz[3], double& b) const;
  bool GetBcomp(EDim comp, const float xyz[3], float& b) const;
  bool GetBcomp(EDim comp, const Point3D<float> xyz, double& b) const;
  bool GetBcomp(EDim comp, const Point3D<float> xyz, float& b) const;

  bool GetBx(const double xyz[3], double& bx) const { return GetBcomp(kX, xyz, bx); }
  bool GetBx(const float xyz[3], float& bx) const { return GetBcomp(kX, xyz, bx); }
  bool GetBy(const double xyz[3], double& by) const { return GetBcomp(kY, xyz, by); }
  bool GetBy(const float xyz[3], float& by) const { return GetBcomp(kY, xyz, by); }
  bool GetBz(const double xyz[3], double& bz) const { return GetBcomp(kZ, xyz, bz); }
  bool GetBz(const float xyz[3], float& bz) const { return GetBcomp(kZ, xyz, bz); }
  void setFactorSol(float v = 1.f) { mFactorSol = v; }
  float getFactorSol() const { return mFactorSol; }

 protected:
  bool GetSegment(float x, float y, float z, int& zSeg, int& rSeg, int& quadrant) const;
  static const float kSolR2Max[kNSolRRanges]; // Rmax2 of each range
  static const float kSolZMax;                // max |Z| for solenoid parametrization

  int GetQuadrant(float x, float y) const
  {
    /// get point quadrant
    return y > 0 ? (x > 0 ? 0 : 1) : (x > 0 ? 3 : 2);
  }

  float CalcPol(const float* cf, float x, float y, float z) const;

 private:
  float mFactorSol; // scaling factor
  SolParam mSolPar[kNSolRRanges][kNSolZRanges][kNQuadrants];

  ClassDef(MagFieldFast, 1);
};

inline float MagFieldFast::CalcPol(const float* cf, float x, float y, float z) const
{
  /** calculate polynomial
   *   cf[0] + cf[1]*x + cf[2]*y + cf[3]*z + cf[4]*xx + cf[5]*xy + cf[6]*xz + cf[7]*yy + cf[8]*yz + cf[9]*zz +
   *   cf[10]*xxx + cf[11]*xxy + cf[12]*xxz + cf[13]*xyy + cf[14]*xyz + cf[15]*xzz + cf[16]*yyy + cf[17]*yyz +
   *cf[18]*yzz + cf[19]*zzz
  **/

  float val = cf[0] + x * (cf[1] + x * (cf[4] + x * cf[10] + y * cf[11] + z * cf[12]) + y * (cf[5] + z * cf[14])) +
              y * (cf[2] + y * (cf[7] + x * cf[13] + y * cf[16] + z * cf[17]) + z * (cf[8])) +
              z * (cf[3] + z * (cf[9] + x * cf[15] + y * cf[18] + z * cf[19]) + x * (cf[6]));

  return val;
}
} // namespace field
} // namespace o2

#endif
