// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackParametrization.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  Oct 1, 2020
/// @brief

/*
  24/09/2020: Added new data member for abs. charge. This is needed for uniform treatment of tracks with non-standard
  charge: 0 (for V0s) and e.g. 2 for hypernuclei.
  In the aliroot AliExternalTrackParam this was treated by derived classes using virtual methods, which we don't use in O2.
  The meaning of mP[kQ2Pt] remains exactly the same, except for q=0 case: in this case the mP[kQ2Pt] is just an alias to
  1/pT, regardless of its sign, and the getCurvature() will 0 (because mAbsCharge is 0).
  The methods returning lab momentum or its combination account for eventual q>1.
 */

#ifndef INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATION_H_
#define INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATION_H_

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonMath.h"
#include "GPUCommonArray.h"
#include "GPUROOTCartesianFwd.h"

#ifndef GPUCA_GPUCODE_DEVICE
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iosfwd>
#endif

#ifndef GPUCA_ALIGPUCODE //Used only by functions that are hidden on the GPU
#include "ReconstructionDataFormats/BaseCluster.h"
#include <string>
#endif

#include "CommonConstants/MathConstants.h"
#include "MathUtils/Utils.h"
#include "MathUtils/Primitive2D.h"
#include "ReconstructionDataFormats/PID.h"

#include "ReconstructionDataFormats/TrackUtils.h"

namespace o2
{
template <typename T>
class BaseCluster;

namespace dataformats
{
class VertexBase;
class DCA;
} // namespace dataformats

namespace track
{
// aliases for track elements
enum ParLabels : int { kY,
                       kZ,
                       kSnp,
                       kTgl,
                       kQ2Pt };
enum CovLabels : int {
  kSigY2,
  kSigZY,
  kSigZ2,
  kSigSnpY,
  kSigSnpZ,
  kSigSnp2,
  kSigTglY,
  kSigTglZ,
  kSigTglSnp,
  kSigTgl2,
  kSigQ2PtY,
  kSigQ2PtZ,
  kSigQ2PtSnp,
  kSigQ2PtTgl,
  kSigQ2Pt2
};

enum DirType : int { DirInward = -1,
                     DirAuto = 0,
                     DirOutward = 1 };

constexpr int kNParams = 5, kCovMatSize = 15, kLabCovMatSize = 21;

constexpr float kCY2max = 100 * 100, // SigmaY<=100cm
  kCZ2max = 100 * 100,               // SigmaZ<=100cm
  kCSnp2max = 1 * 1,                 // SigmaSin<=1
  kCTgl2max = 1 * 1,                 // SigmaTan<=1
  kC1Pt2max = 100 * 100,             // Sigma1/Pt<=100 1/GeV
  kMostProbablePt = 0.6f,            // Most Probable Pt (GeV), for running with Bz=0
  kCalcdEdxAuto = -999.f;            // value indicating request for dedx calculation

// access to covariance matrix by row and column
GPUconstexpr() int CovarMap[kNParams][kNParams] = {{0, 1, 3, 6, 10},
                                                   {1, 2, 4, 7, 11},
                                                   {3, 4, 5, 8, 12},
                                                   {6, 7, 8, 9, 13},
                                                   {10, 11, 12, 13, 14}};

// access to covariance matrix diagonal elements
GPUconstexpr() int DiagMap[kNParams] = {0, 2, 5, 9, 14};

constexpr float HugeF = o2::constants::math::VeryBig;

template <typename value_T = float>
class TrackParametrization
{ // track parameterization, kinematics only.

 public:
  using value_t = value_T;
  using dim2_t = gpu::gpustd::array<value_t, 2>;
  using dim3_t = gpu::gpustd::array<value_t, 3>;
  using params_t = gpu::gpustd::array<value_t, kNParams>;

#ifndef GPUCA_GPUCODE_DEVICE
  static_assert(std::is_floating_point_v<value_t>);
#endif

  GPUdDefault() TrackParametrization() = default;
  GPUd() TrackParametrization(value_t x, value_t alpha, const params_t& par, int charge = 1);
  GPUd() TrackParametrization(const dim3_t& xyz, const dim3_t& pxpypz, int charge, bool sectorAlpha = true);
  GPUdDefault() TrackParametrization(const TrackParametrization&) = default;
  GPUdDefault() TrackParametrization(TrackParametrization&&) = default;
  GPUdDefault() TrackParametrization& operator=(const TrackParametrization& src) = default;
  GPUdDefault() TrackParametrization& operator=(TrackParametrization&& src) = default;
  GPUdDefault() ~TrackParametrization() = default;

  GPUd() void set(value_t x, value_t alpha, const params_t& par, int charge = 1);
  GPUd() const value_t* getParams() const;
  GPUd() value_t getParam(int i) const;
  GPUd() value_t getX() const;
  GPUd() value_t getAlpha() const;
  GPUd() value_t getY() const;
  GPUd() value_t getZ() const;
  GPUd() value_t getSnp() const;
  GPUd() value_t getTgl() const;
  GPUd() value_t getQ2Pt() const;
  GPUd() value_t getCharge2Pt() const;
  GPUd() int getAbsCharge() const;
  GPUd() PID getPID() const;
  GPUd() void setPID(const PID pid);

  /// calculate cos^2 and cos of track direction in rphi-tracking
  GPUd() value_t getCsp2() const;
  GPUd() value_t getCsp() const;

  GPUd() void setX(value_t v);
  GPUd() void setParam(value_t v, int i);
  GPUd() void setAlpha(value_t v);
  GPUd() void setY(value_t v);
  GPUd() void setZ(value_t v);
  GPUd() void setSnp(value_t v);
  GPUd() void setTgl(value_t v);
  GPUd() void setQ2Pt(value_t v);
  GPUd() void setAbsCharge(int q);

  // derived getters
  GPUd() bool getXatLabR(value_t r, value_t& x, value_t bz, DirType dir = DirAuto) const;
  GPUd() void getCircleParamsLoc(value_t bz, o2::math_utils::CircleXY<value_t>& circle) const;
  GPUd() void getCircleParams(value_t bz, o2::math_utils::CircleXY<value_t>& circle, value_t& sna, value_t& csa) const;
  GPUd() void getLineParams(o2::math_utils::IntervalXY<value_t>& line, value_t& sna, value_t& csa) const;
  GPUd() value_t getCurvature(value_t b) const;
  GPUd() int getCharge() const;
  GPUd() int getSign() const;
  GPUd() value_t getPhi() const;
  GPUd() value_t getPhiPos() const;

  GPUd() value_t getPtInv() const;
  GPUd() value_t getP2Inv() const;
  GPUd() value_t getP2() const;
  GPUd() value_t getPInv() const;
  GPUd() value_t getP() const;
  GPUd() value_t getPt() const;

  GPUd() value_t getTheta() const;
  GPUd() value_t getEta() const;
  GPUd() math_utils::Point3D<value_t> getXYZGlo() const;
  GPUd() void getXYZGlo(dim3_t& xyz) const;
  GPUd() bool getPxPyPzGlo(dim3_t& pxyz) const;
  GPUd() bool getPosDirGlo(gpu::gpustd::array<value_t, 9>& posdirp) const;

  // methods for track params estimate at other point
  GPUd() bool getYZAt(value_t xk, value_t b, value_t& y, value_t& z) const;
  GPUd() value_t getZAt(value_t xk, value_t b) const;
  GPUd() value_t getYAt(value_t xk, value_t b) const;
  GPUd() math_utils::Point3D<value_t> getXYZGloAt(value_t xk, value_t b, bool& ok) const;

  // parameters manipulation
  GPUd() bool correctForELoss(value_t xrho, bool anglecorr = false, value_t dedx = kCalcdEdxAuto);
  GPUd() bool rotateParam(value_t alpha);
  GPUd() bool propagateParamTo(value_t xk, value_t b);
  GPUd() bool propagateParamTo(value_t xk, const dim3_t& b);

  GPUd() bool propagateParamToDCA(const math_utils::Point3D<value_t>& vtx, value_t b, dim2_t* dca = nullptr, value_t maxD = 999.f);

  GPUd() void invertParam();

  GPUd() bool isValid() const;
  GPUd() void invalidate();

  GPUd() uint16_t getUserField() const;
  GPUd() void setUserField(uint16_t v);

  GPUd() void printParam() const;
#ifndef GPUCA_ALIGPUCODE
  std::string asString() const;
#endif

 protected:
  GPUd() void updateParam(value_t delta, int i);
  GPUd() void updateParams(const value_t delta[kNParams]);

 private:
  //
  static constexpr value_t InvalidX = -99999.f;
  value_t mX = 0.f;             /// X of track evaluation
  value_t mAlpha = 0.f;         /// track frame angle
  value_t mP[kNParams] = {0.f}; /// 5 parameters: Y,Z,sin(phi),tg(lambda),q/pT
  char mAbsCharge = 1;          /// Extra info about the abs charge, to be taken into account only if not 1
  PID mPID{};                   /// 8 bit PID
  uint16_t mUserField = 0;      /// field provided to user

  ClassDefNV(TrackParametrization, 3);
};

//____________________________________________________________
template <typename value_T>
GPUdi() TrackParametrization<value_T>::TrackParametrization(value_t x, value_t alpha, const params_t& par, int charge)
  : mX{x}, mAlpha{alpha}, mAbsCharge{char(gpu::CAMath::Abs(charge))}
{
  // explicit constructor
  for (int i = 0; i < kNParams; i++) {
    mP[i] = par[i];
  }
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::set(value_t x, value_t alpha, const params_t& par, int charge)
{
  mX = x;
  mAlpha = alpha;
  mAbsCharge = char(gpu::CAMath::Abs(charge));
  for (int i = 0; i < kNParams; i++) {
    mP[i] = par[i];
  }
}

//____________________________________________________________
template <typename value_T>
GPUdi() const typename TrackParametrization<value_T>::value_t* TrackParametrization<value_T>::getParams() const
{
  return mP;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getParam(int i) const
{
  return mP[i];
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getX() const
{
  return mX;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getAlpha() const
{
  return mAlpha;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getY() const
{
  return mP[kY];
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getZ() const
{
  return mP[kZ];
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getSnp() const
{
  return mP[kSnp];
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getTgl() const
{
  return mP[kTgl];
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getQ2Pt() const
{
  return mP[kQ2Pt];
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getCharge2Pt() const
{
  return mAbsCharge ? mP[kQ2Pt] : 0.f;
}

//____________________________________________________________
template <typename value_T>
GPUdi() int TrackParametrization<value_T>::getAbsCharge() const
{
  return mAbsCharge;
}

//____________________________________________________________
template <typename value_T>
GPUdi() PID TrackParametrization<value_T>::getPID() const
{
  return mPID;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setPID(const PID pid)
{
  mPID = pid;
  setAbsCharge(pid.getCharge());
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getCsp2() const
{
  const value_t csp2 = (1.f - mP[kSnp]) * (1.f + mP[kSnp]);
  return csp2 > o2::constants::math::Almost0 ? csp2 : o2::constants::math::Almost0;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getCsp() const
{
  return gpu::CAMath::Sqrt(getCsp2());
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setX(value_t v)
{
  mX = v;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setParam(value_t v, int i)
{
  mP[i] = v;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setAlpha(value_t v)
{
  mAlpha = v;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setY(value_t v)
{
  mP[kY] = v;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setZ(value_t v)
{
  mP[kZ] = v;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setSnp(value_t v)
{
  mP[kSnp] = v;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setTgl(value_t v)
{
  mP[kTgl] = v;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setQ2Pt(value_t v)
{
  mP[kQ2Pt] = v;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setAbsCharge(int q)
{
  mAbsCharge = gpu::CAMath::Abs(q);
}

//_______________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::getCircleParamsLoc(value_t bz, o2::math_utils::CircleXY<value_t>& c) const
{
  // get circle params in track local frame, for straight line just set to local coordinates
  c.rC = getCurvature(bz);
  // treat as straight track if sagitta between the vertex and middle of TPC is below 0.01 cm
  constexpr value_t MinSagitta = 0.01f, TPCMidR = 160.f, MinCurv = 8 * MinSagitta / (TPCMidR * TPCMidR);
  if (gpu::CAMath::Abs(c.rC) > MinCurv) {
    c.rC = 1.f / getCurvature(bz);
    value_t sn = getSnp(), cs = gpu::CAMath::Sqrt((1.f - sn) * (1.f + sn));
    c.xC = getX() - sn * c.rC; // center in tracking
    c.yC = getY() + cs * c.rC; // frame. Note: r is signed!!!
    c.rC = gpu::CAMath::Abs(c.rC);
  } else {
    c.rC = 0.f; // signal straight line
    c.xC = getX();
    c.yC = getY();
  }
}

//_______________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::getCircleParams(value_t bz, o2::math_utils::CircleXY<value_t>& c, value_t& sna, value_t& csa) const
{
  // get circle params in loc and lab frame, for straight line just set to global coordinates
  getCircleParamsLoc(bz, c);
  o2::math_utils::detail::sincos(getAlpha(), sna, csa);
  o2::math_utils::detail::rotateZ<value_t>(c.xC, c.yC, c.xC, c.yC, sna, csa); // center in global frame
}

//_______________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::getLineParams(o2::math_utils::IntervalXY<value_t>& ln, value_t& sna, value_t& csa) const
{
  // get line parameterization as { x = x0 + xSlp*t, y = y0 + ySlp*t }
  o2::math_utils::detail::sincos(getAlpha(), sna, csa);
  o2::math_utils::detail::rotateZ<value_t>(getX(), getY(), ln.getX0(), ln.getY0(), sna, csa); // reference point in global frame
  value_t snp = getSnp(), csp = gpu::CAMath::Sqrt((1.f - snp) * (1.f + snp));
  ln.setDX(csp * csa - snp * sna);
  ln.setDY(snp * csa + csp * sna);
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getCurvature(value_t b) const
{
  return mAbsCharge ? mP[kQ2Pt] * b * o2::constants::math::B2C : 0.;
}

//____________________________________________________________
template <typename value_T>
GPUdi() int TrackParametrization<value_T>::getCharge() const
{
  return getSign() > 0 ? mAbsCharge : -mAbsCharge;
}

//____________________________________________________________
template <typename value_T>
GPUdi() int TrackParametrization<value_T>::getSign() const
{
  return mAbsCharge ? (mP[kQ2Pt] > 0.f ? 1 : -1) : 0;
}

//_______________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getPhi() const
{
  // track pt direction phi (in 0:2pi range)
  value_t phi = gpu::CAMath::ASin(getSnp()) + getAlpha();
  math_utils::detail::bringTo02Pi<value_t>(phi);
  return phi;
}

//_______________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getPhiPos() const
{
  // angle of track position (in -pi:pi range)
  value_t phi = gpu::CAMath::ATan2(getY(), getX()) + getAlpha();
  math_utils::detail::bringTo02Pi<value_t>(phi);
  return phi;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getPtInv() const
{
  // return the inverted track pT
  const value_t ptInv = gpu::CAMath::Abs(mP[kQ2Pt]);
  return (mAbsCharge > 1) ? ptInv / mAbsCharge : ptInv;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getP2Inv() const
{
  // return the inverted track momentum^2
  const value_t p2 = mP[kQ2Pt] * mP[kQ2Pt] / (1.f + getTgl() * getTgl());
  return (mAbsCharge > 1) ? p2 * mAbsCharge * mAbsCharge : p2;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getP2() const
{
  // return the track momentum^2
  const value_t p2inv = getP2Inv();
  return (p2inv > o2::constants::math::Almost0) ? 1.f / p2inv : o2::constants::math::VeryBig;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getPInv() const
{
  // return the inverted track momentum^2
  const value_t pInv = gpu::CAMath::Abs(mP[kQ2Pt]) / gpu::CAMath::Sqrt(1.f + getTgl() * getTgl());
  return (mAbsCharge > 1) ? pInv / mAbsCharge : pInv;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getP() const
{
  // return the track momentum
  const value_t pInv = getPInv();
  return (pInv > o2::constants::math::Almost0) ? 1.f / pInv : o2::constants::math::VeryBig;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getPt() const
{
  // return the track transverse momentum
  value_t ptI = gpu::CAMath::Abs(mP[kQ2Pt]);
  if (mAbsCharge > 1) {
    ptI /= mAbsCharge;
  }
  return (ptI > o2::constants::math::Almost0) ? 1.f / ptI : o2::constants::math::VeryBig;
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getTheta() const
{
  return constants::math::PIHalf - gpu::CAMath::ATan(mP[3]);
}

//____________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getEta() const
{
  return -gpu::CAMath::Log(gpu::CAMath::Tan(0.5f * getTheta()));
}

//_______________________________________________________
template <typename value_T>
GPUdi() math_utils::Point3D<typename TrackParametrization<value_T>::value_t> TrackParametrization<value_T>::getXYZGlo() const
{
#ifndef GPUCA_ALIGPUCODE
  return math_utils::Rotation2D<value_t>(getAlpha())(math_utils::Point3D<value_t>(getX(), getY(), getZ()));
#else // mockup on GPU without ROOT
  float sina, cosa;
  gpu::CAMath::SinCos(getAlpha(), sina, cosa);
  return math_utils::Point3D<value_t>(cosa * getX() + sina * getY(), cosa * getY() - sina * getX(), getZ());
#endif
}

//_______________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::getXYZGlo(dim3_t& xyz) const
{
  // track coordinates in lab frame
  xyz[0] = getX();
  xyz[1] = getY();
  xyz[2] = getZ();
  math_utils::detail::rotateZ<value_t>(xyz, getAlpha());
}

//_______________________________________________________
template <typename value_T>
GPUdi() math_utils::Point3D<typename TrackParametrization<value_T>::value_t> TrackParametrization<value_T>::getXYZGloAt(value_t xk, value_t b, bool& ok) const
{
  //----------------------------------------------------------------
  // estimate global X,Y,Z in global frame at given X
  //----------------------------------------------------------------
  value_t y = 0.f, z = 0.f;
  ok = getYZAt(xk, b, y, z);
  if (ok) {
#ifndef GPUCA_ALIGPUCODE
    return math_utils::Rotation2D<value_t>(getAlpha())(math_utils::Point3D<value_t>(xk, y, z));
#else // mockup on GPU without ROOT
    float sina, cosa;
    gpu::CAMath::SinCos(getAlpha(), sina, cosa);
    return math_utils::Point3D<value_t>(cosa * xk + sina * y, cosa * y - sina * xk, z);
#endif
  } else {
    return math_utils::Point3D<value_t>();
  }
}

//____________________________________________________________
template <typename value_T>
GPUdi() bool TrackParametrization<value_T>::isValid() const
{
  return mX != InvalidX;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::invalidate()
{
  mX = InvalidX;
}

template <typename value_T>
GPUdi() uint16_t TrackParametrization<value_T>::getUserField() const
{
  return mUserField;
}

template <typename value_T>
GPUdi() void TrackParametrization<value_T>::setUserField(uint16_t v)
{
  mUserField = v;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::updateParam(value_t delta, int i)
{
  mP[i] += delta;
}

//____________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrization<value_T>::updateParams(const value_t delta[kNParams])
{
  for (int i = kNParams; i--;) {
    mP[i] += delta[i];
  }
}

} // namespace track
} // namespace o2

#endif /* INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATION_H_ */
