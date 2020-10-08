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

#include "GPUCommonRtypes.h"

#ifndef __OPENCL__
#include <algorithm>
#include <array>
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

//Forward declarations, since we cannot include the headers if we eventually want to use track.h on GPU
#include "GPUROOTCartesianFwd.h"

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
constexpr int CovarMap[kNParams][kNParams] = {{0, 1, 3, 6, 10},
                                              {1, 2, 4, 7, 11},
                                              {3, 4, 5, 8, 12},
                                              {6, 7, 8, 9, 13},
                                              {10, 11, 12, 13, 14}};

// access to covariance matrix diagonal elements
constexpr int DiagMap[kNParams] = {0, 2, 5, 9, 14};

constexpr float HugeF = o2::constants::math::VeryBig;

template <typename value_T = float>
class TrackParametrization
{ // track parameterization, kinematics only.

 public:
  using value_t = value_T;
  using dim2_t = std::array<value_t, 2>;
  using dim3_t = std::array<value_t, 3>;
  using params_t = std::array<value_t, kNParams>;

  static_assert(std::is_floating_point_v<value_t>);

  TrackParametrization() = default;
  TrackParametrization(value_t x, value_t alpha, const params_t& par, int charge = 1);
  TrackParametrization(const dim3_t& xyz, const dim3_t& pxpypz, int charge, bool sectorAlpha = true);
  TrackParametrization(const TrackParametrization&) = default;
  TrackParametrization(TrackParametrization&&) = default;
  TrackParametrization& operator=(const TrackParametrization& src) = default;
  TrackParametrization& operator=(TrackParametrization&& src) = default;
  ~TrackParametrization() = default;

  const value_t* getParams() const;
  value_t getParam(int i) const;
  value_t getX() const;
  value_t getAlpha() const;
  value_t getY() const;
  value_t getZ() const;
  value_t getSnp() const;
  value_t getTgl() const;
  value_t getQ2Pt() const;
  value_t getCharge2Pt() const;
  int getAbsCharge() const;
  PID getPID() const;
  void setPID(const PID pid);

  /// calculate cos^2 and cos of track direction in rphi-tracking
  value_t getCsp2() const;
  value_t getCsp() const;

  void setX(value_t v);
  void setParam(value_t v, int i);
  void setAlpha(value_t v);
  void setY(value_t v);
  void setZ(value_t v);
  void setSnp(value_t v);
  void setTgl(value_t v);
  void setQ2Pt(value_t v);
  void setAbsCharge(int q);

  // derived getters
  bool getXatLabR(value_t r, value_t& x, value_t bz, DirType dir = DirAuto) const;
  void getCircleParamsLoc(value_t bz, o2::math_utils::CircleXY& circle) const;
  void getCircleParams(value_t bz, o2::math_utils::CircleXY& circle, value_t& sna, value_t& csa) const;
  void getLineParams(o2::math_utils::IntervalXY& line, value_t& sna, value_t& csa) const;
  value_t getCurvature(value_t b) const;
  int getCharge() const;
  int getSign() const;
  value_t getPhi() const;
  value_t getPhiPos() const;

  value_t getPtInv() const;
  value_t getP2Inv() const;
  value_t getP2() const;
  value_t getPInv() const;
  value_t getP() const;
  value_t getPt() const;

  value_t getTheta() const;
  value_t getEta() const;
  math_utils::Point3D<value_t> getXYZGlo() const;
  void getXYZGlo(dim3_t& xyz) const;
  bool getPxPyPzGlo(dim3_t& pxyz) const;
  bool getPosDirGlo(std::array<value_t, 9>& posdirp) const;

  // methods for track params estimate at other point
  bool getYZAt(value_t xk, value_t b, value_t& y, value_t& z) const;
  value_t getZAt(value_t xk, value_t b) const;
  value_t getYAt(value_t xk, value_t b) const;
  math_utils::Point3D<value_t> getXYZGloAt(value_t xk, value_t b, bool& ok) const;

  // parameters manipulation
  bool correctForELoss(value_t xrho, value_t mass, bool anglecorr = false, value_t dedx = kCalcdEdxAuto);
  bool rotateParam(value_t alpha);
  bool propagateParamTo(value_t xk, value_t b);
  bool propagateParamTo(value_t xk, const dim3_t& b);

  bool propagateParamToDCA(const math_utils::Point3D<value_t>& vtx, value_t b, dim2_t* dca = nullptr, value_t maxD = 999.f);

  void invertParam();

  bool isValid() const;
  void invalidate();

  uint16_t getUserField() const;
  void setUserField(uint16_t v);

#ifndef GPUCA_ALIGPUCODE
  void printParam() const;
  std::string asString() const;
#endif

 protected:
  void updateParam(value_t delta, int i);
  void updateParams(const value_t delta[kNParams]);

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
inline TrackParametrization<value_T>::TrackParametrization(value_t x, value_t alpha, const params_t& par, int charge)
  : mX{x}, mAlpha{alpha}, mAbsCharge{char(std::abs(charge))}
{
  // explicit constructor
  std::copy(par.begin(), par.end(), mP);
}

//____________________________________________________________
template <typename value_T>
inline const typename TrackParametrization<value_T>::value_t* TrackParametrization<value_T>::getParams() const
{
  return mP;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getParam(int i) const
{
  return mP[i];
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getX() const
{
  return mX;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getAlpha() const
{
  return mAlpha;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getY() const
{
  return mP[kY];
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getZ() const
{
  return mP[kZ];
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getSnp() const
{
  return mP[kSnp];
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getTgl() const
{
  return mP[kTgl];
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getQ2Pt() const
{
  return mP[kQ2Pt];
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getCharge2Pt() const
{
  return mAbsCharge ? mP[kQ2Pt] : 0.f;
}

//____________________________________________________________
template <typename value_T>
inline int TrackParametrization<value_T>::getAbsCharge() const
{
  return mAbsCharge;
}

//____________________________________________________________
template <typename value_T>
inline PID TrackParametrization<value_T>::getPID() const
{
  return mPID;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::setPID(const PID pid)
{
  mPID = pid;
  setAbsCharge(pid.getCharge());
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getCsp2() const
{
  value_t csp2 = (1.f - mP[kSnp]) * (1.f + mP[kSnp]);
  return csp2 > o2::constants::math::Almost0 ? csp2 : o2::constants::math::Almost0;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getCsp() const
{
  return sqrtf(getCsp2());
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::setX(value_t v)
{
  mX = v;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::setParam(value_t v, int i)
{
  mP[i] = v;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::setAlpha(value_t v)
{
  mAlpha = v;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::setY(value_t v)
{
  mP[kY] = v;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::setZ(value_t v)
{
  mP[kZ] = v;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::setSnp(value_t v)
{
  mP[kSnp] = v;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::setTgl(value_t v)
{
  mP[kTgl] = v;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::setQ2Pt(value_t v)
{
  mP[kQ2Pt] = v;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::setAbsCharge(int q)
{
  mAbsCharge = std::abs(q);
}

//_______________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::getCircleParamsLoc(value_t bz, o2::math_utils::CircleXY& c) const
{
  // get circle params in track local frame, for straight line just set to local coordinates
  c.rC = getCurvature(bz);
  // treat as straight track if sagitta between the vertex and middle of TPC is below 0.01 cm
  constexpr value_t MinSagitta = 0.01f, TPCMidR = 160.f, MinCurv = 8 * MinSagitta / (TPCMidR * TPCMidR);
  if (std::abs(c.rC) > MinCurv) {
    c.rC = 1.f / getCurvature(bz);
    value_t sn = getSnp(), cs = sqrtf((1.f - sn) * (1.f + sn));
    c.xC = getX() - sn * c.rC; // center in tracking
    c.yC = getY() + cs * c.rC; // frame. Note: r is signed!!!
    c.rC = fabs(c.rC);
  } else {
    c.rC = 0.f; // signal straight line
    c.xC = getX();
    c.yC = getY();
  }
}

//_______________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::getCircleParams(value_t bz, o2::math_utils::CircleXY& c, value_t& sna, value_t& csa) const
{
  // get circle params in loc and lab frame, for straight line just set to global coordinates
  getCircleParamsLoc(bz, c);
  o2::math_utils::sincos(getAlpha(), sna, csa);
  o2::math_utils::rotateZ(c.xC, c.yC, c.xC, c.yC, sna, csa); // center in global frame
}

//_______________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::getLineParams(o2::math_utils::IntervalXY& ln, value_t& sna, value_t& csa) const
{
  // get line parameterization as { x = x0 + xSlp*t, y = y0 + ySlp*t }
  o2::math_utils::sincos(getAlpha(), sna, csa);
  o2::math_utils::rotateZ(getX(), getY(), ln.xP, ln.yP, sna, csa); // reference point in global frame
  value_t snp = getSnp(), csp = sqrtf((1.f - snp) * (1.f + snp));
  ln.dxP = csp * csa - snp * sna;
  ln.dyP = snp * csa + csp * sna;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getCurvature(value_t b) const
{
  return mAbsCharge ? mP[kQ2Pt] * b * o2::constants::math::B2C : 0.;
}

//____________________________________________________________
template <typename value_T>
inline int TrackParametrization<value_T>::getCharge() const
{
  return getSign() > 0 ? mAbsCharge : -mAbsCharge;
}

//____________________________________________________________
template <typename value_T>
inline int TrackParametrization<value_T>::getSign() const
{
  return mAbsCharge ? (mP[kQ2Pt] > 0.f ? 1 : -1) : 0;
}

//_______________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getPhi() const
{
  // track pt direction phi (in 0:2pi range)
  value_t phi = asinf(getSnp()) + getAlpha();
  math_utils::BringTo02Pi(phi);
  return phi;
}

//_______________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getPhiPos() const
{
  // angle of track position (in -pi:pi range)
  value_t phi = atan2f(getY(), getX()) + getAlpha();
  math_utils::BringTo02Pi(phi);
  return phi;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getPtInv() const
{
  // return the inverted track pT
  const value_t ptInv = fabs(mP[kQ2Pt]);
  return (mAbsCharge > 1) ? ptInv / mAbsCharge : ptInv;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getP2Inv() const
{
  // return the inverted track momentum^2
  const value_t p2 = mP[kQ2Pt] * mP[kQ2Pt] / (1.f + getTgl() * getTgl());
  return (mAbsCharge > 1) ? p2 * mAbsCharge * mAbsCharge : p2;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getP2() const
{
  // return the track momentum^2
  const value_t p2inv = getP2Inv();
  return (p2inv > o2::constants::math::Almost0) ? 1.f / p2inv : o2::constants::math::VeryBig;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getPInv() const
{
  // return the inverted track momentum^2
  const value_t pInv = fabs(mP[kQ2Pt]) / sqrtf(1.f + getTgl() * getTgl());
  return (mAbsCharge > 1) ? pInv / mAbsCharge : pInv;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getP() const
{
  // return the track momentum
  const value_t pInv = getPInv();
  return (pInv > o2::constants::math::Almost0) ? 1.f / pInv : o2::constants::math::VeryBig;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getPt() const
{
  // return the track transverse momentum
  value_t ptI = fabs(mP[kQ2Pt]);
  if (mAbsCharge > 1) {
    ptI /= mAbsCharge;
  }
  return (ptI > o2::constants::math::Almost0) ? 1.f / ptI : o2::constants::math::VeryBig;
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getTheta() const
{
  return constants::math::PIHalf - std::atan(mP[3]);
}

//____________________________________________________________
template <typename value_T>
inline typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getEta() const
{
  return -std::log(std::tan(0.5f * getTheta()));
}

#ifndef GPUCA_ALIGPUCODE //These functions clash with GPU code and are thus hidden
//_______________________________________________________
template <typename value_T>
inline math_utils::Point3D<typename TrackParametrization<value_T>::value_t> TrackParametrization<value_T>::getXYZGlo() const
{
  return math_utils::Rotation2D(getAlpha())(math_utils::Point3D<value_t>(getX(), getY(), getZ()));
}
#endif

//_______________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::getXYZGlo(dim3_t& xyz) const
{
  // track coordinates in lab frame
  xyz[0] = getX();
  xyz[1] = getY();
  xyz[2] = getZ();
  math_utils::RotateZ(xyz, getAlpha());
}

#ifndef GPUCA_ALIGPUCODE //These functions clash with GPU code and are thus hidden
//_______________________________________________________
template <typename value_T>
inline math_utils::Point3D<typename TrackParametrization<value_T>::value_t> TrackParametrization<value_T>::getXYZGloAt(value_t xk, value_t b, bool& ok) const
{
  //----------------------------------------------------------------
  // estimate global X,Y,Z in global frame at given X
  //----------------------------------------------------------------
  value_t y = 0.f, z = 0.f;
  ok = getYZAt(xk, b, y, z);
  return ok ? math_utils::Rotation2D(getAlpha())(math_utils::Point3D<value_t>(xk, y, z)) : math_utils::Point3D<value_t>();
}
#endif

//____________________________________________________________
template <typename value_T>
inline bool TrackParametrization<value_T>::isValid() const
{
  return mX != InvalidX;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::invalidate()
{
  mX = InvalidX;
}

template <typename value_T>
inline uint16_t TrackParametrization<value_T>::getUserField() const
{
  return mUserField;
}

template <typename value_T>
inline void TrackParametrization<value_T>::setUserField(uint16_t v)
{
  mUserField = v;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::updateParam(value_t delta, int i)
{
  mP[i] += delta;
}

//____________________________________________________________
template <typename value_T>
inline void TrackParametrization<value_T>::updateParams(const value_t delta[kNParams])
{
  for (int i = kNParams; i--;) {
    mP[i] += delta[i];
  }
}

} // namespace track
} // namespace o2

#endif /* INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATION_H_ */
