// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Track.h
/// \brief Base track model for the Barrel, params only, w/o covariance
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_BASE_TRACK
#define ALICEO2_BASE_TRACK

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
#endif

#include "CommonConstants/MathConstants.h"
#include "MathUtils/Utils.h"

//Forward declarations, since we cannot include the headers if we eventually want to use track.h on GPU
namespace ROOT
{
namespace Math
{
template <class T, unsigned int D1, unsigned int D2, class R>
class SMatrix;
template <class T, unsigned int D>
class MatRepSym;
template <class T, unsigned int D1, unsigned int D2>
class MatRepStd;
template <class CoordSystem, class Tag>
class PositionVector3D;
template <class T>
class Cartesian3D;
class DefaultCoordinateSystemTag;
} // namespace Math
} // namespace ROOT
template <typename T>
using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;

namespace o2
{
template <typename T>
class BaseCluster;

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

constexpr int kNParams = 5, kCovMatSize = 15, kLabCovMatSize = 21;

constexpr float kCY2max = 100 * 100, // SigmaY<=100cm
  kCZ2max = 100 * 100,               // SigmaZ<=100cm
  kCSnp2max = 1 * 1,                 // SigmaSin<=1
  kCTgl2max = 1 * 1,                 // SigmaTan<=1
  kC1Pt2max = 100 * 100,             // Sigma1/Pt<=100 1/GeV
  kCalcdEdxAuto = -999.f;            // value indicating request for dedx calculation

// access to covariance matrix by row and column
constexpr int CovarMap[kNParams][kNParams] = {{0, 1, 3, 6, 10},
                                              {1, 2, 4, 7, 11},
                                              {3, 4, 5, 8, 12},
                                              {6, 7, 8, 9, 13},
                                              {10, 11, 12, 13, 14}};

// access to covariance matrix diagonal elements
constexpr int DiagMap[kNParams] = {0, 2, 5, 9, 14};

constexpr float HugeF = 1e33; // large float as dummy value

// helper function
float BetheBlochSolid(float bg, float rho = 2.33f, float kp1 = 0.20f, float kp2 = 3.00f, float meanI = 173e-9f,
                      float meanZA = 0.49848f);
void g3helx3(float qfield, float step, std::array<float, 7>& vect);

class TrackPar
{ // track parameterization, kinematics only.
 public:
  TrackPar() = default;
  TrackPar(float x, float alpha, const std::array<float, kNParams>& par);
  TrackPar(const std::array<float, 3>& xyz, const std::array<float, 3>& pxpypz, int sign, bool sectorAlpha = true);
  TrackPar(const TrackPar&) = default;
  TrackPar& operator=(const TrackPar& src) = default;
  ~TrackPar() = default;

  const float* getParams() const { return mP; }
  float getParam(int i) const { return mP[i]; }
  float getX() const { return mX; }
  float getAlpha() const { return mAlpha; }
  float getY() const { return mP[kY]; }
  float getZ() const { return mP[kZ]; }
  float getSnp() const { return mP[kSnp]; }
  float getTgl() const { return mP[kTgl]; }
  float getQ2Pt() const { return mP[kQ2Pt]; }

  void setX(float v) { mX = v; }
  void setParam(float v, int i) { mP[i] = v; }
  void setAlpha(float v) { mAlpha = v; }
  void setY(float v) { mP[kY] = v; }
  void setZ(float v) { mP[kZ] = v; }
  void setSnp(float v) { mP[kSnp] = v; }
  void setTgl(float v) { mP[kTgl] = v; }
  void setQ2Pt(float v) { mP[kQ2Pt] = v; }

  // derived getters
  float getCurvature(float b) const { return mP[kQ2Pt] * b * o2::constants::math::B2C; }
  float getSign() const { return mP[kQ2Pt] > 0 ? 1.f : -1.f; }
  float getPhi() const;
  float getPhiPos() const;

  float getP2Inv() const;
  float getP2() const;
  float getPInv() const;
  float getP() const;
  float getPt() const;

  float getTheta() const { return constants::math::PIHalf - std::atan(mP[3]); }
  float getEta() const { return -std::log(std::tan(0.5 * getTheta())); }
  Point3D<float> getXYZGlo() const;
  void getXYZGlo(std::array<float, 3>& xyz) const;
  bool getPxPyPzGlo(std::array<float, 3>& pxyz) const;
  bool getPosDirGlo(std::array<float, 9>& posdirp) const;

  // methods for track params estimate at other point
  bool getYZAt(float xk, float b, float& y, float& z) const;
  float getZAt(float xk, float b) const;
  float getYAt(float xk, float b) const;
  Point3D<float> getXYZGloAt(float xk, float b, bool& ok) const;

  // parameters manipulation
  bool rotateParam(float alpha);
  bool propagateParamTo(float xk, float b);
  bool propagateParamTo(float xk, const std::array<float, 3>& b);
  void invertParam();

  void printParam() const;

 protected:
  void updateParam(float delta, int i) { mP[i] += delta; }
  void updateParams(const float delta[kNParams])
  {
    for (int i = kNParams; i--;) {
      mP[i] += delta[i];
    }
  }
  // derived getters

 private:
  //
  float mX = 0.f;             /// X of track evaluation
  float mAlpha = 0.f;         /// track frame angle
  float mP[kNParams] = {0.f}; /// 5 parameters: Y,Z,sin(phi),tg(lambda),q/pT

  ClassDefNV(TrackPar, 1);
};

class TrackParCov : public TrackPar
{ // track+error parameterization
  using MatrixDSym5 = ROOT::Math::SMatrix<double, kNParams, kNParams, ROOT::Math::MatRepSym<double, kNParams>>;
  using MatrixD5 = ROOT::Math::SMatrix<double, kNParams, kNParams, ROOT::Math::MatRepStd<double, kNParams, kNParams>>;

 public:
  TrackParCov() : TrackPar{} {}
  TrackParCov(float x, float alpha, const std::array<float, kNParams>& par, const std::array<float, kCovMatSize>& cov);
  TrackParCov(const std::array<float, 3>& xyz, const std::array<float, 3>& pxpypz,
              const std::array<float, kLabCovMatSize>& cv, int sign, bool sectorAlpha = true);
  TrackParCov(const TrackParCov& src) = default;
  TrackParCov& operator=(const TrackParCov& src) = default;
  ~TrackParCov() = default;

  const float* getCov() const { return mC; }
  float getSigmaY2() const { return mC[kSigY2]; }
  float getSigmaZY() const { return mC[kSigZY]; }
  float getSigmaZ2() const { return mC[kSigZ2]; }
  float getSigmaSnpY() const { return mC[kSigSnpY]; }
  float getSigmaSnpZ() const { return mC[kSigSnpZ]; }
  float getSigmaSnp2() const { return mC[kSigSnp2]; }
  float getSigmaTglY() const { return mC[kSigTglY]; }
  float getSigmaTglZ() const { return mC[kSigTglZ]; }
  float getSigmaTglSnp() const { return mC[kSigTglSnp]; }
  float getSigmaTgl2() const { return mC[kSigTgl2]; }
  float getSigma1PtY() const { return mC[kSigQ2PtY]; }
  float getSigma1PtZ() const { return mC[kSigQ2PtZ]; }
  float getSigma1PtSnp() const { return mC[kSigQ2PtSnp]; }
  float getSigma1PtTgl() const { return mC[kSigQ2PtTgl]; }
  float getSigma1Pt2() const { return mC[kSigQ2Pt2]; }
  float getCovarElem(int i, int j) const { return mC[CovarMap[i][j]]; }
  float getDiagError2(int i) const { return mC[DiagMap[i]]; }
  void print() const;

  // parameters + covmat manipulation
  bool rotate(float alpha);
  bool propagateTo(float xk, float b);
  bool propagateTo(float xk, const std::array<float, 3>& b);
  void invert();

  float getPredictedChi2(const std::array<float, 2>& p, const std::array<float, 3>& cov) const;

  template <typename T>
  float getPredictedChi2(const BaseCluster<T>& p) const
  {
    const std::array<float, 2> pyz = {p.getY(), p.getZ()};
    const std::array<float, 3> cov = {p.getSigmaY2(), p.getSigmaYZ(), p.getSigmaZ2()};
    return getPredictedChi2(pyz, cov);
  }

  float getPredictedChi2(const TrackParCov& rhs) const;

  void buildCombinedCovMatrix(const TrackParCov& rhs, MatrixDSym5& cov) const;
  float getPredictedChi2(const TrackParCov& rhs, MatrixDSym5& covToSet) const;
  bool update(const TrackParCov& rhs, const MatrixDSym5& covInv);

  bool update(const std::array<float, 2>& p, const std::array<float, 3>& cov);

  template <typename T>
  bool update(const BaseCluster<T>& p)
  {
    const std::array<float, 2> pyz = {p.getY(), p.getZ()};
    const std::array<float, 3> cov = {p.getSigmaY2(), p.getSigmaYZ(), p.getSigmaZ2()};
    return update(pyz, cov);
  }

  bool update(const TrackParCov& rhs);

  bool correctForMaterial(float x2x0, float xrho, float mass, bool anglecorr = false, float dedx = kCalcdEdxAuto);

  void resetCovariance(float s2 = 0);
  void checkCovariance();

 protected:
  void setCov(float v, int i) { mC[i] = v; }
  void updateCov(const float delta[kCovMatSize])
  {
    for (int i = kCovMatSize; i--;)
      mC[i] += delta[i];
  }

 protected:
  float mC[kCovMatSize] = {0.f}; // 15 covariance matrix elements

  ClassDefNV(TrackParCov, 1);
};

//____________________________________________________________
inline TrackPar::TrackPar(float x, float alpha, const std::array<float, kNParams>& par) : mX{x}, mAlpha{alpha}
{
  // explicit constructor
  std::copy(par.begin(), par.end(), mP);
}

//_______________________________________________________
inline void TrackPar::getXYZGlo(std::array<float, 3>& xyz) const
{
  // track coordinates in lab frame
  xyz[0] = getX();
  xyz[1] = getY();
  xyz[2] = getZ();
  utils::RotateZ(xyz, getAlpha());
}

//_______________________________________________________
inline float TrackPar::getPhi() const
{
  float phi = asinf(getSnp()) + getAlpha();
  utils::BringToPMPi(phi);
  return phi;
}

#ifndef GPUCA_ALIGPUCODE //These functions clash with GPU code and are thus hidden
//_______________________________________________________
inline Point3D<float> TrackPar::getXYZGlo() const
{
  return Rotation2D(getAlpha())(Point3D<float>(getX(), getY(), getZ()));
}

//_______________________________________________________
inline Point3D<float> TrackPar::getXYZGloAt(float xk, float b, bool& ok) const
{
  //----------------------------------------------------------------
  // estimate global X,Y,Z in global frame at given X
  //----------------------------------------------------------------
  float y = 0.f, z = 0.f;
  ok = getYZAt(xk, b, y, z);
  return ok ? Rotation2D(getAlpha())(Point3D<float>(xk, y, z)) : Point3D<float>();
}
#endif

//_______________________________________________________
inline float TrackPar::getPhiPos() const
{
  // angle of track position
  float phi = atan2f(getY(), getX()) + getAlpha();
  utils::BringToPMPi(phi);
  return phi;
}

//____________________________________________________________
inline float TrackPar::getP2Inv() const
{
  // return the inverted track momentum^2
  return getQ2Pt() * getQ2Pt() / (1.f + getTgl() * getTgl());
}

//____________________________________________________________
inline float TrackPar::getPInv() const
{
  // return the inverted track momentum^2
  return fabs(getQ2Pt()) / sqrtf(1.f + getTgl() * getTgl());
}

//____________________________________________________________
inline float TrackPar::getP2() const
{
  // return the track momentum^2
  auto p2inv = getP2Inv();
  return (p2inv > o2::constants::math::Almost0) ? 1. / p2inv : o2::constants::math::VeryBig;
}

//____________________________________________________________
inline float TrackPar::getP() const
{
  // return the track momentum
  float pInv = getPInv();
  return (pInv > o2::constants::math::Almost0) ? 1. / pInv : o2::constants::math::VeryBig;
}

//____________________________________________________________
inline float TrackPar::getPt() const
{
  // return the track transverse momentum
  float ptI = fabs(getQ2Pt());
  return (ptI > o2::constants::math::Almost0) ? 1.f / ptI : o2::constants::math::VeryBig;
}

//============================================================

//____________________________________________________________
inline TrackParCov::TrackParCov(float x, float alpha, const std::array<float, kNParams>& par,
                                const std::array<float, kCovMatSize>& cov)
  : TrackPar{x, alpha, par}
{
  // explicit constructor
  std::copy(cov.begin(), cov.end(), mC);
}
} // namespace track
} // namespace o2

#endif
