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

/// \file TrackFwd.h
/// \brief Base forward track model, params only, w/o covariance
///
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#ifndef ALICEO2_BASE_TRACKFWD
#define ALICEO2_BASE_TRACKFWD

#include <Rtypes.h>
#include <TMath.h>
#include "Math/SMatrix.h"
#include "MathUtils/Utils.h"
#include "ReconstructionDataFormats/TrackUtils.h"
#include "MathUtils/Primitive2D.h"

namespace o2::track
{

using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix55Std = ROOT::Math::SMatrix<double, 5>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

class TrackParFwd
{ // Forward track parameterization, kinematics only.
 public:
  TrackParFwd() = default;
  ~TrackParFwd() = default;

  TrackParFwd(const TrackParFwd& tp) = default;
  TrackParFwd& operator=(const TrackParFwd& tp) = default;
  TrackParFwd(TrackParFwd&&) = delete;
  TrackParFwd& operator=(TrackParFwd&&) = delete;

  /// return Z coordinate (cm)
  Double_t getZ() const { return mZ; }
  /// set Z coordinate (cm)
  void setZ(Double_t z) { mZ = z; }
  Double_t getX() const { return mParameters(0); }
  void setX(Double_t x) { mParameters(0) = x; }

  Double_t getY() const { return mParameters(1); }
  void setY(Double_t y) { mParameters(1) = y; }

  void setPhi(Double_t phi) { mParameters(2) = phi; }
  Double_t getPhi() const { return mParameters(2); }

  Double_t getSnp() const
  {
    return o2::math_utils::sin(mParameters(2));
  }

  Double_t getCsp2() const
  {
    auto snp = o2::math_utils::sin(mParameters(2));
    Double_t csp;
    csp = std::sqrt((1. - snp) * (1. + snp));
    return csp * csp;
  }

  void setTanl(Double_t tanl) { mParameters(3) = tanl; }
  Double_t getTanl() const { return mParameters(3); }

  Double_t getTgl() const { return mParameters(3); } // for the sake of helixhelper

  void setInvQPt(Double_t invqpt) { mParameters(4) = invqpt; }
  Double_t getInvQPt() const { return mParameters(4); } // return Inverse charged pt
  Double_t getPt() const { return TMath::Abs(1.f / mParameters(4)); }
  Double_t getInvPt() const { return TMath::Abs(mParameters(4)); }
  Double_t getPx() const { return TMath::Cos(getPhi()) * getPt(); }                   // return px
  Double_t getPy() const { return TMath::Sin(getPhi()) * getPt(); }                   // return py
  Double_t getPz() const { return getTanl() * getPt(); }                              // return pz
  Double_t getP() const { return getPt() * TMath::Sqrt(1. + getTanl() * getTanl()); } // return total momentum
  Double_t getInverseMomentum() const { return 1.f / getP(); }

  Double_t getTheta() const { return TMath::PiOver2() - TMath::ATan(getTanl()); }
  Double_t getEta() const { return -TMath::Log(TMath::Tan(getTheta() / 2)); } // return total momentum

  Double_t getCurvature(double b) const
  {
    auto invqpt = getInvQPt();
    return o2::constants::math::B2C * b * invqpt;
  }

  /// return the charge (assumed forward motion)
  Double_t getCharge() const { return TMath::Sign(1., mParameters(4)); }
  /// set the charge (assumed forward motion)
  void setCharge(Double_t charge)
  {
    if (charge * mParameters(4) < 0.) {
      mParameters(4) *= -1.;
    }
  }

  /// return track parameters
  const SMatrix5& getParameters() const { return mParameters; }
  /// set track parameters
  void setParameters(const SMatrix5& parameters) { mParameters = parameters; }
  /// add track parameters
  void addParameters(const SMatrix5& parameters) { mParameters += parameters; }

  /// return the chi2 of the track when the associated cluster was attached
  Double_t getTrackChi2() const { return mTrackChi2; }
  /// set the chi2 of the track when the associated cluster was attached
  void setTrackChi2(Double_t chi2) { mTrackChi2 = chi2; }

  // Track parameter propagation
  void propagateParamToZlinear(double zEnd);
  void propagateParamToZquadratic(double zEnd, double zField);
  void propagateParamToZhelix(double zEnd, double zField);
  void getCircleParams(float bz, o2::math_utils::CircleXY<float>& c, float& sna, float& csa) const;

 protected:
  Double_t mZ = 0.; ///< Z coordinate (cm)

  /// Track parameters ordered as follow:      <pre>
  /// X       = X coordinate   (cm)
  /// Y       = Y coordinate   (cm)
  /// PHI     = azimutal angle
  /// TANL    = tangent of \lambda (dip angle)
  /// INVQPT    = Inverse transverse momentum (GeV/c ** -1) times charge (assumed forward motion)  </pre>
  SMatrix5 mParameters{};   ///< \brief Track parameters
  Double_t mTrackChi2 = 0.; ///< Chi2 of the track when the associated cluster was attached

  ClassDefNV(TrackParFwd, 1);
};

class TrackParCovFwd : public TrackParFwd
{ // Forward track+error parameterization
 public:
  using TrackParFwd::TrackParFwd; // inherit base constructors

  TrackParCovFwd() = default;
  ~TrackParCovFwd() = default;
  TrackParCovFwd& operator=(const TrackParCovFwd& tpf) = default;
  TrackParCovFwd(const Double_t z, const SMatrix5& parameters, const SMatrix55Sym& covariances, const Double_t chi2);

  const SMatrix55Sym& getCovariances() const { return mCovariances; }
  void setCovariances(const SMatrix55Sym& covariances) { mCovariances = covariances; }
  void deleteCovariances() { mCovariances = SMatrix55Sym(); }

  Double_t getSigma2X() const { return mCovariances(0, 0); }
  Double_t getSigma2Y() const { return mCovariances(1, 1); }
  Double_t getSigmaXY() const { return mCovariances(0, 1); }
  Double_t getSigma2Phi() const { return mCovariances(2, 2); }
  Double_t getSigma2Tanl() const { return mCovariances(3, 3); }
  Double_t getSigma2InvQPt() const { return mCovariances(4, 4); }

  // Propagate parameters and covariances matrix
  void propagateToZlinear(double zEnd);
  void propagateToZquadratic(double zEnd, double zField);
  void propagateToZhelix(double zEnd, double zField);
  void propagateToZ(double zEnd, double zField); // Parameters: helix; errors: quadratic

  // Add Multiple Coulomb Scattering effects
  void addMCSEffect(double x2X0);

  // Kalman filter/fitting
  bool update(const std::array<float, 2>& p, const std::array<float, 2>& cov);

  // Propagate fwd track to vertex including MCS effects
  bool propagateToVtxhelixWithMCS(double z, const std::array<float, 2>& p, const std::array<float, 2>& cov, double field, double x_over_X0);
  bool propagateToVtxlinearWithMCS(double z, const std::array<float, 2>& p, const std::array<float, 2>& cov, double x_over_X0);

 private:
  /// Covariance matrix of track parameters, ordered as follows:    <pre>
  ///  <X,X>         <Y,X>           <PHI,X>       <TANL,X>        <INVQPT,X>
  ///  <X,Y>         <Y,Y>           <PHI,Y>       <TANL,Y>        <INVQPT,Y>
  /// <X,PHI>       <Y,PHI>         <PHI,PHI>     <TANL,PHI>      <INVQPT,PHI>
  /// <X,TANL>      <Y,TANL>       <PHI,TANL>     <TANL,TANL>     <INVQPT,TANL>
  /// <X,INVQPT>   <Y,INVQPT>     <PHI,INVQPT>   <TANL,INVQPT>   <INVQPT,INVQPT>  </pre>
  SMatrix55Sym mCovariances{}; ///< \brief Covariance matrix of track parameters
  ClassDefNV(TrackParCovFwd, 1);
};

} // namespace o2::track

#endif
