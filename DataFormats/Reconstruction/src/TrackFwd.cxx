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

#include "ReconstructionDataFormats/TrackFwd.h"
#include "Math/MatrixFunctions.h"

namespace o2
{
namespace track
{
using namespace std;

//_________________________________________________________________________
TrackParCovFwd::TrackParCovFwd(const Double_t z, const SMatrix5& parameters, const SMatrix55Sym& covariances, const Double_t chi2)
{
  setZ(z);
  setParameters(parameters);
  setCovariances(covariances);
  setTrackChi2(chi2);
}

//__________________________________________________________________________
void TrackParFwd::propagateParamToZlinear(double zEnd)
{
  // Track parameters linearly extrapolated to the plane at "zEnd".

  if (getZ() == zEnd) {
    return; // nothing to be done if same z
  }

  // Compute track parameters
  auto dZ = (zEnd - getZ());
  auto phi0 = getPhi();
  auto [sinphi0, cosphi0] = o2::math_utils::sincosd(phi0);
  auto invtanl0 = 1.0 / getTanl();
  auto n = dZ * invtanl0;
  mParameters(0) += n * cosphi0;
  mParameters(1) += n * sinphi0;
  mZ = zEnd;
}

//__________________________________________________________________________
void TrackParCovFwd::propagateToZlinear(double zEnd)
{
  // Track parameters and their covariances linearly extrapolated to the plane at "zEnd".

  // Calculate the jacobian related to the track parameters extrapolated to "zEnd"
  auto dZ = (zEnd - getZ());
  auto phi0 = getPhi();
  auto tanl0 = getTanl();
  auto invtanl0 = 1.0 / tanl0;
  auto [sinphi0, cosphi0] = o2::math_utils::sincosd(phi0);
  auto n = dZ * invtanl0;
  auto m = n * invtanl0;

  // Extrapolate track parameters to "zEnd"
  mParameters(0) += n * cosphi0;
  mParameters(1) += n * sinphi0;
  setZ(zEnd);

  // Calculate Jacobian
  SMatrix55Std jacob = ROOT::Math::SMatrixIdentity();
  jacob(0, 2) = -n * sinphi0;
  jacob(0, 3) = -m * cosphi0;
  jacob(1, 2) = n * cosphi0;
  jacob(1, 3) = -m * sinphi0;

  // Extrapolate track parameter covariances to "zEnd"
  setCovariances(ROOT::Math::Similarity(jacob, mCovariances));
}

//__________________________________________________________________________
void TrackParFwd::propagateParamToZquadratic(double zEnd, double zField)
{
  // Track parameters extrapolated to the plane at "zEnd" considering a helix

  if (getZ() == zEnd) {
    return; // nothing to be done if same z
  }

  // Compute track parameters
  auto dZ = (zEnd - getZ());
  auto phi0 = getPhi();
  auto [sinphi0, cosphi0] = o2::math_utils::sincosd(phi0);
  auto invtanl0 = 1.0 / getTanl();
  auto invqpt0 = getInvQPt();
  auto Hz = std::copysign(1, zField);
  auto k = TMath::Abs(o2::constants::math::B2C * zField);
  auto n = dZ * invtanl0;
  auto theta = -invqpt0 * dZ * k * invtanl0;

  mParameters(0) += n * cosphi0 - 0.5 * n * theta * Hz * sinphi0;
  mParameters(1) += n * sinphi0 + 0.5 * n * theta * Hz * cosphi0;
  mParameters(2) += Hz * theta;
  setZ(zEnd);
}

//__________________________________________________________________________
void TrackParCovFwd::propagateToZquadratic(double zEnd, double zField)
{
  // Extrapolate track parameters and covariances matrix to "zEnd"
  // using quadratic track model

  if (getZ() == zEnd) {
    return; // nothing to be done if same z
  }

  // Compute track parameters
  auto dZ = (zEnd - getZ());
  auto phi0 = getPhi();
  auto [sinphi0, cosphi0] = o2::math_utils::sincosd(phi0);
  auto invtanl0 = 1.0 / getTanl();
  auto invqpt0 = getInvQPt();
  auto Hz = std::copysign(1, zField);
  auto k = TMath::Abs(o2::constants::math::B2C * zField);
  auto n = dZ * invtanl0;
  auto m = n * invtanl0;
  auto theta = -invqpt0 * dZ * k * invtanl0;

  // Extrapolate track parameters to "zEnd"
  mParameters(0) += n * cosphi0 - 0.5 * n * theta * Hz * sinphi0;
  mParameters(1) += n * sinphi0 + 0.5 * n * theta * Hz * cosphi0;
  mParameters(2) += Hz * theta;
  mZ = zEnd;

  // Calculate Jacobian
  SMatrix55Std jacob = ROOT::Math::SMatrixIdentity();
  jacob(0, 2) = -n * theta * 0.5 * Hz * cosphi0 - n * sinphi0;
  jacob(0, 3) = Hz * m * theta * sinphi0 - m * cosphi0;
  jacob(0, 4) = k * m * 0.5 * Hz * dZ * sinphi0;
  jacob(1, 2) = -n * theta * 0.5 * Hz * sinphi0 + n * cosphi0;
  jacob(1, 3) = -Hz * m * theta * cosphi0 - m * sinphi0;
  jacob(1, 4) = -k * m * 0.5 * Hz * dZ * cosphi0;
  jacob(2, 3) = -Hz * theta * invtanl0;
  jacob(2, 4) = -Hz * k * n;

  // Extrapolate track parameter covariances to "zEnd"
  setCovariances(ROOT::Math::Similarity(jacob, mCovariances));
}

//__________________________________________________________________________
void TrackParFwd::propagateParamToZhelix(double zEnd, double zField)
{
  // Track parameters extrapolated to the plane at "zEnd"
  // using helix track model

  if (getZ() == zEnd) {
    return; // nothing to be done if same z
  }

  // Compute track parameters
  auto dZ = (zEnd - getZ());
  auto phi0 = getPhi();
  auto tanl0 = getTanl();
  auto invtanl0 = 1.0 / tanl0;
  auto invqpt0 = getInvQPt();
  auto qpt0 = 1.0 / invqpt0;
  auto [sinphi0, cosphi0] = o2::math_utils::sincosd(phi0);

  auto k = TMath::Abs(o2::constants::math::B2C * zField);
  auto invk = 1.0 / k;
  auto theta = -invqpt0 * dZ * k * invtanl0;
  auto [sintheta, costheta] = o2::math_utils::sincosd(theta);
  auto Hz = std::copysign(1, zField);
  auto Y = sinphi0 * qpt0 * invk;
  auto X = cosphi0 * qpt0 * invk;
  auto YC = Y * costheta;
  auto YS = Y * sintheta;
  auto XC = X * costheta;
  auto XS = X * sintheta;

  // Extrapolate track parameters to "zEnd"
  mParameters(0) += Hz * (Y - YC) - XS;
  mParameters(1) += Hz * (-X + XC) - YS;
  mParameters(2) += Hz * theta;
  mZ = zEnd;
}

//__________________________________________________________________________
void TrackParCovFwd::propagateToZhelix(double zEnd, double zField)
{
  // Extrapolate track parameters and covariances matrix to "zEnd"
  // using helix track model

  auto dZ = (zEnd - getZ());
  auto phi0 = getPhi();
  auto tanl0 = getTanl();
  auto invtanl0 = 1.0 / tanl0;
  auto invqpt0 = getInvQPt();
  auto qpt0 = 1.0 / invqpt0;
  auto [sinphi0, cosphi0] = o2::math_utils::sincosd(phi0);
  auto k = TMath::Abs(o2::constants::math::B2C * zField);
  auto invk = 1.0 / k;
  auto theta = -invqpt0 * dZ * k * invtanl0;
  auto [sintheta, costheta] = o2::math_utils::sincosd(theta);
  auto Hz = std::copysign(1, zField);
  auto L = qpt0 * qpt0 * invk;
  auto N = dZ * invtanl0 * qpt0;
  auto O = sintheta * cosphi0;
  auto P = sinphi0 * costheta;
  auto R = sinphi0 * sintheta;
  auto S = cosphi0 * costheta;
  auto Y = sinphi0 * qpt0 * invk;
  auto X = cosphi0 * qpt0 * invk;
  auto YC = Y * costheta;
  auto YS = Y * sintheta;
  auto XC = X * costheta;
  auto XS = X * sintheta;
  auto T = qpt0 * costheta;
  auto U = qpt0 * sintheta;
  auto V = qpt0;
  auto n = dZ * invtanl0;
  auto m = n * invtanl0;

  // Extrapolate track parameters to "zEnd"
  mParameters(0) += Hz * (Y - YC) - XS;
  mParameters(1) += Hz * (-X + XC) - YS;
  mParameters(2) += Hz * theta;
  mZ = zEnd;

  // Calculate Jacobian
  SMatrix55Std jacob = ROOT::Math::SMatrixIdentity();
  jacob(0, 2) = Hz * X - Hz * XC + YS;
  jacob(0, 3) = Hz * R * m - S * m;
  jacob(0, 4) = -Hz * N * R + Hz * T * Y - Hz * V * Y + N * S + U * X;
  jacob(1, 2) = Hz * Y - Hz * YC - XS;
  jacob(1, 3) = -Hz * O * m - P * m;
  jacob(1, 4) = Hz * N * O - Hz * T * X + Hz * V * X + N * P + U * Y;
  jacob(2, 3) = -Hz * theta * invtanl0;
  jacob(2, 4) = -Hz * k * n;

  // Extrapolate track parameter covariances to "zEnd"
  setCovariances(ROOT::Math::Similarity(jacob, mCovariances));
}

//__________________________________________________________________________
void TrackParCovFwd::propagateToZ(double zEnd, double zField)
{
  // Security for zero B field
  if (zField == 0.0) {
    propagateToZlinear(zEnd);
    return;
  }

  // Extrapolate track parameters and covariances matrix to "zEnd"
  // Parameters: helix track model; Error propagation: Quadratic

  auto dZ = (zEnd - getZ());
  auto phi0 = getPhi();
  auto tanl0 = getTanl();
  auto invtanl0 = 1.0 / tanl0;
  auto invqpt0 = getInvQPt();
  auto qpt0 = 1.0 / invqpt0;
  auto [sinphi0, cosphi0] = o2::math_utils::sincosd(phi0);
  auto k = TMath::Abs(o2::constants::math::B2C * zField);
  auto invk = 1.0 / k;
  auto theta = -invqpt0 * dZ * k * invtanl0;
  auto [sintheta, costheta] = o2::math_utils::sincosd(theta);
  auto Hz = std::copysign(1, zField);
  auto Y = sinphi0 * qpt0 * invk;
  auto X = cosphi0 * qpt0 * invk;
  auto YC = Y * costheta;
  auto YS = Y * sintheta;
  auto XC = X * costheta;
  auto XS = X * sintheta;
  auto n = dZ * invtanl0;
  auto m = n * invtanl0;

  // Extrapolate track parameters to "zEnd"
  // Helix
  mParameters(0) += Hz * (Y - YC) - XS;
  mParameters(1) += Hz * (-X + XC) - YS;
  mParameters(2) += Hz * theta;
  mZ = zEnd;

  // Jacobian (quadratic)
  SMatrix55Std jacob = ROOT::Math::SMatrixIdentity();
  jacob(0, 2) = -n * theta * 0.5 * Hz * cosphi0 - n * sinphi0;
  jacob(0, 3) = Hz * m * theta * sinphi0 - m * cosphi0;
  jacob(0, 4) = k * m * 0.5 * Hz * dZ * sinphi0;
  jacob(1, 2) = -n * theta * 0.5 * Hz * sinphi0 + n * cosphi0;
  jacob(1, 3) = -Hz * m * theta * cosphi0 - m * sinphi0;
  jacob(1, 4) = -k * m * 0.5 * Hz * dZ * cosphi0;
  jacob(2, 3) = -Hz * theta * invtanl0;
  jacob(2, 4) = -Hz * k * n;

  // Extrapolate track parameter covariances to "zEnd"
  setCovariances(ROOT::Math::Similarity(jacob, mCovariances));
}

//__________________________________________________________________________
bool TrackParCovFwd::update(const std::array<float, 2>& p, const std::array<float, 2>& cov)
{
  /// Kalman update step: computes new track parameters with a new cluster position and uncertainties
  /// The current track is expected to have been propagated to the cluster z position

  using SVector2 = ROOT::Math::SVector<double, 2>;
  using SMatrix22 = ROOT::Math::SMatrix<double, 2>;
  using SMatrix25 = ROOT::Math::SMatrix<double, 2, 5>;
  using SMatrix52 = ROOT::Math::SMatrix<double, 5, 2>;

  SMatrix55Sym I = ROOT::Math::SMatrixIdentity();
  SMatrix25 H_k;
  SMatrix22 V_k;
  SVector2 m_k(p[0], p[1]), r_k_kminus1;
  V_k(0, 0) = cov[0];
  V_k(1, 1) = cov[1];
  H_k(0, 0) = 1.0;
  H_k(1, 1) = 1.0;

  // Covariance of residuals
  SMatrix22 invResCov = (V_k + ROOT::Math::Similarity(H_k, mCovariances));
  invResCov.Invert();

  // Kalman Gain Matrix
  SMatrix52 K_k = mCovariances * ROOT::Math::Transpose(H_k) * invResCov;

  // Update Parameters
  r_k_kminus1 = m_k - H_k * mParameters; // Residuals of prediction
  mParameters = mParameters + K_k * r_k_kminus1;

  // Update covariances Matrix
  SMatrix55Std updatedCov;
  auto& CP = mCovariances;
  auto& sigmax2 = cov[0];
  auto& sigmay2 = cov[1];
  auto A = 1. / (sigmax2 * sigmay2 + sigmax2 * CP(1, 1) + sigmay2 * CP(0, 0) + CP(0, 0) * CP(1, 1) - CP(0, 1) * CP(0, 1));
  auto AX = A * sigmax2;
  auto AY = A * sigmay2;
  auto B = sigmax2 * sigmay2;
  auto C = (sigmax2 + CP(0, 0)) * (sigmay2 + CP(1, 1));
  auto D = 1 / (-C + CP(0, 1) * CP(0, 1));
  auto E = sigmax2 + CP(0, 0);
  auto F = sigmay2 + CP(1, 1);
  auto G = -C + CP(0, 1) * CP(0, 1);

  // Explicit evaluation of "updatedCov = (I - K_k * H_k) * mCovariances"
  updatedCov(0, 0) = AX * (sigmay2 * CP(0, 0) + CP(0, 0) * CP(1, 1) - CP(0, 1) * CP(0, 1));
  updatedCov(0, 1) = AX * sigmay2 * CP(0, 1);
  updatedCov(0, 2) = AX * (sigmay2 * CP(0, 2) - CP(0, 1) * CP(1, 2) + CP(0, 2) * CP(1, 1));
  updatedCov(0, 3) = AX * (sigmay2 * CP(0, 3) - CP(0, 1) * CP(1, 3) + CP(0, 3) * CP(1, 1));
  updatedCov(0, 4) = AX * (sigmay2 * CP(0, 4) - CP(0, 1) * CP(1, 4) + CP(0, 4) * CP(1, 1));
  updatedCov(1, 1) = AY * (sigmax2 * CP(1, 1) + CP(0, 0) * CP(1, 1) - CP(0, 1) * CP(0, 1));
  updatedCov(1, 2) = AY * (sigmax2 * CP(1, 2) + CP(0, 0) * CP(1, 2) - CP(0, 1) * CP(0, 2));
  updatedCov(1, 3) = AY * (sigmax2 * CP(1, 3) + CP(0, 0) * CP(1, 3) - CP(0, 1) * CP(0, 3));
  updatedCov(1, 4) = AY * (sigmax2 * CP(1, 4) + CP(0, 0) * CP(1, 4) - CP(0, 1) * CP(0, 4));
  updatedCov(2, 2) = D * (G * CP(2, 2) - CP(0, 2) * (-F * CP(0, 2) + CP(0, 1) * CP(1, 2)) - CP(1, 2) * (-E * CP(1, 2) + CP(0, 1) * CP(0, 2)));
  updatedCov(2, 3) = D * (G * CP(2, 3) - CP(0, 2) * (-F * CP(0, 3) + CP(0, 1) * CP(1, 3)) - CP(1, 2) * (-E * CP(1, 3) + CP(0, 1) * CP(0, 3)));
  updatedCov(2, 4) = D * (G * CP(2, 4) - CP(0, 2) * (-F * CP(0, 4) + CP(0, 1) * CP(1, 4)) - CP(1, 2) * (-E * CP(1, 4) + CP(0, 1) * CP(0, 4)));
  updatedCov(3, 3) = D * (G * CP(3, 3) - CP(0, 3) * (-F * CP(0, 3) + CP(0, 1) * CP(1, 3)) - CP(1, 3) * (-E * CP(1, 3) + CP(0, 1) * CP(0, 3)));
  updatedCov(3, 4) = D * (G * CP(3, 4) - CP(0, 3) * (-F * CP(0, 4) + CP(0, 1) * CP(1, 4)) - CP(1, 3) * (-E * CP(1, 4) + CP(0, 1) * CP(0, 4)));
  updatedCov(4, 4) = D * (G * CP(4, 4) - CP(0, 4) * (-F * CP(0, 4) + CP(0, 1) * CP(1, 4)) - CP(1, 4) * (-E * CP(1, 4) + CP(0, 1) * CP(0, 4)));

  mCovariances(0, 0) = updatedCov(0, 0);
  mCovariances(0, 1) = updatedCov(0, 1);
  mCovariances(0, 2) = updatedCov(0, 2);
  mCovariances(0, 3) = updatedCov(0, 3);
  mCovariances(0, 4) = updatedCov(0, 4);
  mCovariances(1, 1) = updatedCov(1, 1);
  mCovariances(1, 2) = updatedCov(1, 2);
  mCovariances(1, 3) = updatedCov(1, 3);
  mCovariances(1, 4) = updatedCov(1, 4);
  mCovariances(2, 2) = updatedCov(2, 2);
  mCovariances(2, 3) = updatedCov(2, 3);
  mCovariances(2, 4) = updatedCov(2, 4);
  mCovariances(3, 3) = updatedCov(3, 3);
  mCovariances(3, 4) = updatedCov(3, 4);
  mCovariances(4, 4) = updatedCov(4, 4);

  auto addChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);
  mTrackChi2 += addChi2Track;

  return true;
}

//__________________________________________________________________________
void TrackParCovFwd::addMCSEffect(double x_over_X0)
{
  /// Add multiple Coulomb scattering effects to the track covariances.
  ///  Only angular and pt MCS effects are evaluated.
  ///  * x_over_X0 is the fraction of the radiation lenght (x/X0).
  ///  * No energy loss correction.

  if (x_over_X0 == 0) { // Nothing to do
    return;
  }

  auto phi0 = getPhi();
  auto tanl0 = getTanl();
  auto invtanl0 = 1.0 / tanl0;
  auto invqpt0 = getInvQPt();

  auto [sinphi0, cosphi0] = o2::math_utils::sincosd(phi0);

  auto csclambda = TMath::Abs(TMath::Sqrt(1 + tanl0 * tanl0) * invtanl0);
  auto pathLengthOverX0 = x_over_X0 * csclambda; //

  // Angular dispersion square of the track (variance) in a plane perpendicular to the trajectory
  auto sigmathetasq = 0.0136 * getInverseMomentum();
  sigmathetasq *= sigmathetasq * pathLengthOverX0;

  // Get covariance matrix
  SMatrix55Sym newParamCov(getCovariances());

  auto A = tanl0 * tanl0 + 1;

  newParamCov(2, 2) += sigmathetasq * A;

  newParamCov(3, 3) += sigmathetasq * A * A;

  newParamCov(4, 4) += sigmathetasq * tanl0 * tanl0 * invqpt0 * invqpt0;

  // Set new covariances
  setCovariances(newParamCov);
}

//_______________________________________________________
void TrackParFwd::getCircleParams(float bz, o2::math_utils::CircleXY<float>& c, float& sna, float& csa) const
{
  c.rC = getCurvature(bz);
  constexpr double MinCurv = 1e-6;
  if (std::abs(c.rC) > MinCurv) {
    c.rC = 1.f / getCurvature(bz);
    double sn = getSnp(), cs = std::sqrt((1.f - sn) * (1.f + sn));
    c.xC = getX() - sn * c.rC; // center in tracking
    c.yC = getY() + cs * c.rC; // frame. Note: r is signed!!!
    c.rC = std::abs(c.rC);
  } else {
    c.rC = 0.f; // signal straight line
    c.xC = getX();
    c.yC = getY();
  }
}
//________________________________________________________________
bool TrackParCovFwd::propagateToVtxhelixWithMCS(double z, const std::array<float, 2>& p, const std::array<float, 2>& cov, double field, double x_over_X0)
{
  // Propagate fwd track to vertex using helix model, adding MCS effects
  addMCSEffect(x_over_X0);
  propagateToZhelix(z, field);
  return update(p, cov);
}
//________________________________________________________________
bool TrackParCovFwd::propagateToVtxlinearWithMCS(double z, const std::array<float, 2>& p, const std::array<float, 2>& cov, double x_over_X0)
{
  // Propagate fwd track to vertex using linear model, adding MCS effects
  addMCSEffect(x_over_X0);
  propagateToZlinear(z);
  return update(p, cov);
}

} // namespace track
} // namespace o2
