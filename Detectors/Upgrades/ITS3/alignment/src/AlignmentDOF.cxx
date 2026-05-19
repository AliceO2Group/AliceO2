// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITS3Align/AlignmentDOF.h"

#include <array>
#include <cmath>
#include <stdexcept>

#include "ITS3Align/AlignmentMath.h"
#include "ITS3Base/SpecsV2.h"
#include "CommonConstants/MathConstants.h"

namespace
{

void validateDerivativeOutput(const DOFSet& dofSet, Eigen::Ref<Eigen::MatrixXd> out)
{
  if (out.rows() != 3 || out.cols() != dofSet.nDOFs()) {
    throw std::invalid_argument(std::format("Derivative buffer shape {}x{} does not match expected 3x{}",
                                            out.rows(), out.cols(), dofSet.nDOFs()));
  }
  out.setZero();
}

} // namespace

void RigidBodyDOFSet::fillDerivatives(const DerivativeContext& ctx, Eigen::Ref<Eigen::MatrixXd> out) const
{
  validateDerivativeOutput(*this, out);

  const double csp = 1. / std::sqrt(1. + (ctx.tgl * ctx.tgl));
  const double uP = ctx.snp * csp;
  const double vP = ctx.tgl * csp;

  out(0, TX) = uP;
  out(0, TY) = -1.;
  out(0, RX) = ctx.trkZ;
  out(0, RY) = ctx.trkZ * uP;
  out(0, RZ) = -ctx.trkY * uP;

  out(1, TX) = vP;
  out(1, TZ) = -1.;
  out(1, RX) = -ctx.trkY;
  out(1, RY) = ctx.trkZ * vP;
  out(1, RZ) = -ctx.trkY * vP;
}

void LegendreDOFSet::fillDerivatives(const DerivativeContext& ctx, Eigen::Ref<Eigen::MatrixXd> out) const
{
  validateDerivativeOutput(*this, out);
  if (ctx.sensorID < 0 || ctx.layerID < 0) {
    throw std::invalid_argument("LegendreDOFSet requires an ITS3 measurement context");
  }

  const double gloX = ctx.measX * std::cos(ctx.measAlpha);
  const double gloY = ctx.measX * std::sin(ctx.measAlpha);
  const auto [u, v] = o2::its3::align::computeUV(gloX, gloY, ctx.measZ, ctx.sensorID, o2::its3::constants::radii[ctx.layerID]);
  const auto pu = o2::its3::align::legendrePols(mOrder, u);
  const auto pv = o2::its3::align::legendrePols(mOrder, v);
  const double phiWidth = o2::its3::align::getSensorPhiWidth(ctx.sensorID, o2::its3::constants::radii[ctx.layerID]);

  // same intergration as `evaluateLegendreShift' but now for each order separateley
  Eigen::VectorXd arcMismatch = Eigen::VectorXd::Zero(nDOFs());
  if (std::abs(u) > o2::constants::math::Almost0) {
    constexpr std::array<double, 8> x = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498, 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
    constexpr std::array<double, 8> w = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
    const double mid = 0.5 * u;
    const double half = 0.5 * u;
    for (int iq = 0; iq < 8; ++iq) {
      const double up = mid + (half * x[iq]);
      const auto puQ = o2::its3::align::legendrePols(mOrder, up);
      int idx = 0;
      for (int i = 0; i <= mOrder; ++i) {
        for (int j = 0; j <= i; ++j) {
          arcMismatch[idx] += w[iq] * puQ[j] * pv[i - j];
          ++idx;
        }
      }
    }
    arcMismatch *= 0.5 * phiWidth * half;
  }

  int idx = 0;
  for (int i = 0; i <= mOrder; ++i) {
    for (int j = 0; j <= i; ++j) {
      const double basis = pu[j] * pv[i - j];
      out(0, idx) = (ctx.dydx * basis) + arcMismatch[idx];
      out(1, idx) = ctx.dzdx * basis;
      ++idx;
    }
  }
}

void InextensionalDOFSet::fillDerivatives(const DerivativeContext& ctx, Eigen::Ref<Eigen::MatrixXd> out) const
{
  validateDerivativeOutput(*this, out);
  if (ctx.layerID < 0) {
    throw std::invalid_argument("InextensionalDOFSet requires an ITS3 measurement context");
  }

  const double r = o2::its3::constants::radii[ctx.layerID];
  const double phi = std::atan2(r * std::sin(ctx.measAlpha), r * std::cos(ctx.measAlpha));
  const double z = ctx.measZ;

  for (int n = 2; n <= mMaxOrder; ++n) {
    const double sn = std::sin(n * phi);
    const double cn = std::cos(n * phi);
    const double n2 = static_cast<double>(n * n);
    const int off = modeOffset(n);

    out(0, off + 0) = -(z / r) * (n * sn + ctx.dydx * n2 * cn);
    out(1, off + 0) = -cn - ctx.dzdx * (z / r) * n2 * cn;

    out(0, off + 1) = (z / r) * (n * cn - ctx.dydx * n2 * sn);
    out(1, off + 1) = -sn * (1. + ctx.dzdx * (z / r) * n2);

    out(0, off + 2) = -cn + ctx.dydx * n * sn;
    out(1, off + 2) = ctx.dzdx * n * sn;

    out(0, off + 3) = -sn - ctx.dydx * n * cn;
    out(1, off + 3) = -ctx.dzdx * n * cn;
  }

  out(0, alphaIdx()) = z / r;
  out(1, alphaIdx()) = -phi;

  out(0, betaIdx()) = -phi - ctx.dydx;
  out(1, betaIdx()) = -ctx.dzdx;
}
