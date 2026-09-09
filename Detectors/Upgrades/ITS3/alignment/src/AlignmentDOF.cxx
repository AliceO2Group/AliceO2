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

#include <cmath>
#include <stdexcept>

#include "ITS3Align/AlignmentMath.h"
#include "ITS3Base/SpecsV2.h"

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
  out(0, TX) = ctx.dydx;
  out(0, TY) = -1.;
  out(0, RX) = ctx.trkZ;
  out(0, RY) = ctx.trkZ * ctx.dydx;
  out(0, RZ) = -ctx.trkY * ctx.dydx;

  out(1, TX) = ctx.dzdx;
  out(1, TZ) = -1.;
  out(1, RX) = -ctx.trkY;
  out(1, RY) = ctx.trkZ * ctx.dzdx;
  out(1, RZ) = -ctx.trkY * ctx.dzdx;
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

  int idx = 0;
  for (int i = 0; i <= mOrder; ++i) {
    for (int j = 0; j <= i; ++j) {
      const double basis = pu[j] * pv[i - j];
      out(0, idx) = ctx.dydx * basis;
      out(1, idx) = ctx.dzdx * basis;
      ++idx;
    }
  }
}

void InextensionalDOFSet::fillDerivatives(const DerivativeContext& ctx, Eigen::Ref<Eigen::MatrixXd> out) const
{
  validateDerivativeOutput(*this, out);
  if (ctx.sensorID < 0 || ctx.layerID < 0) {
    throw std::invalid_argument("InextensionalDOFSet requires an ITS3 measurement context");
  }

  const double r = o2::its3::constants::radii[ctx.layerID];
  const double gloX = ctx.measX * std::cos(ctx.measAlpha);
  const double gloY = ctx.measX * std::sin(ctx.measAlpha);
  const auto [u, v] = o2::its3::align::computeUV(gloX, gloY, ctx.measZ, ctx.sensorID, r);
  const double cPhi = o2::its3::align::phiScale(r);
  const double zOverR = ctx.measZ / r;

  // The residual derivative for a mode with displacement M = (M_r, M_phi, M_z)
  // along the local (r, phi, z) directions is
  //   row0 = dydx * M_r - M_phi,   row1 = dzdx * M_r - M_z
  // (cf. the rigid-body case, where M = (1,0,0) gives (dydx, dzdx)).
  const auto fill = [&out, &ctx](int idx, double mR, double mPhi, double mZ) {
    out(0, idx) = (ctx.dydx * mR) - mPhi;
    out(1, idx) = (ctx.dzdx * mR) - mZ;
  };

  const int order = std::max(mMaxOrder, hasExtensional() ? mExtOrderPhi : 0);
  const auto pu = o2::its3::align::legendrePols(order, u);
  const auto pu1 = o2::its3::align::legendrePolsD1(mMaxOrder, u);
  const auto pu2 = o2::its3::align::legendrePolsD2(mMaxOrder, u);

  for (int k = 0; k <= mMaxOrder; ++k) {
    // f_k: u_z = P_k, u_phi = -(z/r) c P'_k, u_r = (z/r) c^2 P''_k
    fill(fIdx(k),
         zOverR * cPhi * cPhi * pu2[k],
         -zOverR * cPhi * pu1[k],
         pu[k]);
    // g_k: u_phi = P_k, u_r = -c P'_k
    fill(gIdx(k), -cPhi * pu1[k], pu[k], 0.);
  }

  if (hasExtensional()) {
    const auto pv = o2::its3::align::legendrePols(mExtOrderZ, v);
    for (int k = 0; k <= mExtOrderPhi; ++k) {
      for (int l = 1; l <= mExtOrderZ; ++l) {
        // h_{k,l}: strictly radial u_r = P_k(u) P_l(v)
        fill(hIdx(k, l), pu[k] * pv[l], 0., 0.);
      }
    }
  }
}
