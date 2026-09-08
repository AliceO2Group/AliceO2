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
///
/// \file MFTFwdTrackHelpers.h
/// \brief Forward-track coordinate helpers for MFT CA candidate finding
///

#ifndef ALICEO2_ITSMFT_TRACKING_MFTFWDTRACKHELPERS_H_
#define ALICEO2_ITSMFT_TRACKING_MFTFWDTRACKHELPERS_H_

#include <array>
#include <cmath>

#include "CommonConstants/MathConstants.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/Constants.h"
#include "MFTTracking/Constants.h"
#include "ReconstructionDataFormats/TrackFwd.h"

namespace o2::itsmft::tracking::detail
{

/// MFT CA uses o2::mft::constants::mft::LayersNumber half-disk layers (same index as GeometryTGeo::getLayer).
/// Physical disk index is halfLayer / 2; ROFOverlapTable stores one LayerTiming per half-layer.

inline float mftLayerZ(int layer)
{
  return o2::mft::constants::mft::LayerZCoordinate()[layer];
}

inline void mftTrackletProject(float xCl, float yCl, float zCl, float pvX, float pvY, float pvZ,
                               float zFrom, float zTo, float bz, float minPt,
                               float& xProj, float& yProj)
{
  if (std::abs(bz) > 0.01f && minPt > 0.f) {
    const float dxTan = xCl - pvX;
    const float dyTan = yCl - pvY;
    const float dzTan = zCl - pvZ;
    const float drTan = std::sqrt(dxTan * dxTan + dyTan * dyTan);
    float invQPt = 1.f / minPt;
    float tanl = (drTan > 1e-6f) ? -std::abs(dzTan) / drTan : -1.f;
    float phi = (drTan > 1e-6f) ? std::atan2(dyTan, dxTan) : 0.f;
    if (std::abs(tanl) > 1e-6f) {
      const float k = std::abs(o2::constants::math::B2C * bz);
      const float hz = (bz > 0.f) ? 1.f : -1.f;
      phi -= 0.5f * hz * invQPt * dzTan * k / tanl;
    }
    ROOT::Math::SVector<double, 5> params{xCl, yCl, phi, tanl, invQPt};
    ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>> cov{};
    cov(0, 0) = cov(1, 1) = cov(2, 2) = cov(3, 3) = 1.;
    const double qptSigma = std::clamp(static_cast<double>(std::abs(invQPt)), 1., 10.);
    cov(4, 4) = qptSigma * qptSigma;
    o2::track::TrackParCovFwd track{zCl, params, cov, 0.};
    track.propagateToZhelix(zTo, bz);
    xProj = static_cast<float>(track.getX());
    yProj = static_cast<float>(track.getY());
  } else {
    const float dz0 = zFrom - pvZ;
    if (std::abs(dz0) < 1e-6f) {
      xProj = xCl;
      yProj = yCl;
      return;
    }
    const float w = (zTo - pvZ) / dz0;
    xProj = pvX + w * (xCl - pvX);
    yProj = pvY + w * (yCl - pvY);
  }
}

inline void mftTrackletProject(float xCl, float yCl, float zCl, float pvX, float pvY, float pvZ,
                               int fromLayer, int toLayer, float bz, float minPt,
                               float& xProj, float& yProj)
{
  mftTrackletProject(xCl, yCl, zCl, pvX, pvY, pvZ, mftLayerZ(fromLayer), mftLayerZ(toLayer),
                     bz, minPt, xProj, yProj);
}

inline void mftTrackletSigmaXY(float x0, float y0, float pvX, float pvY, float pvZ,
                               float sigma2X0, float sigma2Y0, float sigma2PvX, float sigma2PvY, float sigma2PvZ,
                               float zFrom, float zTo, float rLayerFrom, float meanDeltaZ, float msAngle,
                               float bendingAngle, float xProj, float yProj, float& sigmaX, float& sigmaY)
{
  const float dz0 = zFrom - pvZ;
  const float tanlRef = (std::abs(rLayerFrom) > 1e-6f) ? zFrom / rLayerFrom : 0.f;
  const float sigma2MS = meanDeltaZ * meanDeltaZ * msAngle * msAngle * (tanlRef * tanlRef + 1.f);
  if (std::abs(dz0) < o2::its::constants::Tolerance) {
    sigmaX = std::sqrt(sigma2X0 + sigma2PvX + sigma2MS);
    sigmaY = std::sqrt(sigma2Y0 + sigma2PvY + sigma2MS);
  } else {
    const float w = (zTo - pvZ) / dz0;
    const float invDz0 = w / dz0;
    const float sigma2W = invDz0 * invDz0 * sigma2PvZ;
    const float dx0 = x0 - pvX;
    const float dy0 = y0 - pvY;
    const float oneMinusW = 1.f - w;
    sigmaX = std::sqrt(oneMinusW * oneMinusW * sigma2PvX + w * w * sigma2X0 + dx0 * dx0 * sigma2W + sigma2MS);
    sigmaY = std::sqrt(oneMinusW * oneMinusW * sigma2PvY + w * w * sigma2Y0 + dy0 * dy0 * sigma2W + sigma2MS);
  }
  const float rProj = std::hypot(xProj, yProj);
  if (rProj > 1e-6f && bendingAngle > 0.f) {
    const float dr = rProj * bendingAngle;
    const float invR = 1.f / rProj;
    const float sinPhi = yProj * invR;
    const float cosPhi = xProj * invR;
    sigmaX = std::sqrt(sigmaX * sigmaX + dr * dr * sinPhi * sinPhi);
    sigmaY = std::sqrt(sigmaY * sigmaY + dr * dr * cosPhi * cosPhi);
  }
}

inline void mftTrackletSigmaXY(float x0, float y0, float pvX, float pvY, float pvZ,
                               float sigma2X0, float sigma2Y0, float sigma2PvX, float sigma2PvY, float sigma2PvZ,
                               int fromLayer, int toLayer, float rLayerFrom, float meanDeltaZ, float msAngle,
                               float bendingAngle, float xProj, float yProj, float& sigmaX, float& sigmaY)
{
  mftTrackletSigmaXY(x0, y0, pvX, pvY, pvZ, sigma2X0, sigma2Y0, sigma2PvX, sigma2PvY, sigma2PvZ,
                     mftLayerZ(fromLayer), mftLayerZ(toLayer), rLayerFrom, meanDeltaZ, msAngle,
                     bendingAngle, xProj, yProj, sigmaX, sigmaY);
}

} // namespace o2::itsmft::tracking::detail

#endif /* ALICEO2_ITSMFT_TRACKING_MFTFWDTRACKHELPERS_H_ */
