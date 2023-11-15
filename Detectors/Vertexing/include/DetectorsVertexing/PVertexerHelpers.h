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

/// \file PVertexerHelpers.h
/// \brief Primary vertex finder helper classes
/// \author ruben.shahoyan@cern.ch

#ifndef O2_PVERTEXER_HELPERS_H
#define O2_PVERTEXER_HELPERS_H

#include "gsl/span"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/FlatHisto2D.h"
#include "SimulationDataFormat/MCEventLabel.h"

namespace o2
{
namespace vertexing
{

using PVertex = o2::dataformats::PrimaryVertex;
using TimeEst = o2::dataformats::TimeStampWithError<float, float>;
using V2TRef = o2::dataformats::VtxTrackRef;
using GIndex = o2::dataformats::VtxTrackIndex;
using GTrackID = o2::dataformats::GlobalTrackID;

struct VertexingInput {
  gsl::span<int> idRange;
  TimeEst timeEst{0, -1.}; // negative error means don't use time info
  float scaleSigma2 = 10;
};

///< weights and scaling params for current vertex
struct VertexSeed : public PVertex {
  double wghSum = 0.;                                                                              // sum of tracks weights
  double wghChi2 = 0.;                                                                             // sum of tracks weighted chi2's
  double tMeanAcc = 0.;                                                                            // sum of track times * inv.err^2 wotj real time error
  double tMeanAccErr = 0.;                                                                         // some of tracks times inv.err^2 with real time error
  double tMeanAccTB = 0.;                                                                          // sum of track times * inv.err^2 with time error from time bracket, i.e. ITS
  double tMeanAccErrTB = 0.;                                                                       // sum of tracks times inv.err^2 with time error from time bracket, i.e. ITS
  double wghSumTB = 0.;                                                                            // sum of weights for tracks with time error from time bracket, i.e. ITS
  double cxx = 0., cyy = 0., czz = 0., cxy = 0., cxz = 0., cyz = 0., cx0 = 0., cy0 = 0., cz0 = 0.; // elements of lin.equation matrix
  float scaleSigma2 = 1.;                                                                          // scaling parameter on top of Tukey param
  float scaleSigma2Prev = 1.;
  float maxScaleSigma2Tested = 0.;
  float scaleSig2ITuk2I = 0; // inverse squared Tukey parameter scaled by scaleSigma2
  int nContributorsTB = 0;   // number of contributors with time error coming from Time Bracker
  int nScaleSlowConvergence = 0;
  int nScaleIncrease = 0;
  int nIterations = 0;

  void setScale(float scale2, float tukey2I)
  {
    scaleSigma2Prev = scaleSigma2;
    scaleSigma2 = scale2;
    scaleSig2ITuk2I = tukey2I / scale2;
  }

  void resetForNewIteration()
  {
    setNContributors(0);
    nContributorsTB = 0;
    //setTimeStamp({0., 0.});
    wghSum = 0.;
    wghChi2 = 0.;
    wghSumTB = 0.;
    tMeanAcc = 0.;
    tMeanAccErr = 0.;
    tMeanAccTB = 0.;
    tMeanAccErrTB = 0.;
    cxx = cyy = czz = cxy = cxz = cyz = cx0 = cy0 = cz0 = 0.;
  }

  VertexSeed() = default;
  VertexSeed(const PVertex& vtx)
    : PVertex(vtx) {}

  void print() const;
};

/// generic track with timestamp
struct TrackWithTimeStamp : o2::track::TrackParCov {
  TimeEst timeEst{};
  auto getTimeMUS() const { return timeEst; }
};

struct TrackVF {
  /** Straight track parameterization in the frame defined by alpha angle.
      Assumed to be defined in the proximity to vertex, so that the
      straight-line extrapolation Y=mY+mTgP*(x-mX) and Z=mZ+mTgL*(x-mX) is
      precise
  */
  enum { kUsed,
         kNoVtx = -1,
         kDiscarded = kNoVtx - 1 };
  enum { kITSTPCAdjust = 0x1,
         kDummyHBin = 0xffff };
  float x;      ///< reference X
  float y;      ///< Y at X
  float z;      ///< Z at X
  float sig2YI = 0.f; ///< YY component of inverse cov.matrix
  float sig2ZI = 0.f; ///< ZZ component of inverse cov.matrix
  float sigYZI = 0.f; ///< YZ component of inverse cov.matrix
  float tgP;    ///< tangent(phi) in tracking frame
  float tgL;    ///< tangent(lambda)
  float cosAlp; ///< cos of alpha frame
  float sinAlp; ///< sin of alpha frame

  TimeEst timeEst;
  float wgh = 0.; ///< track weight wrt current vertex seed
  float wghHisto = 0.; // weight based on track errors, used for histogramming
  int entry;           ///< track entry in the input vector
  int vtxID = kNoVtx; ///< assigned vertex
  GTrackID gid{};
  uint16_t bin = kDummyHBin; // seeds histo bin
  uint16_t flags = 0;
  //
  void setITSTPCAdjusted() { flags &= kITSTPCAdjust; }
  bool isITSTPCAdjusted() const { return flags & kITSTPCAdjust; }
  bool canAssign() const { return wgh > 0. && vtxID == kNoVtx; }
  bool canUse() const { return vtxID == kNoVtx; }
  bool canUse(float zmin, float zmax) const
  {
    return canUse() && (z > zmin && z < zmax);
  }
  bool operator<(const TrackVF& trc) const { return z < trc.z; }

  float getZForXY(float vx, float vy) const
  {
    return z + tgL * (vx * cosAlp + vy * sinAlp - x);
  }

  // weighted distance^2 to other track (accounting for own errors only)
  float getDist2(const TrackVF& o) const
  {
    auto dt = timeEst.getTimeStamp() - o.timeEst.getTimeStamp();
    auto dte2 = timeEst.getTimeStampError() * timeEst.getTimeStampError() + o.timeEst.getTimeStampError() * o.timeEst.getTimeStampError();
    auto dtnorm2 = dt * dt / dte2;
    auto dz = z - o.z;
    return dtnorm2 + dz * dz * sig2ZI;
  }

  // weighted distance^2 to other track (accounting for both track errors errors only)
  float getDist2Sym(const TrackVF& o) const
  {
    auto dt = timeEst.getTimeStamp() - o.timeEst.getTimeStamp();
    auto dz = z - o.z;
    float dte2 = o.timeEst.getTimeStampError() * o.timeEst.getTimeStampError() + timeEst.getTimeStampError() * timeEst.getTimeStampError();
    return dt / dte2 + dz * dz / (1. / sig2ZI + 1. / o.sig2ZI);
  }

  float getResiduals(const PVertex& vtx, float& dy, float& dz) const
  {
    // get residuals (Y and Z DCA in track frame) and calculate chi2
    float dx = vtx.getX() * cosAlp + vtx.getY() * sinAlp - x; // VX rotated to track frame - trackX
    dy = y + tgP * dx - (-vtx.getX() * sinAlp + vtx.getY() * cosAlp);
    dz = z + tgL * dx - vtx.getZ();
    return (dy * dy * sig2YI + dz * dz * sig2ZI) + 2. * dy * dz * sigYZI;
  }

  float getResiduals(const PVertex& vtx) const
  {
    // get residuals (Y and Z DCA in track frame) and calculate chi2
    float dx = vtx.getX() * cosAlp + vtx.getY() * sinAlp - x; // VX rotated to track frame - trackX
    auto dy = y + tgP * dx - (-vtx.getX() * sinAlp + vtx.getY() * cosAlp);
    auto dz = z + tgL * dx - vtx.getZ();
    return (dy * dy * sig2YI + dz * dz * sig2ZI) + 2. * dy * dz * sigYZI;
  }

  float evalChi2ToVertex(const PVertex& vtx, bool useTime)
  {
    constexpr float NDOF2I = 1. / 2, NDOF3I = 1. / 3;
    float chi2T = getResiduals(vtx); // track-vertex residuals and chi2
    if (useTime) {
      float dt = timeEst.getTimeStamp() - vtx.getTimeStamp().getTimeStamp();
      chi2T += dt * dt / (timeEst.getTimeStampError() * timeEst.getTimeStampError());
      chi2T *= NDOF3I;
    } else {
      chi2T *= NDOF2I;
    }
    return chi2T;
  }

  TrackVF() = default;
  TrackVF(const o2::track::TrackParCov& src, const TimeEst& t_est, int _entry, GTrackID _gid, float addHTErr2 = 0., float addHZErr2 = 0.)
    : x(src.getX()), y(src.getY()), z(src.getZ()), tgL(src.getTgl()), tgP(src.getSnp() / std::sqrt(1. - src.getSnp()) * (1. + src.getSnp())), timeEst(t_est), entry(_entry), gid(_gid)
  {
    o2::math_utils::sincos(src.getAlpha(), sinAlp, cosAlp);
    double syy = src.getSigmaY2(), szz = src.getSigmaZ2(), syz = src.getSigmaZY();
    auto det = syy * szz - syz * syz;
    if (det <= 1e-20) {
      wghHisto = -1;
      reportBadTrack(src, t_est, gid);
      return;
    }
    auto detI = 1. / det;
    sig2YI = szz * detI;
    sig2ZI = syy * detI;
    sigYZI = -syz * detI;
    wghHisto = 1. / ((szz + addHZErr2) * (t_est.getTimeStampError() * t_est.getTimeStampError() + addHTErr2));
  }

  void reportBadTrack(const o2::track::TrackParCov& src, const TimeEst& t_est, GTrackID _gid);

  ClassDefNV(TrackVF, 1);
};

struct SeedHistoTZ : public o2::dataformats::FlatHisto2D_f {
  using o2::dataformats::FlatHisto2D<float>::FlatHisto2D;

  uint16_t fillAndFlagBin(float x, float y, float w)
  {
    uint32_t bin = getBin(x, y);
    if (isValidBin(bin)) {
      if (isBinEmpty(bin)) {
        filledBins.push_back(bin);
      }
      fillBin(bin, w);
      nEntries++;
      return uint16_t(bin);
    }
    return 0xffff;
  }

  void clear()
  {
    o2::dataformats::FlatHisto2D<float>::clear();
    filledBins.clear();
  }

  int findPeakBin();

  std::vector<int> filledBins;
  int nEntries{};
};

struct TimeZCluster {
  std::vector<int> trackIDs{};
  TimeEst timeEst{};
};

// structure to produce debug dump for neighbouring vertices comparison
struct PVtxCompDump {
  PVertex vtx0{};
  PVertex vtx1{};
  float chi2z{0};
  float chi2t{0};
  bool rej = false;
  PVtxCompDump() = default;
  ClassDefNV(PVtxCompDump, 1);
};

// structure to produce debug dump for DBSCAN clusters
struct TrackVFDump {
  float z = 0;
  float ze2i = 0.;
  float t = 0;
  float te = 0;
  float wh = 0.;
  TrackVFDump() = default;
  ClassDefNV(TrackVFDump, 1);
};

struct InteractionCandidate : public o2::InteractionRecord {
  float time = 0;
  float amplitude = 0;
  uint32_t flag = 0; // origin, etc.
};

} // namespace vertexing
} // namespace o2

#endif
