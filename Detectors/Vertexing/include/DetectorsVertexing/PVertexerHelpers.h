// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace vertexing
{

using PVertex = o2::dataformats::PrimaryVertex;
using TimeEst = o2::dataformats::TimeStampWithError<float, float>;
using V2TRef = o2::dataformats::VtxTrackRef;
using GIndex = o2::dataformats::VtxTrackIndex;
using GTrackID = o2::dataformats::GlobalTrackID;

///< weights and scaling params for current vertex
struct VertexSeed : public PVertex {
  double wghSum = 0.;                                                                              // sum of tracks weights
  double wghChi2 = 0.;                                                                             // sum of tracks weighted chi2's
  double tMeanAcc = 0.;                                                                            // sum of track times * inv.err^2
  double tMeanAccErr = 0.;                                                                         // some of tracks times inv.err^2
  double cxx = 0., cyy = 0., czz = 0., cxy = 0., cxz = 0., cyz = 0., cx0 = 0., cy0 = 0., cz0 = 0.; // elements of lin.equation matrix
  float scaleSigma2 = 1.;                                                                          // scaling parameter on top of Tukey param
  float scaleSigma2Prev = 1.;
  float maxScaleSigma2Tested = 0.;
  float scaleSig2ITuk2I = 0; // inverse squared Tukey parameter scaled by scaleSigma2
  int nScaleSlowConvergence = 0;
  int nScaleIncrease = 0;
  int nIterations = 0;
  bool useConstraint = true;
  bool fillErrors = true;

  void setScale(float scale2, float tukey2I)
  {
    scaleSigma2Prev = scaleSigma2;
    scaleSigma2 = scale2;
    scaleSig2ITuk2I = tukey2I / scale2;
  }

  void resetForNewIteration()
  {
    setNContributors(0);
    //setTimeStamp({0., 0.});
    wghSum = 0;
    wghChi2 = 0;
    tMeanAcc = 0;
    tMeanAccErr = 0;
    cxx = cyy = czz = cxy = cxz = cyz = cx0 = cy0 = cz0 = 0.;
  }

  VertexSeed() = default;
  VertexSeed(const PVertex& vtx, bool _constraint, bool _errors)
    : PVertex(vtx), useConstraint(_constraint), fillErrors(_errors) {}

  void print() const;
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
  float x;      ///< reference X
  float y;      ///< Y at X
  float z;      ///< Z at X
  float sig2YI; ///< YY component of inverse cov.matrix
  float sig2ZI; ///< ZZ component of inverse cov.matrix
  float sigYZI; ///< YZ component of inverse cov.matrix
  float tgP;    ///< tangent(phi) in tracking frame
  float tgL;    ///< tangent(lambda)
  float cosAlp; ///< cos of alpha frame
  float sinAlp; ///< sin of alpha frame

  TimeEst timeEst;
  float wgh = 0.; ///< track weight wrt current vertex seed
  GTrackID entry;
  int16_t bin = -1; // seeds histo bin
  uint8_t flags = 0;
  int vtxID = kNoVtx; ///< assigned vertex

  //
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
    auto dtnorm2 = (timeEst.getTimeStamp() - o.timeEst.getTimeStamp()) / timeEst.getTimeStampError();
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

  TrackVF() = default;
  TrackVF(const o2::track::TrackParCov& src, const TimeEst& t_est, GTrackID _entry)
    : x(src.getX()), y(src.getY()), z(src.getZ()), tgL(src.getTgl()), tgP(src.getSnp() / std::sqrt(1. - src.getSnp()) * (1. + src.getSnp())), timeEst(t_est), entry(_entry)
  {
    o2::math_utils::sincos(src.getAlpha(), sinAlp, cosAlp);
    auto det = src.getSigmaY2() * src.getSigmaZ2() - src.getSigmaZY() * src.getSigmaZY();
    auto detI = 1. / det;
    sig2YI = src.getSigmaZ2() * detI;
    sig2ZI = src.getSigmaY2() * detI;
    sigYZI = -src.getSigmaZY() * detI;
  }
};

struct VertexingInput {
  gsl::span<int> idRange;
  TimeEst timeEst{0, -1.}; // negative error means don't use time info
  float scaleSigma2 = 10;
  bool useConstraint = false;
  bool fillErrors = true;
};

struct SeedHisto {
  float range = 20;
  float binSize = 0.5;
  float binSizeInv = 0.;
  int nFilled = 0;
  std::vector<int> data;

  SeedHisto() = delete;
  SeedHisto(float _range = 20., float _binsize = 0.5) : range(_range), binSize(_binsize)
  {
    auto zr = 2 * range;
    int nzb = zr / binSize;
    if (nzb * binSize < zr - 1e-9) {
      nzb++;
    }
    binSizeInv = 1. / binSize;
    range = nzb * binSize / 2.;
    data.resize(nzb);
  }

  int size() const { return data.size(); }

  void fill(float z)
  {
    incrementBin(findBin(z));
  }

  void incrementBin(int bin)
  {
    data[bin]++;
    nFilled++;
  }

  void decrementBin(int bin)
  {
    data[bin]--;
    nFilled--;
  }

  int findBin(float z)
  {
    auto d = z + range;
    if (d < 0.) {
      return 0;
    }
    uint32_t n = d * binSizeInv;
    return n < data.size() ? n : data.size() - 1;
  }

  int findHighestPeakBin() const
  {
    if (nFilled < 2) {
      return -1;
    }
    int n = data.size(), maxBin = -1, maxv = 0;
    for (int i = 0; i < n; i++) {
      if (data[i] > maxv) {
        maxv = data[(maxBin = i)];
      }
    }
    return maxBin;
  }

  bool isValidBin(int ib) const
  {
    return static_cast<uint32_t>(ib) < data.size();
  }

  float getBinCenter(int ib) const
  {
    return (ib + 0.5) * binSize - range; // no check for being in the range!!!
  }

  void discardBin(int ib)
  { // no check for being in the range!!!
    nFilled -= data[ib];
    data[ib] = 0;
  }
};

struct TimeZCluster {
  TimeEst timeEst;
  int first = -1;
  int last = -1;
  int count = 0;

  void clear()
  {
    first = last = -1;
    count = 0;
  }

  void addTrack(int i, const TimeEst& trcT)
  {
    auto trcTErr2 = trcT.getTimeStampError() * trcT.getTimeStampError();
    auto trcTErr2Inv = 1. / trcTErr2;
    if (first < 0) {
      first = last = i;
      timeEst.setTimeStamp(trcT.getTimeStamp());
      timeEst.setTimeStampError(trcT.getTimeStampError());
    } else {
      auto vtxTErr2Inv = 1. / (timeEst.getTimeStampError() * timeEst.getTimeStampError());
      auto vtxTErr2UpdInv = trcTErr2Inv + vtxTErr2Inv;
      auto vtxTErr2Upd = 1. / vtxTErr2UpdInv;
      timeEst.setTimeStamp((timeEst.getTimeStamp() * vtxTErr2Inv + trcT.getTimeStamp() * trcTErr2Inv) * vtxTErr2Upd);
      timeEst.setTimeStampError(std::sqrt(vtxTErr2Upd));
      if (i > last) {
        last = i;
      }
    }
    count++;
  }

  bool isCompatible(const TimeEst& c, float margin, float cut) const
  {
    if (first < 0) {
      return true;
    }
    float dt = timeEst.getTimeStamp() - c.getTimeStamp();
    if (c.getTimeStampError() && timeEst.getTimeStampError()) {
      float trcTErr2 = c.getTimeStampError() * c.getTimeStampError();
      float err = trcTErr2 + timeEst.getTimeStampError() + margin;
      return dt * dt / err < cut;
    } else {
      return std::abs(dt) < cut;
    }
  }

  void merge(TimeZCluster& c)
  {
    if (c.first < last) {
      first = c.first;
    } else {
      last = c.last;
    }
    if (timeEst.getTimeStampError() && c.timeEst.getTimeStampError()) { // weighted average
      auto cTErr2 = c.timeEst.getTimeStampError() * c.timeEst.getTimeStampError();
      auto cTErr2Inv = 1. / cTErr2;

      auto tErr2 = timeEst.getTimeStampError();
      auto tErr2Inv = 1. / (tErr2 * tErr2);
      auto tErr2UpdInv = cTErr2Inv + tErr2Inv;
      auto tErr2Upd = 1. / tErr2UpdInv;
      timeEst.setTimeStamp((timeEst.getTimeStamp() * tErr2Inv + c.timeEst.getTimeStamp() * cTErr2Inv) * tErr2Upd);
      timeEst.setTimeStampError(std::sqrt(tErr2Upd));
    } else {
      timeEst.setTimeStamp((timeEst.getTimeStamp() * count + c.timeEst.getTimeStamp() * c.count) / (count + c.count));
      count += c.count;
      c.count = 0;
    }
  }
};

} // namespace vertexing
} // namespace o2

#endif
