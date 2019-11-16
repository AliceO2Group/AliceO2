// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DCAFitter.h
/// \brief Defintions for DCA fitter class
/// \author ruben.shahoyan@cern.ch

#ifndef _ALICEO2_DCA_FITTER_
#define _ALICEO2_DCA_FITTER_

#include <TMath.h>
#include <Rtypes.h>

//#define _ADAPT_FOR_ALIROOT_  // to make it compatible with AliRoot AliExternalTrackParam

#ifdef _ADAPT_FOR_ALIROOT_
#include "AliExternalTrackParam.h"
typedef AliExternalTrackParam Track;
typedef float ftype_t;  // type precision for standard calculation (prefer float)
typedef double dtype_t; // type precision for calculation with risk of round-off errors
#define CONSTRDEF \
  {               \
  }
#else
#define CONSTRDEF = default

#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace base
{
#endif

class DCAFitter
{
 public:
#ifdef _ADAPT_FOR_ALIROOT_
#define getSigmaY2 GetSigmaY2
#define getSigmaZ2 GetSigmaZ2
#define getSigmaZY GetSigmaZY
#define getAlpha GetAlpha
#define getX GetX
#define getY GetY
#define getZ GetZ
#define getSnp GetSnp
#define getTgl GetTgl
#define getParam GetParameter
#define getCurvature GetC
#define propagateTo PropagateTo
#define propagateParamTo PropagateParamOnlyTo
  //
#else
  using Track = o2::track::TrackParCov;
  using ftype_t = float;  // type precision for standard calculation (prefer float)
  using dtype_t = double; // type precision for calculation with risk of round-off errors
#endif

  // ---> Auxiliary structs used by DCA finder

  //----------------------------------------------------
  ///< Inverse cov matrix of the point defined by the track
  struct TrackCovI {
    ftype_t sxx, syy, syz, szz;

    TrackCovI(const Track& trc) { set(trc); }
    TrackCovI() CONSTRDEF;
    void set(const Track& trc)
    {
      // we assign Y error to X for DCA calculation of 2 points
      // (otherwise for quazi-collinear tracks the X will not be constrained)
      ftype_t cyy = trc.getSigmaY2(), czz = trc.getSigmaZ2(), cyz = trc.getSigmaZY(), cxx = cyy;
      ftype_t detYZ = cyy * czz - cyz * cyz;
      if (detYZ > 0.) {
        sxx = 1. / cxx;
        syy = czz / detYZ;
        syz = -cyz / detYZ;
        szz = cyy / detYZ;
      } else {
        syy = 0.0; // failure
      }
    }
  };

  //----------------------------------------------------
  ///< Derivative (up to 2) of the TrackParam position over its running param X
  struct TrackDeriv2 {
    ftype_t dydx, dzdx, d2ydx2, d2zdx2;

    TrackDeriv2() CONSTRDEF;
    TrackDeriv2(const Track& trc, ftype_t bz) { set(trc, bz); }
    void set(const Track& trc, ftype_t bz)
    {
      ftype_t snp = trc.getSnp(), csp = TMath::Sqrt((1. - snp) * (1. + snp)), cspI = 1. / csp, crv2c = trc.getCurvature(bz) * cspI;
      dydx = snp * cspI;            // = snp/csp
      dzdx = trc.getTgl() * cspI;   // = tgl/csp
      d2ydx2 = crv2c * cspI * cspI; // = crv/csp^3
      d2zdx2 = crv2c * dzdx * dydx; // = crv*tgl*snp/csp^3
    }
  };

  //----------------------------------------------------
  //< precalculated track radius, center, alpha sin,cos and their combinations
  struct TrcAuxPar {
    dtype_t c, s;          // cos ans sin of track alpha
    dtype_t cc, cs, ss;    // products
    dtype_t r, xCen, yCen; // helix radius and center in lab

    TrcAuxPar() CONSTRDEF;
    TrcAuxPar(const Track& trc, ftype_t bz) { set(trc, bz); }

    void set(const Track& trc, ftype_t bz)
    {
      c = TMath::Cos(trc.getAlpha());
      s = TMath::Sin(trc.getAlpha());
      setRCen(trc, bz);
      cc = c * c;
      ss = s * s;
      cs = c * s;
    }

    void setRCen(const Track& tr, ftype_t bz);

    void glo2loc(ftype_t vX, ftype_t vY, ftype_t& vXL, ftype_t& vYL) const
    {
      // rotate XY in global frame to the frame of track with angle A
      vXL = vX * c + vY * s;
      vYL = -vX * s + vY * c;
    }

    void loc2glo(ftype_t vXL, ftype_t vYL, ftype_t& vX, ftype_t& vY) const
    {
      // rotate XY in local alpha frame to global frame
      vX = vXL * c - vYL * s;
      vY = vXL * s + vYL * c;
    }
    void glo2loc(dtype_t vX, dtype_t vY, dtype_t& vXL, dtype_t& vYL) const
    {
      // rotate XY in global frame to the frame of track with angle A
      vXL = vX * c + vY * s;
      vYL = -vX * s + vY * c;
    }

    void loc2glo(dtype_t vXL, dtype_t vYL, dtype_t& vX, dtype_t& vY) const
    {
      // rotate XY in local alpha frame to global frame
      vX = vXL * c - vYL * s;
      vY = vXL * s + vYL * c;
    }
  };

  //----------------------------------------------------
  //< coefficients of the track-point contribution to the PCA (Vx,Vy,Vz) to 2 points in lab frame represented via local points coordinates as
  //< Vx = mXX0*x0+mXY0*y0+mXZ0*z0 + mXX1*x1+mXY1*y1+mXZ1*z1
  //< Vy = mYX0*x0+mYY0*y0+mYZ0*z0 + mYX1*x1+mYY1*y1+mYZ1*z1
  //< Vz = mZX0*x0+mZY0*y0+mZZ0*z0 + mZX1*x1+mZY1*y1+mZZ1*z1
  //< where {x0,y0,z0} and {x1,y1,z1} are track positions in their local frames
  struct TrackCoefVtx {
    ftype_t mXX, mXY, mXZ, mYX, mYY, mYZ, mZX, mZY, mZZ;
    TrackCoefVtx() CONSTRDEF;
  };

  //----------------------------------------------------
  ///< particular point on track trajectory (in track proper alpha-frame)
  struct Triplet {
    ftype_t x, y, z;
    Triplet(Track& trc) { set(trc); }
    Triplet(ftype_t px = 0, ftype_t py = 0, ftype_t pz = 0) : x(px), y(py), z(pz) {}
    void set(const Track& trc)
    {
      x = trc.getX();
      y = trc.getY();
      z = trc.getZ();
    }
  };

  //----------------------------------------------------
  //< crossing coordinates of 2 circles
  struct CrossInfo {
    ftype_t xDCA[2];
    ftype_t yDCA[2];
    int nDCA;

    CrossInfo() CONSTRDEF;
    CrossInfo(const TrcAuxPar& trc0, const TrcAuxPar& trc1) { set(trc0, trc1); }
    void set(const TrcAuxPar& trc0, const TrcAuxPar& trc1);

    void notTouchingXY(ftype_t dist, ftype_t xDist, ftype_t yDist, const TrcAuxPar& trcA, ftype_t rBSign)
    {
      // fast method to calculate DCA between 2 circles, assuming that they don't touch each outer:
      // the parametric equation of lines connecting the centers is x = xA + t/dist * xDist, y = yA + t/dist * yDist
      // with xA,yY being the center of the circle A ( = trcA.xCen, trcA.yCen ), xDist = trcB.xCen = trcA.xCen ...
      // There are 2 special cases:
      // (a) small circle is inside the large one: provide rBSign as -trcB.r
      // (b) circle are side by side: provide rBSign as trcB.r
      nDCA = 1;
      auto t2d = (dist + trcA.r - rBSign) / dist;
      xDCA[0] = trcA.xCen + 0.5 * (xDist * t2d);
      yDCA[0] = trcA.yCen + 0.5 * (yDist * t2d);
    }
  };

  struct Derivatives {
    dtype_t dChidx0, dChidx1;                   // 1st derivatives of chi2 vs tracks local parameters X
    dtype_t dChidx0dx0, dChidx1dx1, dChidx0dx1; // 2nd derivatives of chi2 vs tracks local parameters X
  };

  // <--- Auxiliary structs used by DCA finder

  //===============================================================================

  DCAFitter() CONSTRDEF;

  int getMaxIter() const
  {
    return mMaxIter;
  }
  ftype_t getMaxR() const { return TMath::Sqrt(mMaxR2); }
  ftype_t getMaxChi2() const { return mMaxChi2; }
  ftype_t getMinParamChange() const { return mMinParamChange; }
  ftype_t getBz() const { return mBz; }
  bool getUseAbsDCA() const { return mUseAbsDCA; }

  void setMaxIter(int n = 20) { mMaxIter = n > 2 ? n : 2; }
  void setMaxR(ftype_t r = 200.) { mMaxR2 = r * r; }
  void setMaxChi2(ftype_t chi2 = 999.) { mMaxChi2 = chi2; }
  void setBz(ftype_t bz) { mBz = bz; }
  void setMinParamChange(ftype_t x = 1e-3) { mMinParamChange = x > 1e-4 ? x : 1.e-4; }
  void setMinRelChi2Change(ftype_t r = 0.9) { mMinRelChi2Change = r > 0.1 ? r : 999.; }
  void setUseAbsDCA(bool v) { mUseAbsDCA = v; }

  void clear()
  {
    mNCandidates = 0;
    mCrossIDCur = 0;
    mCrossIDAlt = -1;
  }

  DCAFitter(ftype_t bz, ftype_t minRelChiChange = 0.9, ftype_t minXChange = 1e-3, ftype_t maxChi = 999, int n = 20, ftype_t maxR = 200.)
  {
    setMaxIter(n);
    setMaxR(maxR);
    setMaxChi2(maxChi);
    setMinParamChange(minXChange);
    setMinRelChi2Change(minRelChiChange);
    setBz(bz);
    setUseAbsDCA(false); // by default use weighted DCA definition (much slower)
  }

  ///< number of validated V0 candidates (at most 2 are possible)
  int getNCandidates() const { return mNCandidates; }

  ///< return PCA candidate (no check for its validity)
  const Triplet& getPCACandidate(int cand) const { return mPCA[cand]; }

  ///< return Chi2 at PCA candidate (no check for its validity)
  ftype_t getChi2AtPCACandidate(int cand) const { return mChi2[cand]; }

  ///< 1st track params propagated to V0 candidate (no check for the candidate validity)
  const Track& getTrack0(int cand) const { return mCandTr0[cand]; }

  ///< 2nd track params propagated to V0 candidate (no check for the candidate validity)
  const Track& getTrack1(int cand) const { return mCandTr1[cand]; }

  ///< calculate parameters tracks at PCA
  int process(const Track& trc0, const Track& trc1)
  {
    // find dca of 2 tracks
    TrcAuxPar trc0Aux(trc0, mBz), trc1Aux(trc1, mBz);
    return process(trc0, trc0Aux, trc1, trc1Aux);
  }

  ///< calculate parameters tracks at PCA, using precalculated aux info // = TrcAuxPar(track) //
  int process(const Track& trc0, const TrcAuxPar& trc0Aux,
              const Track& trc1, const TrcAuxPar& trc1Aux);

  ///< minimizer for abs distance definition of DCA, starting with cached tracks
  bool processCandidateDCA(const TrcAuxPar& trc0Aux, const TrcAuxPar& trc1Aux);

  ///< minimizer for weighted distance definition of DCA (chi2), starting with cached tracks
  bool processCandidateChi2(const TrcAuxPar& trc0Aux, const TrcAuxPar& trc1Aux);

  ///< minimize w/o preliminary propagation to XY crossing points
  int processAsIs(const Track& trc0, const Track& trc1);

  ///< calculate squared distance between 2 tracks
  static ftype_t getDistance2(const Track& trc0, const Track& trc1);

  ///< calculate half sum of squared distances between 2 tracks and vertex
  static ftype_t getDistance2(ftype_t x, ftype_t y, ftype_t z, const Track& trc0, const Track& trc1);

 protected:
  void calcPCACoefs(const TrcAuxPar& trc0Aux, const TrackCovI& trcEI0,
                    const TrcAuxPar& trc1Aux, const TrackCovI& trcEI1,
                    TrackCoefVtx& trCFVT0, TrackCoefVtx& trCFVT1) const;

  ///< PCA with weighted DCA definition
  void calcPCA(const Track& trc0, const TrackCoefVtx& trCFVT0,
               const Track& trc1, const TrackCoefVtx& trCFVT1,
               Triplet& v) const;

  ///< PCA with abs DCA definition
  void calcPCA(const Track& trc0, const TrcAuxPar& trc0Aux,
               const Track& trc1, const TrcAuxPar& trc1Aux,
               Triplet& v) const;

  ///< chi2 (weighted distance)
  ftype_t calcChi2(const Triplet& pca,
                   const Track& trc0, const TrcAuxPar& trc0Aux, const TrackCovI& trcEI0,
                   const Track& trc1, const TrcAuxPar& trc1Aux, const TrackCovI& trcEI1) const;

  ///< DCA (abs distance)
  ftype_t calcDCA(const Track& tPnt0, const TrcAuxPar& trc0Aux, const Track& tPnt1, const TrcAuxPar& trc1Aux) const;
  ftype_t calcDCA(const Triplet& pca, const Track& trc0, const TrcAuxPar& trc0Aux, const Track& trc1, const TrcAuxPar& trc1Aux) const;

  Triplet calcResid(const Track& trc, const TrcAuxPar& alpCS, const Triplet& vtx) const
  {
    ftype_t vlX, vlY; // Vertex XY in track local frame
    alpCS.glo2loc(vtx.x, vtx.y, vlX, vlY);
    return Triplet(trc.getX() - vlX, trc.getY() - vlY, trc.getZ() - vtx.z);
  }

  void chi2Deriv(const Track& trc0, const TrackDeriv2& tDer0, const TrcAuxPar& trc0Aux, const TrackCovI& trcEI0, const TrackCoefVtx& trCFVT0,
                 const Track& trc1, const TrackDeriv2& tDer1, const TrcAuxPar& trc1Aux, const TrackCovI& trcEI1, const TrackCoefVtx& trCFVT1,
                 Derivatives& deriv) const;

  void DCADeriv(const Track& trc0, const TrackDeriv2& tDer0, const TrcAuxPar& trc0Aux,
                const Track& trc1, const TrackDeriv2& tDer1, const TrcAuxPar& trc1Aux,
                Derivatives& deriv) const;

  bool closerToAlternative(ftype_t x, ftype_t y) const;

 private:
  bool mUseAbsDCA;           // ignore track errors (minimize abs DCA to vertex)
  int mMaxIter;              // max iterations
  ftype_t mMaxR2;            // max radius to consider
  ftype_t mMaxChi2;          // max chi2 to accept
  ftype_t mMinRelChi2Change; // stop iterations if relative chi2 change is less than requested
  ftype_t mMinParamChange;   // stop iterations when both X params change by less than this value

  ftype_t mBz; // mag field for simple propagation

  // Working arrays
  CrossInfo mCrossings; //! analystical XY crossings (max 2) of the seeds
  int mCrossIDCur;      //! XY crossing being tested
  int mCrossIDAlt;      //! XY crossing alternative to the one being tested. Abandon fit if it converges to it

  int mNCandidates;                      //! number of consdered candidates
  Triplet mPCA[2];                       //! PCA for 2 possible cases
  ftype_t mChi2[2];                      //! Chi2 at PCA candidate
  TrackCovI mTrcEI0[2], mTrcEI1[2];      //! errors for each track candidate
  TrackCoefVtx mTrCFVT0[2], mTrCFVT1[2]; //! coefficients of PCA vs track points for each track
  Track mCandTr0[2], mCandTr1[2];        //! Tracks at PCA, max 2 candidates. Note: Errors are at seed XY point

  ClassDefNV(DCAFitter, 1);
};

#ifndef _ADAPT_FOR_ALIROOT_
}
}
#endif

#endif // _O2_DCA_FITTER_
