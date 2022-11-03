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

/// @file   AlignmentTrack.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Track model for the alignment

/**
  * Track model for the alignment: trackParam_t for kinematics
  * proper with number of multiple scattering kinks.
  * Full support for derivatives and residuals calculation
  */

#ifndef ALIGNMENTTRACK_H
#define ALIGNMENTTRACK_H

#include "Align/AlignmentPoint.h"
#include "ReconstructionDataFormats/Track.h"
#include <TObjArray.h>
#include <TArrayI.h>
#include "DetectorsBase/Propagator.h"

namespace o2
{
namespace align
{

class AlignmentTrack : public trackParam_t, public TObject
{
 public:
  using trackParam_t = o2::track::TrackParametrizationWithError<double>;
  using PropagatorD = o2::base::PropagatorD;
  using MatCorrType = PropagatorD::MatCorrType;
  using trackParam_t::setParam;
  static constexpr double MaxDefStep = 3.0;
  static constexpr double MaxDefSnp = 0.95;

  enum { kCosmicBit = BIT(14),
         kFieldONBit = BIT(15),
         kResidDoneBit = BIT(16),
         kDerivDoneBit = BIT(17),
         kKalmanDoneBit = BIT(18) };
  enum { kNKinParBOFF = 4, // N params for ExternalTrackParam part w/o field
         kNKinParBON = 5,  // N params for ExternalTrackParam part with field
         kParY = 0,        // track parameters
         kParZ,
         kParSnp,
         kParTgl,
         kParQ2Pt
  };
  AlignmentTrack() = default;
  ~AlignmentTrack() override = default;
  void defineDOFs();
  int getNPoints() const { return mPoints.size(); }
  int getInnerPointID() const { return mInnerPointID; }
  const AlignmentPoint* getInnerPoint() const { return getPoint(mInnerPointID); }
  AlignmentPoint* getPoint(int i) { return (AlignmentPoint*)mPoints[i]; }
  const AlignmentPoint* getPoint(int i) const { return mPoints[i]; }
  auto getPoints() { return mPoints; }
  auto& addDetectorPoint() { return mDetPoints.emplace_back(); }
  void suppressLastPoints(int n);
  void setRefPoint(AlignmentPoint* p) { mPoints.emplace_back(p); }
  int getNLocPar() const { return mNLocPar; }
  int getNLocExtPar() const { return mNLocExtPar; }
  //
  void Clear(Option_t* opt = "") final;
  void Print(Option_t* opt = "") const final;
  virtual void dumpCoordinates() const;
  //
  template <typename P>
  void copyFrom(const o2::track::TrackParametrizationWithError<P>& trc);
  bool propagateToPoint(trackParam_t& tr, const AlignmentPoint* pnt, double maxStep, double maxSnp = 0.95, MatCorrType mt = MatCorrType::USEMatCorrLUT, track::TrackLTIntegral* tLT = nullptr, int signCorr = 0);
  bool propagateParamToPoint(trackParam_t& tr, const AlignmentPoint* pnt, double maxStep = 3, double maxSnp = 0.95, MatCorrType mt = MatCorrType::USEMatCorrLUT, int signCorr = 0);             // param only
  bool propagateParamToPoint(trackParam_t* trSet, int nTr, const AlignmentPoint* pnt, double maxStep = 3, double maxSnp = 0.95, MatCorrType mt = MatCorrType::USEMatCorrLUT, int signCorr = 0); // params only
  //
  bool calcResiduals(const double* params = nullptr);
  bool calcResidDeriv(double* params = nullptr);
  bool calcResidDerivGlo(AlignmentPoint* pnt);
  bool testLocalSolution();
  //
  bool isCosmic() const { return TestBit(kCosmicBit); }
  void setCosmic(bool v = true) { SetBit(kCosmicBit, v); }
  bool getFieldON() const { return TestBit(kFieldONBit); }
  void setFieldON(bool v = true) { SetBit(kFieldONBit, v); }
  bool getResidDone() const { return TestBit(kResidDoneBit); }
  void setResidDone(bool v = true) { SetBit(kResidDoneBit, v); }
  bool getDerivDone() const { return TestBit(kDerivDoneBit); }
  void setDerivDone(bool v = true) { SetBit(kDerivDoneBit, v); }
  bool getKalmanDone() const { return TestBit(kKalmanDoneBit); }
  void setKalmanDone(bool v = true) { SetBit(kKalmanDoneBit, v); }
  //
  void sortPoints();
  bool iniFit();
  bool residKalman();
  bool processMaterials();
  bool combineTracks(trackParam_t& trcL, const trackParam_t& trcU);
  //
  void setChi2(double c) { mChi2 = c; };
  double getChi2() const { return mChi2; }
  void setChi2Ini(double c) { mChi2Ini = c; };
  double getChi2Ini() const { return mChi2Ini; }
  double getChi2CosmUp() const { return mChi2CosmUp; }
  double getChi2CosmDn() const { return mChi2CosmDn; }
  //
  void imposePtBOff(double pt) { setQ2Pt(1. / pt); }
  // propagation methods
  void copyFrom(const trackParam_t* etp);
  bool applyMatCorr(trackParam_t& trPar, const double* corrDiag, const AlignmentPoint* pnt);
  bool applyMatCorr(trackParam_t* trSet, int ntr, const double* corrDiaf, const AlignmentPoint* pnt);
  bool applyMatCorr(trackParam_t& trPar, const double* corrpar);
  //
  double getResidual(int dim, int pntID) const { return mResid[dim][pntID]; }
  const double* getDResDLoc(int dim, int pntID) const { return mDResDLoc[dim].data() + (pntID * mNLocPar); }
  const double* getDResDGlo(int dim, int id) const { return mDResDGlo[dim].data() + id; }
  const int* getGloParID() const { return mGloParID.data(); }
  //
  void setParams(trackParam_t& tr, double x, double alp, const double* par, bool add);
  void setParams(trackParam_t* trSet, int ntr, double x, double alp, const double* par, bool add);
  void setParam(trackParam_t& tr, int par, double val);
  void setParam(trackParam_t* trSet, int ntr, int par, double val);
  void modParam(trackParam_t& tr, int par, double delta);
  void modParam(trackParam_t* trSet, int ntr, int par, double delta);
  //
  void richardsonDeriv(const trackParam_t* trSet, const double* delta,
                       const AlignmentPoint* pnt, double& derY, double& derZ);
  //
  const double* getLocPars() const { return mLocPar.data(); }
  std::vector<double>& getLocParsV() { return mLocPar; }
  void setLocPars(const double* pars);
  void addLocPars(const double* pars);
  //
 protected:
  //
  bool calcResidDeriv(double* params, bool invert, int pFrom, int pTo);
  bool calcResiduals(const double* params, bool invert, int pFrom, int pTo);
  bool fitLeg(trackParam_t& trc, int pFrom, int pTo, bool& inv);
  bool processMaterials(trackParam_t& trc, int pFrom, int pTo);
  //
  void checkExpandDerGloBuffer(unsigned int minSize);
  //
  static double richardsonExtrap(double* val, int ord = 1);
  static double richardsonExtrap(const double* val, int ord = 1);
  //
  // ---------- dummies ----------
  AlignmentTrack(const AlignmentTrack&);
  AlignmentTrack& operator=(const AlignmentTrack&);
  //
 protected:
  int mNLocPar = 0;                         // number of local params
  int mNLocExtPar = 0;                      // number of local params for the external track param
  int mNGloPar = 0;                         // number of free global parameters the track depends on
  int mNDF = 0;                             // number of degrees of freedom
  int mInnerPointID = 0;                    // ID of inner point in sorted track. For 2-leg cosmics - innermost point of lower leg
  bool mNeedInv[2] = {};                    // set if one of cosmic legs need inversion
  double mChi2 = 0;                         // chi2 with current residuals
  double mChi2CosmUp = 0;                   // chi2 for cosmic upper leg
  double mChi2CosmDn = 0;                   // chi2 for cosmic down leg
  double mChi2Ini = 0;                      // chi2 with current residuals
  std::vector<AlignmentPoint*> mPoints{};   // alignment points pointers sorted in X
  std::vector<AlignmentPoint> mDetPoints{}; // alignment points added by detectors
  std::vector<double> mResid[2];            // residuals array
  std::vector<double> mDResDLoc[2];         // array for derivatives over local params
  std::vector<double> mDResDGlo[2];         // array for derivatives over global params
  std::vector<double> mLocPar;              // local parameters array
  std::vector<int> mGloParID;               // IDs of relevant global params
 private:
  bool propagate(trackParam_t& tr, const AlignmentPoint* pnt, double maxStep, double maxSnp, MatCorrType mt, track::TrackLTIntegral* tLT, int signCorr = 0);
  //
  ClassDefOverride(AlignmentTrack, 2)
};

//____________________________________________________________________________________________
inline void AlignmentTrack::setParams(trackParam_t& tr, double x, double alp, const double* par, bool add)
{
  // set track params
  const double kDefQ2PtCosm = 1;
  const double kDefG2PtColl = 1. / 0.6;
  params_t tmp;
  std::copy(par, par + kNKinParBON, std::begin(tmp));
  tr.set(x, alp, tmp);
  if (add) { // par is correction to reference params
    for (size_t i = 0; i < kNKinParBON; ++i) {
      const double val = tr.getParam(i) + this->getParam(i);
      tr.setParam(val, i);
    }
  }
  if (!getFieldON()) {
    tr.setQ2Pt(isCosmic() ? kDefQ2PtCosm : kDefG2PtColl); // only 4 params are valid
  }
}

//____________________________________________________________________________________________
inline void AlignmentTrack::setParams(trackParam_t* trSet, int ntr, double x, double alp, const double* par, bool add)
{
  // set parames for multiple tracks (VECTORIZE THIS)
  if (!add) { // full parameter supplied
    for (int itr = ntr; itr--;) {
      setParams(trSet[itr], x, alp, par, false);
    }
    return;
  }
  params_t partr{0}; // par is a correction to reference parameter
  for (int i = mNLocExtPar; i--;) {
    partr[i] = getParam(i) + par[i];
  }
  for (int itr = ntr; itr--;) {
    setParams(trSet[itr], x, alp, partr.data(), false);
  }
}

//____________________________________________________________________________________________
inline void AlignmentTrack::setParam(trackParam_t& tr, int par, double val)
{
  // set track parameter
  tr.setParam(val, par);
}

//____________________________________________________________________________________________
inline void AlignmentTrack::setParam(trackParam_t* trSet, int ntr, int par, double val)
{
  // set parames for multiple tracks (VECTORIZE THIS)
  for (int i = 0; i < ntr; ++i) {
    setParam(trSet[i], par, val);
  }
}

//____________________________________________________________________________________________
inline void AlignmentTrack::modParam(trackParam_t& tr, int par, double delta)
{
  // modify track parameter
  const auto val = tr.getParam(par) + delta;
  setParam(tr, par, val);
}

//____________________________________________________________________________________________
inline void AlignmentTrack::modParam(trackParam_t* trSet, int ntr, int par, double delta)
{
  // modify track parameter (VECTORIZE THOS)
  for (int i = 0; i < ntr; ++i) {
    modParam(trSet[i], par, delta);
  }
}

//______________________________________________
inline void AlignmentTrack::copyFrom(const trackParam_t* etp)
{
  // assign kinematics
  set(etp->getX(), etp->getAlpha(), etp->getParams(), etp->getCov().data());
}

//______________________________________________
template <typename P>
inline void AlignmentTrack::copyFrom(const o2::track::TrackParametrizationWithError<P>& trc)
{
  setX(trc.getX());
  setAlpha(trc.getAlpha());
  setAbsCharge(trc.getAbsCharge());
  setPID(trc.getPID());
  for (int i = o2::track::kNParams; i--;) {
    setParam(double(trc.getParam(i)), i);
  }
  for (int i = o2::track::kCovMatSize; i--;) {
    setCov(double(trc.getCov()[i]), i);
  }
}

} // namespace align
} // namespace o2
#endif
