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

/// \file   NDRegression.h
/// \author Gabor Biro, biro.gabor@wigner.hu

#ifndef ALICEO2_NDREGRESSION_H
#define ALICEO2_NDREGRESSION_H

#include "TNamed.h"
#include "TMath.h"
#include "TFormula.h"
#include "TObjString.h"
#include "TString.h"
#include "TLinearFitter.h"
#include "TMinuit.h"
#include "TVectorD.h"
#include "THn.h"
#include "TH1.h"
#include "Rtypes.h"
#include <memory>
#include "CommonUtils/TreeStream.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TCanvas.h"

class TreeStreamRedirector;
class THn;

namespace o2
{

using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
using utils::TreeStreamRedirector;

namespace nd_regression
{

class NDRegression : public TNamed
{
 public:
  NDRegression() = default;
  NDRegression(const char* name, const char* title);
  ~NDRegression() = default;
  // NDRegression();

  bool init();
  void SetCuts(Double_t nSigma = 6, Double_t robustFraction = 0.95, Int_t estimator = 1);

  void SetStreamer(shared_ptr<TreeStreamRedirector>& streamer) { fStreamer = streamer; }
  void EvaluateUni(const Int_t nvectors, const Double_t* data, Double_t& mean, Double_t& sigma, const Int_t hSub);
  Bool_t SetHistogram(THn* histo);

  Bool_t MakeFit(TTree* tree, const char* formulaVal, const char* formulaVar, const char* selection, const char* formulaKernel, const char* dimensionFormula, Double_t weightCut = 1e-5, Int_t entries = 1e9, Bool_t useBinNorm = kTRUE);
  Bool_t MakeRobustStatistic(TVectorD& values, TVectorD& errors, TObjArray& pointArray, TObjArray& kernelArrayI2, Int_t& nvarFormula, Double_t weightCut, Double_t robustFraction);

  // Bool_t MakeFit(TTree * tree , const char *formulaVal, const char * formulaVar, const char*selection, const char * formulaKernel,  const char * dimensionFormula, Double_t weightCut=0.00001, Int_t entries=1000000000, Bool_t useBinNorm=kTRUE);

 protected:
  shared_ptr<TreeStreamRedirector> fStreamer; // ! streamer to keep - test intermediate data
  THn* fHistPoints;                           //  histogram local point distoribution

  TObjArray* fLocalFitParam;   // local fit parameters + RMS + chi2
  TObjArray* fLocalFitQuality; // local fit npoints chi2
  TObjArray* fLocalFitCovar;   // local fit covariance matrix

  TMatrixD* fLocalRobustStat; // local robust statistic

  Int_t* fBinIndex;     //[fNParameters] working arrays current bin index
  Double_t* fBinCenter; //[fNParameters] working current local variables - bin center
  Double_t* fBinDelta;  //[fNParameters] working current local variables - bin delta
  Double_t* fBinWidth;  //[fNParameters] working current local variables - bin delta

  Int_t fCutType; //  type of the cut 0- no cut 1-cut localmean=median,2-cut localmen=rosbut mean

  Double_t fRobustFractionLTS; //  fraction of data used for the robust mean and robust rms estimator (LTS https://en.wikipedia.org/wiki/Least_trimmed_squares)
  Double_t fRobustRMSLTSCut;   //  cut on the robust RMS  |value-localmean|<fRobustRMSLTSCut*localRMS

 private:
  NDRegression& operator=(const NDRegression&);
  NDRegression(const NDRegression&);
  ClassDef(o2::nd_regression::NDRegression, 1);
};

} // namespace nd_regression
} // namespace o2

#endif
