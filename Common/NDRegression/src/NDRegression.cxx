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

#include <iostream>
#include "NDRegression/NDRegression.h"

using namespace o2::nd_regression;

ClassImp(o2::nd_regression::NDRegression);

NDRegression::NDRegression(const char* name, const char* title) : TNamed(name, title) {}

Bool_t NDRegression::init()
{
  fHistPoints = NULL;
  fLocalRobustStat = NULL;

  fRobustRMSLTSCut = 0.0;
  fRobustFractionLTS = 0.0;
  fCutType = 0;

  std::cout << "Init done" << std::endl;
  return true;
}

void NDRegression::SetCuts(Double_t nSigma, Double_t robustFraction,
                           Int_t estimator)
{
  //
  //
  //
  fRobustFractionLTS =
    robustFraction;          //  fraction of data used for the robust mean and robust
                             //  rms estimator (LTS
                             //  https://en.wikipedia.org/wiki/Least_trimmed_squares)
  fRobustRMSLTSCut = nSigma; //  cut on the robust RMS
                             //  |value-localmean|<fRobustRMSLTSCut*localRMS
  fCutType = estimator;      //  type of the cut
                             //  0: no cut
                             //  1: cut localmean=median,
                             //  2: cut localmen=rosbut mean
                             //  3: use linearFitter.EvalRobust() instead of Eval
}

Bool_t NDRegression::SetHistogram(THn* histo)
{
  //
  // Setup the local regression ayout according THn hitogram binning
  //
  if (fHistPoints != 0) {
    ::Error("NDRegression::SetHistogram", "Histogram initialized\n");
    return false;
  }
  fHistPoints = histo;
  fLocalFitParam = new TObjArray(fHistPoints->GetNbins());
  fLocalFitParam->SetOwner(kTRUE);
  fLocalFitQuality = new TObjArray(fHistPoints->GetNbins());
  fLocalFitQuality->SetOwner(kTRUE);
  fLocalFitCovar = new TObjArray(fHistPoints->GetNbins());
  fLocalFitCovar->SetOwner(kTRUE);

  //
  // Check histogram
  //
  Int_t ndim = histo->GetNdimensions();
  for (Int_t idim = 0; idim < ndim; idim++) {
    TAxis* axis = histo->GetAxis(idim);
    if (axis->GetNbins() < 2) {
      ::Error("NDRegression::SetHistogram", "Invalid binning nbins<2 %d", axis->GetNbins());
      return false;
    }
    if (axis->GetXmin() >= axis->GetXmax()) {
      ::Error("NDRegression::SetHistogram", "Invalid range <%f,%f", axis->GetXmin(), axis->GetXmax());
      return false;
    }
  }

  return true;
}

//_____________________________________________________________________________
void NDRegression::EvaluateUni(const Int_t nvectors, const Double_t* data, Double_t& mean, Double_t& sigma, const Int_t hSub)
{
  //
  // Robust estimator in 1D case MI version - (faster than ROOT version)
  //
  // For the univariate case
  // estimates of location and scatter are returned in mean and sigma parameters
  // the algorithm works on the same principle as in multivariate case -
  // it finds a subset of size hSub with smallest sigma, and then returns mean and
  // sigma of this subset
  //

  Int_t hh = hSub;
  if (nvectors < 2) {
    ::Error("AliMathBase::EvaluateUni", "%s", Form("nvectors = %d, should be > 1", nvectors));
    return;
  }
  if (hh == nvectors) {
    mean = TMath::Mean(nvectors, data);
    sigma = TMath::RMS(nvectors, data);
    return;
  }
  if (hh < 2)
    hh = (nvectors + 2) / 2;
  Double_t faclts[] = {2.6477, 2.5092, 2.3826, 2.2662, 2.1587, 2.0589, 1.9660, 1.879, 1.7973, 1.7203, 1.6473};
  Int_t* index = new Int_t[nvectors];
  TMath::Sort(nvectors, data, index, kFALSE);

  Int_t nquant = TMath::Min(Int_t(Double_t(((hh * 1. / nvectors) - 0.5) * 40)) + 1, 11);
  Double_t factor = faclts[TMath::Max(0, nquant - 1)];

  Double_t sumx = 0;
  Double_t sumx2 = 0;
  Double_t bestmean = 0;
  Double_t bestsigma = (data[index[nvectors - 1]] - data[index[0]] + 1.); // maximal possible sigma
  bestsigma *= bestsigma;

  for (Int_t i = 0; i < hh; i++) {
    sumx += data[index[i]];
    sumx2 += data[index[i]] * data[index[i]];
  }

  Double_t norm = 1. / Double_t(hh);
  Double_t norm2 = 1. / Double_t(hh - 1);
  for (Int_t i = hh; i < nvectors; i++) {
    Double_t cmean = sumx * norm;
    Double_t csigma = (sumx2 - hh * cmean * cmean) * norm2;
    if (csigma < bestsigma) {
      bestmean = cmean;
      bestsigma = csigma;
    }

    sumx += data[index[i]] - data[index[i - hh]];
    sumx2 += data[index[i]] * data[index[i]] - data[index[i - hh]] * data[index[i - hh]];
  }

  Double_t bstd = factor * TMath::Sqrt(TMath::Abs(bestsigma));
  mean = bestmean;
  sigma = bstd;
  delete[] index;
}

Bool_t NDRegression::MakeRobustStatistic(TVectorD& values, TVectorD& errors, TObjArray& pointArray, TObjArray& kernelArrayI2, Int_t& nvarFormula, Double_t weightCut, Double_t robustFraction)
{
  //
  // Calculate robust statistic information
  // use raw array to make faster calcualtion
  const Int_t kMaxDim = 100;
  Double_t* pvalues = values.GetMatrixArray();
  Double_t* pvecVar[kMaxDim] = {0};
  Double_t* pvecKernelI2[kMaxDim] = {0};
  for (Int_t idim = 0; idim < pointArray.GetEntries(); idim++) {
    pvecVar[idim] = ((TVectorD*)(pointArray.At(idim)))->GetMatrixArray();
    pvecKernelI2[idim] = ((TVectorD*)(kernelArrayI2.At(idim)))->GetMatrixArray();
  }

  Double_t nchi2Cut = -2 * TMath::Log(weightCut); // transform probability to nsigma cut
  if (robustFraction > 1)
    robustFraction = 1;

  Int_t nbins = fHistPoints->GetNbins(); //
  Int_t npoints = values.GetNrows();     // number of points for fit
  if (fLocalRobustStat) {
    delete fLocalRobustStat;
  }
  fLocalRobustStat = new TMatrixD(nbins, 3);

  TVectorD valueLocal(npoints);
  for (Int_t ibin = 0; ibin < nbins; ibin++) {
    fHistPoints->GetBinContent(ibin, fBinIndex); //
    for (Int_t idim = 0; idim < nvarFormula; idim++) {
      fBinCenter[idim] = fHistPoints->GetAxis(idim)->GetBinCenter(fBinIndex[idim]);
      fBinWidth[idim] = fHistPoints->GetAxis(idim)->GetBinWidth(fBinIndex[idim]);
    }
    Int_t indexLocal = 0;
    for (Int_t ipoint = 0; ipoint < npoints; ipoint++) {
      Double_t sumChi2 = 0;
      for (Int_t idim = 0; idim < nvarFormula; idim++) {
        //TVectorD &vecVar=*((TVectorD*)(pointArray.UncheckedAt(idim)));
        //TVectorD &vecKernel=*((TVectorD*)(kernelArray.UncheckedAt(idim)));
        fBinDelta[idim] = pvecVar[idim][ipoint] - fBinCenter[idim];
        sumChi2 += (fBinDelta[idim] * fBinDelta[idim]) * pvecKernelI2[idim][ipoint];
        if (sumChi2 > nchi2Cut)
          break; //continue;
      }
      if (sumChi2 > nchi2Cut)
        continue;
      valueLocal[indexLocal] = pvalues[ipoint];
      indexLocal++;
    }
    Double_t median = 0, meanX = 0, rmsX = 0;
    if (indexLocal * robustFraction - 1 > 3) {
      median = TMath::Median(indexLocal, valueLocal.GetMatrixArray());
      EvaluateUni(indexLocal, valueLocal.GetMatrixArray(), meanX, rmsX, indexLocal * robustFraction - 1);
    }
    (*fLocalRobustStat)(ibin, 0) = median;
    (*fLocalRobustStat)(ibin, 1) = meanX;
    (*fLocalRobustStat)(ibin, 2) = rmsX;
  }
  return true;
}

Bool_t NDRegression::MakeFit(TTree* tree, const char* formulaVal, const char* formulaVar, const char* selection, const char* formulaKernel, const char* dimensionFormula, Double_t weightCut, Int_t entries, Bool_t useBinNorm)
{
  //
  //  Make a local fit in grid as specified by the input THn histogram
  //  Histogram has to be set before invocation of method
  //
  //  Output:
  //    array of fit parameters and covariance matrices
  //
  //  Input Parameters:
  //   tree        - input tree
  //   formulaVal  - : separated variable:error string
  //   formulaVar  - : separate varaible list
  //   selection   - selection (cut) for TTreeDraw
  //   kernelWidth - : separated list of width of kernel for local fitting
  //   dimenstionFormula - dummy for the moment
  //
  //Algorithm:
  //   1.) Check consistency of input data
  //
  //   2.) Cache input data from tree to the array of vector TVectorD
  //
  //   3.) Calculate robust local mean and robust local RMS in case outlier removal algorithm specified
  //
  //   4.) Make local fit
  //
  //  const Double_t kEpsilon=1e-6;
  const Int_t kMaxDim = 100;
  Int_t binRange[kMaxDim] = {0};
  Double_t nchi2Cut = -2 * TMath::Log(weightCut); // transform probability to nsigma cut
  if (fHistPoints == NULL) {
    ::Error("NDRegression::MakeFit", "ND histogram not initialized\n");
    return kFALSE;
  }
  if (tree == NULL || tree->GetEntries() == 0) {
    ::Error("NDRegression::MakeFit", "Empty tree\n");
    return kFALSE;
  }
  if (formulaVar == NULL || formulaVar == 0) {
    ::Error("NDRegression::MakeFit", "Empty variable list\n");
    return kFALSE;
  }
  if (formulaKernel == NULL) {
    ::Error("NDRegression::MakeFit", "Kernel width not specified\n");
    return kFALSE;
  }
  // fUseBinNorm = useBinNorm;
  //
  // fInputTree = tree; // should be better TRef?
  // fFormulaVal = new TObjString(formulaVal);
  // fFormulaVar = new TObjString(formulaVar);
  // fSelection = new TObjString(selection);
  // fKernelWidthFormula = new TObjString(formulaKernel);
  // fPolDimensionFormula = new TObjString(dimensionFormula);
  auto arrayFormulaVar = TString(formulaVar).Tokenize(":");
  Int_t nvarFormula = arrayFormulaVar->GetEntries();
  if (nvarFormula != fHistPoints->GetNdimensions()) {
    ::Error("NDRegression::MakeFit", "Histogram/points mismatch\n");
    return kFALSE;
  }

  // for (int i = 0; i < nvarFormula; i++) {
  //   std::cout << ((TString*)(arrayFormulaVar->At(i)))->View() << "\n";
  // }

  TObjArray* arrayKernel = TString(formulaKernel).Tokenize(":");
  Int_t nwidthFormula = arrayKernel->GetEntries();
  if (nvarFormula != nwidthFormula) {
    delete arrayKernel;
    delete arrayFormulaVar;
    ::Error("NDRegression::MakeFit", "Variable/Kernel mismath\n");
    return kFALSE;
  }
  // fNParameters = nvarFormula;

  //
  // 2.) Load input data
  //
  //
  Int_t entriesVal = tree->Draw(formulaVal, selection, "goffpara", entries);
  if (entriesVal == 0) {
    ::Error("NDRegression::MakeFit", "Empty point list\t%s\t%s\n", formulaVal, selection);
    return kFALSE;
  }
  if (tree->GetVal(0) == NULL || (tree->GetVal(1) == NULL)) {
    ::Error("NDRegression::MakeFit", "Wrong selection\t%s\t%s\n", formulaVar, selection);
    return kFALSE;
  }
  TVectorD values(entriesVal, tree->GetVal(0));
  TVectorD errors(entriesVal, tree->GetVal(1));
  Double_t* pvalues = values.GetMatrixArray();
  Double_t* perrors = errors.GetMatrixArray();

  // std::cout << std::endl;
  // values.Print();
  // std::cout << std::endl;

  // 2.b) variables
  TObjArray pointArray(nvarFormula);
  Int_t entriesVar = tree->Draw(formulaVar, selection, "goffpara", entries);
  if (entriesVal != entriesVar) {
    ::Error("NDRegression::MakeFit", "Wrong selection\t%s\t%s\n", formulaVar, selection);
    return kFALSE;
  }
  for (Int_t ipar = 0; ipar < nvarFormula; ipar++)
    pointArray.AddAt(new TVectorD(entriesVar, tree->GetVal(ipar)), ipar);

  // 2.c) kernel array
  TObjArray kernelArrayI2(nvarFormula);
  tree->Draw(formulaKernel, selection, "goffpara", entries);
  for (Int_t ipar = 0; ipar < nvarFormula; ipar++) {
    TVectorD* vdI2 = new TVectorD(entriesVar, tree->GetVal(ipar));
    for (int k = entriesVar; k--;) { // to speed up, precalculate inverse squared
      double kv = (*vdI2)[k];
      if (TMath::Abs(kv) < 1e-12)
        ::Fatal("NDRegression::MakeFit", "Kernel width=%f for entry %d of par:%d", kv, k, ipar);
      (*vdI2)[k] = 1. / (kv * kv);
    }
    kernelArrayI2.AddAt(vdI2, ipar);
  }
  //
  Double_t* pvecVar[kMaxDim] = {0};
  Double_t* pvecKernelI2[kMaxDim] = {0};
  for (Int_t idim = 0; idim < pointArray.GetEntries(); idim++) {
    pvecVar[idim] = ((TVectorD*)(pointArray.At(idim)))->GetMatrixArray();
    pvecKernelI2[idim] = ((TVectorD*)(kernelArrayI2.At(idim)))->GetMatrixArray();
    binRange[idim] = fHistPoints->GetAxis(idim)->GetNbins();
  }
  //
  //
  //
  Int_t nbins = fHistPoints->GetNbins();
  fBinIndex = new Int_t[fHistPoints->GetNdimensions()];
  fBinCenter = new Double_t[fHistPoints->GetNdimensions()];
  fBinDelta = new Double_t[fHistPoints->GetNdimensions()];
  fBinWidth = new Double_t[fHistPoints->GetNdimensions()];

  //
  // 3.)
  //
  if (fCutType > 0 && fRobustRMSLTSCut > 0) {
    MakeRobustStatistic(values, errors, pointArray, kernelArrayI2, nvarFormula, weightCut, fRobustFractionLTS);
  }
  //

  // 4.) Make local fits
  //

  Double_t* binHypFit = new Double_t[2 * fHistPoints->GetNdimensions()];
  //
  TLinearFitter fitter(1 + 2 * nvarFormula, TString::Format("hyp%d", 2 * nvarFormula).Data());
  for (Int_t ibin = 0; ibin < nbins; ibin++) {
    fHistPoints->GetBinContent(ibin, fBinIndex);
    Bool_t isUnderFlowBin = kFALSE;
    Bool_t isOverFlowBin = kFALSE;
    for (Int_t idim = 0; idim < nvarFormula; idim++) {
      if (fBinIndex[idim] == 0)
        isUnderFlowBin = kTRUE;
      if (fBinIndex[idim] > binRange[idim])
        isOverFlowBin = kTRUE;
      fBinCenter[idim] = fHistPoints->GetAxis(idim)->GetBinCenter(fBinIndex[idim]);
      fBinWidth[idim] = fHistPoints->GetAxis(idim)->GetBinWidth(fBinIndex[idim]);
    }
    if (isUnderFlowBin || isOverFlowBin)
      continue;
    fitter.ClearPoints();
    // add fit points
    for (Int_t ipoint = 0; ipoint < entriesVal; ipoint++) {
      Double_t sumChi2 = 0;
      if (fCutType > 0 && fRobustRMSLTSCut > 0) {
        Double_t localRMS = (*fLocalRobustStat)(ibin, 2);
        Double_t localMean = (*fLocalRobustStat)(ibin, 1);
        Double_t localMedian = (*fLocalRobustStat)(ibin, 0);
        if (fCutType == 1) {
          if (TMath::Abs(pvalues[ipoint] - localMedian) > fRobustRMSLTSCut * localRMS)
            continue;
        }
        if (fCutType == 2) {
          if (TMath::Abs(pvalues[ipoint] - localMean) > fRobustRMSLTSCut * localRMS)
            continue;
        }
      }
      for (Int_t idim = 0; idim < nvarFormula; idim++) {
        //TVectorD &vecVar=*((TVectorD*)(pointArray.UncheckedAt(idim)));
        //TVectorD &vecKernel=*((TVectorD*)(kernelArray.UncheckedAt(idim)));
        fBinDelta[idim] = pvecVar[idim][ipoint] - fBinCenter[idim];
        sumChi2 += (fBinDelta[idim] * fBinDelta[idim]) * pvecKernelI2[idim][ipoint];
        if (sumChi2 > nchi2Cut)
          break; //continue;
        if (useBinNorm) {
          binHypFit[2 * idim] = fBinDelta[idim] / fBinWidth[idim];
          binHypFit[2 * idim + 1] = binHypFit[2 * idim] * binHypFit[2 * idim];
        } else {
          binHypFit[2 * idim] = fBinDelta[idim];
          binHypFit[2 * idim + 1] = fBinDelta[idim] * fBinDelta[idim];
        }
      }
      if (sumChi2 > nchi2Cut)
        continue;
      //      Double_t weight=TMath::Exp(-sumChi2*0.5);
      //      fitter.AddPoint(binHypFit,pvalues[ipoint], perrors[ipoint]/weight);
      Double_t weightI = TMath::Exp(sumChi2 * 0.5);
      fitter.AddPoint(binHypFit, pvalues[ipoint], perrors[ipoint] * weightI);
    }
    TVectorD* fitParam = new TVectorD(nvarFormula * 2 + 1);
    TVectorD* fitQuality = new TVectorD(3);
    TMatrixD* fitCovar = new TMatrixD(nvarFormula * 2 + 1, nvarFormula * 2 + 1);
    Double_t normRMS = 0;
    Int_t nBinPoints = fitter.GetNpoints();
    Bool_t fitOK = kFALSE;
    (*fitQuality)[0] = 0;
    (*fitQuality)[1] = 0;
    (*fitQuality)[2] = 0;

    if (fitter.GetNpoints() > nvarFormula * 2 + 2) {

      if (fCutType == 3)
        fitOK = (fitter.EvalRobust() == 0);
      else
        fitOK = (fitter.Eval() == 0);

      if (fitOK) {
        normRMS = fitter.GetChisquare() / (fitter.GetNpoints() - fitter.GetNumberFreeParameters());
        fitter.GetParameters(*fitParam);
        fitter.GetCovarianceMatrix(*fitCovar);
        (*fitQuality)[0] = nBinPoints;
        (*fitQuality)[1] = normRMS;
        (*fitQuality)[2] = ibin;
        fLocalFitParam->AddAt(fitParam, ibin);
        fLocalFitQuality->AddAt(fitQuality, ibin);
        fLocalFitCovar->AddAt(fitCovar, ibin);
      }
    }
    if (fStreamer) {
      TVectorD pfBinCenter(nvarFormula, fBinCenter);
      Double_t median = 0, mean = 0, rms = 0;
      if (fLocalRobustStat) {
        median = (*fLocalRobustStat)(ibin, 0);
        mean = (*fLocalRobustStat)(ibin, 1);
        rms = (*fLocalRobustStat)(ibin, 2);
      }
      (*fStreamer) << "localFit"
                   << "ibin=" << ibin <<                                                                                              // bin index
        "fitOK=" << fitOK << "localMedian=" << median << "localMean=" << mean << "localRMS=" << rms << "nBinPoints=" << nBinPoints << // center of the bin
        "binCenter.=" << &pfBinCenter <<                                                                                              //
        "normRMS=" << normRMS << "fitParam.=" << fitParam << "fitCovar.=" << fitCovar << "fitOK=" << fitOK << "\n";
    }
    if (!fitOK) { // avoid memory leak for failed fits
      delete fitParam;
      delete fitQuality;
      delete fitCovar;
    }
  }

  return kTRUE;
}

Double_t NDRegression::GetCorrND(Double_t index, Double_t par0)
{
  //
  //
  NDRegression* corr =
    (NDRegression*)fgVisualCorrection->At(index);
  if (!corr)
    return 0;
  return corr->Eval(&par0);
}

Double_t NDRegression::GetCorrNDError(Double_t index, Double_t par0)
{
  //
  //
  NDRegression* corr =
    (NDRegression*)fgVisualCorrection->At(index);
  if (!corr)
    return 0;
  return corr->EvalError(&par0);
}

Double_t NDRegression::GetCorrND(Double_t index, Double_t par0,
                                 Double_t par1)
{
  //
  //
  NDRegression* corr =
    (NDRegression*)fgVisualCorrection->At(index);
  if (!corr)
    return 0;
  Double_t par[2] = {par0, par1};
  return corr->Eval(par);
}
Double_t NDRegression::GetCorrNDError(Double_t index, Double_t par0,
                                      Double_t par1)
{
  //
  //
  NDRegression* corr =
    (NDRegression*)fgVisualCorrection->At(index);
  if (!corr)
    return 0;
  Double_t par[2] = {par0, par1};
  return corr->EvalError(par);
}

void PlotData(TH1F* hData, TString xTitle = "xTitle", TString yTitle = "yTitle", Color_t color = kBlack, TString zTitle = "zTitle", Double_t rms = 999999., Double_t eRms = 0., Double_t mean = 999999., Double_t eMean = 0.)
{
  //
  //
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadLeftMargin(0.14);
  gStyle->SetPadBottomMargin(0.12);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPadGridX(1);
  gStyle->SetPadGridY(1);
  gStyle->SetOptStat(0);
  //
  if (color == (kRed + 2)) {
    hData->SetMarkerStyle(20);
  }
  if (color == (kBlue + 2)) {
    hData->SetMarkerStyle(21);
  }
  if (color == (kGreen + 2)) {
    hData->SetMarkerStyle(22);
    hData->SetMarkerSize(1.3);
  }

  hData->SetMarkerColor(color);
  hData->SetLineColor(color);
  hData->GetXaxis()->SetTitle(xTitle.Data());
  hData->GetYaxis()->SetTitle(yTitle.Data());
  hData->GetZaxis()->SetTitle(zTitle.Data());
  hData->GetXaxis()->SetTitleOffset(1.2);
  hData->GetXaxis()->SetTitleSize(0.05);
  hData->GetYaxis()->SetTitleOffset(1.3);
  hData->GetYaxis()->SetTitleSize(0.05);
  hData->GetXaxis()->SetLabelSize(0.035);
  hData->GetYaxis()->SetLabelSize(0.035);
  hData->GetXaxis()->SetDecimals();
  hData->GetYaxis()->SetDecimals();
  hData->GetZaxis()->SetDecimals();
  hData->Sumw2();
  hData->Draw("pe1");

  if (mean != 999999.) {
    TPaveText* text1 = new TPaveText(0.21, 0.82, 0.51, 0.92, "NDC");
    text1->SetTextFont(43);
    text1->SetTextSize(30.);
    text1->SetBorderSize(1);
    text1->SetFillColor(kWhite);
    text1->AddText(Form("Mean: %0.2f #pm %0.2f", mean, eMean));
    text1->AddText(Form("RMS: %0.2f #pm %0.2f", rms, eRms));
    text1->Draw();
  }
  if (rms != 999999. && mean == 999999.) {
    TPaveText* text1 = new TPaveText(0.21, 0.87, 0.51, 0.92, "NDC");
    text1->SetTextFont(43);
    text1->SetTextSize(30.);
    text1->SetBorderSize(1);
    text1->SetFillColor(kWhite);
    text1->AddText(Form("RMS: %0.2f", rms));
    text1->Draw();
  }
}
