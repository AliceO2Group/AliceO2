/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/


///////////////////////////////////////////////////////////////////////////
// Class AliMathBase
// 
// Subset of  matheamtical functions  not included in the TMath
//

///////////////////////////////////////////////////////////////////////////
#include "TMath.h"
#include "AliMathBase.h"
#include "Riostream.h"
#include "TH1F.h"
#include "TH3.h"
#include "TF1.h"
#include "TLinearFitter.h"

//
// includes neccessary for test functions
//

#include "TSystem.h"
#include "TRandom.h"
#include "TStopwatch.h"
 
ClassImp(AliMathBase) // Class implementation to enable ROOT I/O
 
AliMathBase::AliMathBase() : TObject()
{
  //
  // Default constructor
  //
}
///////////////////////////////////////////////////////////////////////////
AliMathBase::~AliMathBase()
{
  //
  // Destructor
  //
}


//_____________________________________________________________________________
void AliMathBase::EvaluateUni(Int_t nvectors, Double_t *data, Double_t &mean
                           , Double_t &sigma, Int_t hh)
{
  //
  // Robust estimator in 1D case MI version - (faster than ROOT version)
  //
  // For the univariate case
  // estimates of location and scatter are returned in mean and sigma parameters
  // the algorithm works on the same principle as in multivariate case -
  // it finds a subset of size hh with smallest sigma, and then returns mean and
  // sigma of this subset
  //

  if (nvectors<2) {
    //AliErrorClass(Form("nvectors = %d, should be > 1",nvectors));
    printf("nvectors = %d, should be > 1\n",nvectors);
    return;
  }
  if (hh<2)
    hh=(nvectors+2)/2;
  Double_t faclts[]={2.6477,2.5092,2.3826,2.2662,2.1587,2.0589,1.9660,1.879,1.7973,1.7203,1.6473};
  Int_t *index=new Int_t[nvectors];
  TMath::Sort(nvectors, data, index, kFALSE);
  
  Int_t    nquant = TMath::Min(Int_t(Double_t(((hh*1./nvectors)-0.5)*40))+1, 11);
  Double_t factor = faclts[TMath::Max(0,nquant-1)];
  
  Double_t sumx  =0;
  Double_t sumx2 =0;
  Int_t    bestindex = -1;
  Double_t bestmean  = 0; 
  Double_t bestsigma = (data[index[nvectors-1]]-data[index[0]]+1.);   // maximal possible sigma
  bestsigma *=bestsigma;

  for (Int_t i=0; i<hh; i++){
    sumx  += data[index[i]];
    sumx2 += data[index[i]]*data[index[i]];
  }
  
  Double_t norm = 1./Double_t(hh);
  Double_t norm2 = 1./Double_t(hh-1);
  for (Int_t i=hh; i<nvectors; i++){
    Double_t cmean  = sumx*norm;
    Double_t csigma = (sumx2 - hh*cmean*cmean)*norm2;
    if (csigma<bestsigma){
      bestmean  = cmean;
      bestsigma = csigma;
      bestindex = i-hh;
    }
    
    sumx  += data[index[i]]-data[index[i-hh]];
    sumx2 += data[index[i]]*data[index[i]]-data[index[i-hh]]*data[index[i-hh]];
  }
  
  Double_t bstd=factor*TMath::Sqrt(TMath::Abs(bestsigma));
  mean  = bestmean;
  sigma = bstd;
  delete [] index;

}



void AliMathBase::EvaluateUniExternal(Int_t nvectors, Double_t *data, Double_t &mean, Double_t &sigma, Int_t hh,  Float_t externalfactor)
{
  // Modified version of ROOT robust EvaluateUni
  // robust estimator in 1D case MI version
  // added external factor to include precision of external measurement
  // 

  if (hh==0)
    hh=(nvectors+2)/2;
  Double_t faclts[]={2.6477,2.5092,2.3826,2.2662,2.1587,2.0589,1.9660,1.879,1.7973,1.7203,1.6473};
  Int_t *index=new Int_t[nvectors];
  TMath::Sort(nvectors, data, index, kFALSE);
  //
  Int_t    nquant = TMath::Min(Int_t(Double_t(((hh*1./nvectors)-0.5)*40))+1, 11);
  Double_t factor = faclts[0];
  if (nquant>0){
    // fix proper normalization - Anja
    factor = faclts[nquant-1];
  }

  //
  //
  Double_t sumx  =0;
  Double_t sumx2 =0;
  Int_t    bestindex = -1;
  Double_t bestmean  = 0; 
  Double_t bestsigma = -1;
  for (Int_t i=0; i<hh; i++){
    sumx  += data[index[i]];
    sumx2 += data[index[i]]*data[index[i]];
  }
  //   
  Double_t kfactor = 2.*externalfactor - externalfactor*externalfactor;
  Double_t norm = 1./Double_t(hh);
  for (Int_t i=hh; i<nvectors; i++){
    Double_t cmean  = sumx*norm;
    Double_t csigma = (sumx2*norm - cmean*cmean*kfactor);
    if (csigma<bestsigma ||  bestsigma<0){
      bestmean  = cmean;
      bestsigma = csigma;
      bestindex = i-hh;
    }
    //
    //
    sumx  += data[index[i]]-data[index[i-hh]];
    sumx2 += data[index[i]]*data[index[i]]-data[index[i-hh]]*data[index[i-hh]];
  }
  
  Double_t bstd=factor*TMath::Sqrt(TMath::Abs(bestsigma));
  mean  = bestmean;
  sigma = bstd;
  delete [] index;
}


//_____________________________________________________________________________
Int_t AliMathBase::Freq(Int_t n, const Int_t *inlist
                        , Int_t *outlist, Bool_t down)
{    
  //
  //  Sort eleements according occurancy 
  //  The size of output array has is 2*n 
  //

  Int_t * sindexS = new Int_t[n];     // temp array for sorting
  Int_t * sindexF = new Int_t[2*n];   
  for (Int_t i=0;i<n;i++) sindexF[i]=0;
  //
  TMath::Sort(n,inlist, sindexS, down);  
  Int_t last      = inlist[sindexS[0]];
  Int_t val       = last;
  sindexF[0]      = 1;
  sindexF[0+n]    = last;
  Int_t countPos  = 0;
  //
  //  find frequency
  for(Int_t i=1;i<n; i++){
    val = inlist[sindexS[i]];
    if (last == val)   sindexF[countPos]++;
    else{      
      countPos++;
      sindexF[countPos+n] = val;
      sindexF[countPos]++;
      last =val;
    }
  }
  if (last==val) countPos++;
  // sort according frequency
  TMath::Sort(countPos, sindexF, sindexS, kTRUE);
  for (Int_t i=0;i<countPos;i++){
    outlist[2*i  ] = sindexF[sindexS[i]+n];
    outlist[2*i+1] = sindexF[sindexS[i]];
  }
  delete [] sindexS;
  delete [] sindexF;
  
  return countPos;

}

//___AliMathBase__________________________________________________________________________
void AliMathBase::TruncatedMean(TH1F * his, TVectorD *param, Float_t down, Float_t up, Bool_t verbose){
  //
  //
  //
  Int_t nbins    = his->GetNbinsX();
  Float_t nentries = his->GetEntries();
  Float_t sum      =0;
  Float_t mean   = 0;
  Float_t sigma2 = 0;
  Float_t ncumul=0;  
  for (Int_t ibin=1;ibin<nbins; ibin++){
    ncumul+= his->GetBinContent(ibin);
    Float_t fraction = Float_t(ncumul)/Float_t(nentries);
    if (fraction>down && fraction<up){
      sum+=his->GetBinContent(ibin);
      mean+=his->GetBinCenter(ibin)*his->GetBinContent(ibin);
      sigma2+=his->GetBinCenter(ibin)*his->GetBinCenter(ibin)*his->GetBinContent(ibin);      
    }
  }
  mean/=sum;
  sigma2= TMath::Sqrt(TMath::Abs(sigma2/sum-mean*mean));
  if (param){
    (*param)[0] = his->GetMaximum();
    (*param)[1] = mean;
    (*param)[2] = sigma2;
    
  }
  if (verbose)  printf("Mean\t%f\t Sigma2\t%f\n", mean,sigma2);
}

void AliMathBase::LTM(TH1F * his, TVectorD *param , Float_t fraction,  Bool_t verbose){
  //
  // LTM
  //
  Int_t nbins    = his->GetNbinsX();
  Int_t nentries = (Int_t)his->GetEntries();
  Double_t *data  = new Double_t[nentries];
  Int_t npoints=0;
  for (Int_t ibin=1;ibin<nbins; ibin++){
    Float_t entriesI = his->GetBinContent(ibin);
    Float_t xcenter= his->GetBinCenter(ibin);
    for (Int_t ic=0; ic<entriesI; ic++){
      if (npoints<nentries){
	data[npoints]= xcenter;
	npoints++;
      }
    }
  }
  Double_t mean, sigma;
  Int_t npoints2=TMath::Min(Int_t(fraction*Float_t(npoints)),npoints-1);
  npoints2=TMath::Max(Int_t(0.5*Float_t(npoints)),npoints2);
  AliMathBase::EvaluateUni(npoints, data, mean,sigma,npoints2);
  delete [] data;
  if (verbose)  printf("Mean\t%f\t Sigma2\t%f\n", mean,sigma);if (param){
    (*param)[0] = his->GetMaximum();
    (*param)[1] = mean;
    (*param)[2] = sigma;    
  }
}

Double_t  AliMathBase::FitGaus(TH1F* his, TVectorD *param, TMatrixD */*matrix*/, Float_t xmin, Float_t xmax, Bool_t verbose){
  //
  //  Fit histogram with gaussian function
  //  
  //  Prameters:
  //       return value- chi2 - if negative ( not enough points)
  //       his        -  input histogram
  //       param      -  vector with parameters 
  //       xmin, xmax -  range to fit - if xmin=xmax=0 - the full histogram range used
  //  Fitting:
  //  1. Step - make logarithm
  //  2. Linear  fit (parabola) - more robust - always converge
  //  3. In case of small statistic bins are averaged
  //  
  static TLinearFitter fitter(3,"pol2");
  TVectorD  par(3);
  TVectorD  sigma(3);
  TMatrixD mat(3,3);
  if (his->GetMaximum()<4) return -1;  
  if (his->GetEntries()<12) return -1;  
  if (his->GetRMS()<mat.GetTol()) return -1;
  Float_t maxEstimate   = his->GetEntries()*his->GetBinWidth(1)/TMath::Sqrt((TMath::TwoPi()*his->GetRMS()));
  Int_t dsmooth = TMath::Nint(6./TMath::Sqrt(maxEstimate));

  if (maxEstimate<1) return -1;
  Int_t nbins    = his->GetNbinsX();
  Int_t npoints=0;
  //


  if (xmin>=xmax){
    xmin = his->GetXaxis()->GetXmin();
    xmax = his->GetXaxis()->GetXmax();
  }
  for (Int_t iter=0; iter<2; iter++){
    fitter.ClearPoints();
    npoints=0;
    for (Int_t ibin=1;ibin<nbins+1; ibin++){
      Int_t countB=1;
      Float_t entriesI =  his->GetBinContent(ibin);
      for (Int_t delta = -dsmooth; delta<=dsmooth; delta++){
	if (ibin+delta>1 &&ibin+delta<nbins-1){
	  entriesI +=  his->GetBinContent(ibin+delta);
	  countB++;
	}
      }
      entriesI/=countB;
      Double_t xcenter= his->GetBinCenter(ibin);
      if (xcenter<xmin || xcenter>xmax) continue;
      Double_t error=1./TMath::Sqrt(countB);
      Float_t   cont=2;
      if (iter>0){
	if (par[0]+par[1]*xcenter+par[2]*xcenter*xcenter>20) return 0;
	cont = TMath::Exp(par[0]+par[1]*xcenter+par[2]*xcenter*xcenter);
	if (cont>1.) error = 1./TMath::Sqrt(cont*Float_t(countB));
      }
      if (entriesI>1&&cont>1){
	fitter.AddPoint(&xcenter,TMath::Log(Float_t(entriesI)),error);
	npoints++;
      }
    }  
    if (npoints>3){
      fitter.Eval();
      fitter.GetParameters(par);
    }else{
      break;
    }
  }
  if (npoints<=3){
    return -1;
  }
  fitter.GetParameters(par);
  fitter.GetCovarianceMatrix(mat);
  if (TMath::Abs(par[1])<mat.GetTol()) return -1;
  if (TMath::Abs(par[2])<mat.GetTol()) return -1;
  Double_t chi2 = fitter.GetChisquare()/Float_t(npoints);
  //fitter.GetParameters();
  if (!param)  param  = new TVectorD(3);
  //if (!matrix) matrix = new TMatrixD(3,3);
  (*param)[1] = par[1]/(-2.*par[2]);
  (*param)[2] = 1./TMath::Sqrt(TMath::Abs(-2.*par[2]));
  (*param)[0] = TMath::Exp(par[0]+ par[1]* (*param)[1] +  par[2]*(*param)[1]*(*param)[1]);
  if (verbose){
    par.Print();
    mat.Print();
    param->Print();
    printf("Chi2=%f\n",chi2);
    TF1 * f1= new TF1("f1","[0]*exp(-(x-[1])^2/(2*[2]*[2]))",his->GetXaxis()->GetXmin(),his->GetXaxis()->GetXmax());
    f1->SetParameter(0, (*param)[0]);
    f1->SetParameter(1, (*param)[1]);
    f1->SetParameter(2, (*param)[2]);    
    f1->Draw("same");
  }
  return chi2;
}

Double_t  AliMathBase::FitGaus(Float_t *arr, Int_t nBins, Float_t xMin, Float_t xMax, TVectorD *param, TMatrixD */*matrix*/, Bool_t verbose){
  //
  //  Fit histogram with gaussian function
  //  
  //  Prameters:
  //     nbins: size of the array and number of histogram bins
  //     xMin, xMax: histogram range
  //     param: paramters of the fit (0-Constant, 1-Mean, 2-Sigma, 3-Sum)
  //     matrix: covariance matrix -- not implemented yet, pass dummy matrix!!!
  //
  //  Return values:
  //    >0: the chi2 returned by TLinearFitter
  //    -3: only three points have been used for the calculation - no fitter was used
  //    -2: only two points have been used for the calculation - center of gravity was uesed for calculation
  //    -1: only one point has been used for the calculation - center of gravity was uesed for calculation
  //    -4: invalid result!!
  //
  //  Fitting:
  //  1. Step - make logarithm
  //  2. Linear  fit (parabola) - more robust - always converge
  //  
  static TLinearFitter fitter(3,"pol2");
  static TMatrixD mat(3,3);
  static Double_t kTol = mat.GetTol();
  fitter.StoreData(kFALSE);
  fitter.ClearPoints();
  TVectorD  par(3);
  TVectorD  sigma(3);
  TMatrixD A(3,3);
  TMatrixD b(3,1);
  Float_t rms = TMath::RMS(nBins,arr);
  Float_t max = TMath::MaxElement(nBins,arr);
  Float_t binWidth = (xMax-xMin)/(Float_t)nBins;

  Float_t meanCOG = 0;
  Float_t rms2COG = 0;
  Float_t sumCOG  = 0;

  Float_t entries = 0;
  Int_t nfilled=0;

  if (!param)  param  = new TVectorD(4);

  for (Int_t i=0; i<nBins; i++){
      entries+=arr[i];
      if (arr[i]>0) nfilled++;
  }
  (*param)[0] = 0;
  (*param)[1] = 0;
  (*param)[2] = 0;
  (*param)[3] = 0;

  if (max<4) return -4;
  if (entries<12) return -4;
  if (rms<kTol) return -4;

  (*param)[3] = entries;

  Int_t npoints=0;
  for (Int_t ibin=0;ibin<nBins; ibin++){
      Float_t entriesI = arr[ibin];
    if (entriesI>1){
      Double_t xcenter = xMin+(ibin+0.5)*binWidth;
      Float_t error    = 1./TMath::Sqrt(entriesI);
      Float_t val = TMath::Log(Float_t(entriesI));
      fitter.AddPoint(&xcenter,val,error);
      if (npoints<3){
	  A(npoints,0)=1;
	  A(npoints,1)=xcenter;
	  A(npoints,2)=xcenter*xcenter;
	  b(npoints,0)=val;
	  meanCOG+=xcenter*entriesI;
	  rms2COG +=xcenter*entriesI*xcenter;
	  sumCOG +=entriesI;
      }
      npoints++;
    }
  }
  
  Double_t chi2 = 0;
  if (npoints>=3){
      if ( npoints == 3 ){
	  //analytic calculation of the parameters for three points
	  A.Invert();
	  TMatrixD res(1,3);
	  res.Mult(A,b);
	  par[0]=res(0,0);
	  par[1]=res(0,1);
	  par[2]=res(0,2);
          chi2 = -3.;
      } else {
          // use fitter for more than three points
	  fitter.Eval();
	  fitter.GetParameters(par);
	  fitter.GetCovarianceMatrix(mat);
	  chi2 = fitter.GetChisquare()/Float_t(npoints);
      }
      if (TMath::Abs(par[1])<kTol) return -4;
      if (TMath::Abs(par[2])<kTol) return -4;

      //if (!param)  param  = new TVectorD(4);
      if ( param->GetNrows()<4 ) param->ResizeTo(4);
      //if (!matrix) matrix = new TMatrixD(3,3);  // !!!!might be a memory leek. use dummy matrix pointer to call this function!

      (*param)[1] = par[1]/(-2.*par[2]);
      (*param)[2] = 1./TMath::Sqrt(TMath::Abs(-2.*par[2]));
      Double_t lnparam0 = par[0]+ par[1]* (*param)[1] +  par[2]*(*param)[1]*(*param)[1];
      if ( lnparam0>307 ) return -4;
      (*param)[0] = TMath::Exp(lnparam0);
      if (verbose){
	  par.Print();
	  mat.Print();
	  param->Print();
	  printf("Chi2=%f\n",chi2);
	  TF1 * f1= new TF1("f1","[0]*exp(-(x-[1])^2/(2*[2]*[2]))",xMin,xMax);
	  f1->SetParameter(0, (*param)[0]);
	  f1->SetParameter(1, (*param)[1]);
	  f1->SetParameter(2, (*param)[2]);
	  f1->Draw("same");
      }
      return chi2;
  }

  if (npoints == 2){
      //use center of gravity for 2 points
      meanCOG/=sumCOG;
      rms2COG /=sumCOG;
      (*param)[0] = max;
      (*param)[1] = meanCOG;
      (*param)[2] = TMath::Sqrt(TMath::Abs(meanCOG*meanCOG-rms2COG));
      chi2=-2.;
  }
  if ( npoints == 1 ){
      meanCOG/=sumCOG;
      (*param)[0] = max;
      (*param)[1] = meanCOG;
      (*param)[2] = binWidth/TMath::Sqrt(12);
      chi2=-1.;
  }
  return chi2;

}


Float_t AliMathBase::GetCOG(Short_t *arr, Int_t nBins, Float_t xMin, Float_t xMax, Float_t *rms, Float_t *sum)
{
    //
    //  calculate center of gravity rms and sum for array 'arr' with nBins an a x range xMin to xMax
    //  return COG; in case of failure return xMin
    //
    Float_t meanCOG = 0;
    Float_t rms2COG = 0;
    Float_t sumCOG  = 0;
    Int_t npoints   = 0;

    Float_t binWidth = (xMax-xMin)/(Float_t)nBins;

    for (Int_t ibin=0; ibin<nBins; ibin++){
	Float_t entriesI = (Float_t)arr[ibin];
	Double_t xcenter = xMin+(ibin+0.5)*binWidth;
	if ( entriesI>0 ){
	    meanCOG += xcenter*entriesI;
	    rms2COG += xcenter*entriesI*xcenter;
	    sumCOG  += entriesI;
	    npoints++;
	}
    }
    if ( sumCOG == 0 ) return xMin;
    meanCOG/=sumCOG;

    if ( rms ){
	rms2COG /=sumCOG;
	(*rms) = TMath::Sqrt(TMath::Abs(meanCOG*meanCOG-rms2COG));
	if ( npoints == 1 ) (*rms) = binWidth/TMath::Sqrt(12);
    }

    if ( sum )
        (*sum) = sumCOG;

    return meanCOG;
}


Double_t AliMathBase::ErfcFast(Double_t x){
  // Fast implementation of the complementary error function
  // The error of the approximation is |eps(x)| < 5E-4
  // See Abramowitz and Stegun, p.299, 7.1.27

  Double_t z = TMath::Abs(x);
  Double_t ans = 1+z*(0.278393+z*(0.230389+z*(0.000972+z*0.078108)));
  ans = 1.0/ans;
  ans *= ans;
  ans *= ans;

  return (x>=0.0 ? ans : 2.0 - ans);
}

///////////////////////////////////////////////////////////////
//////////////         TEST functions /////////////////////////
///////////////////////////////////////////////////////////////






TGraph2D * AliMathBase::MakeStat2D(TH3 * his, Int_t delta0, Int_t delta1, Int_t type){
  //
  //
  //
  // delta - number of bins to integrate
  // type - 0 - mean value

  TAxis * xaxis  = his->GetXaxis();
  TAxis * yaxis  = his->GetYaxis();
  //  TAxis * zaxis  = his->GetZaxis();
  Int_t   nbinx  = xaxis->GetNbins();
  Int_t   nbiny  = yaxis->GetNbins();
  const Int_t nc=1000;
  char name[nc];
  Int_t icount=0;
  TGraph2D  *graph = new TGraph2D(nbinx*nbiny);
  TF1 f1("f1","gaus");
  for (Int_t ix=0; ix<nbinx;ix++)
    for (Int_t iy=0; iy<nbiny;iy++){
      Float_t xcenter = xaxis->GetBinCenter(ix); 
      Float_t ycenter = yaxis->GetBinCenter(iy); 
      snprintf(name,nc,"%s_%d_%d",his->GetName(), ix,iy);
      TH1 *projection = his->ProjectionZ(name,ix-delta0,ix+delta0,iy-delta1,iy+delta1);
      Float_t stat= 0;
      if (type==0) stat = projection->GetMean();
      if (type==1) stat = projection->GetRMS();
      if (type==2 || type==3){
	TVectorD vec(3);
	AliMathBase::LTM((TH1F*)projection,&vec,0.7);
	if (type==2) stat= vec[1];
	if (type==3) stat= vec[0];	
      }
      if (type==4|| type==5){
	projection->Fit(&f1);
	if (type==4) stat= f1.GetParameter(1);
	if (type==5) stat= f1.GetParameter(2);
      }
      //printf("%d\t%f\t%f\t%f\n", icount,xcenter, ycenter, stat);
      graph->SetPoint(icount,xcenter, ycenter, stat);
      icount++;
    }
  return graph;
}

TGraph * AliMathBase::MakeStat1D(TH3 * his, Int_t delta1, Int_t type){
  //
  //
  //
  // delta - number of bins to integrate
  // type - 0 - mean value

  TAxis * xaxis  = his->GetXaxis();
  TAxis * yaxis  = his->GetYaxis();
  //  TAxis * zaxis  = his->GetZaxis();
  Int_t   nbinx  = xaxis->GetNbins();
  Int_t   nbiny  = yaxis->GetNbins();
  const Int_t nc=1000;
  char name[nc];
  Int_t icount=0;
  TGraph  *graph = new TGraph(nbinx);
  TF1 f1("f1","gaus");
  for (Int_t ix=0; ix<nbinx;ix++){
    Float_t xcenter = xaxis->GetBinCenter(ix); 
    //    Float_t ycenter = yaxis->GetBinCenter(iy); 
    snprintf(name,nc,"%s_%d",his->GetName(), ix);
    TH1 *projection = his->ProjectionZ(name,ix-delta1,ix+delta1,0,nbiny);
    Float_t stat= 0;
    if (type==0) stat = projection->GetMean();
    if (type==1) stat = projection->GetRMS();
    if (type==2 || type==3){
      TVectorD vec(3);
	AliMathBase::LTM((TH1F*)projection,&vec,0.7);
	if (type==2) stat= vec[1];
	if (type==3) stat= vec[0];	
    }
    if (type==4|| type==5){
      projection->Fit(&f1);
      if (type==4) stat= f1.GetParameter(1);
      if (type==5) stat= f1.GetParameter(2);
    }
      //printf("%d\t%f\t%f\t%f\n", icount,xcenter, ycenter, stat);
    graph->SetPoint(icount,xcenter, stat);
    icount++;
  }
  return graph;
}

Double_t AliMathBase::TruncatedGaus(Double_t mean, Double_t sigma, Double_t cutat)
{
  // return number generated according to a gaussian distribution N(mean,sigma) truncated at cutat
  //
  Double_t value;
  do{
    value=gRandom->Gaus(mean,sigma);
  }while(TMath::Abs(value-mean)>cutat);
  return value;
}

Double_t AliMathBase::TruncatedGaus(Double_t mean, Double_t sigma, Double_t leftCut, Double_t rightCut)
{
  // return number generated according to a gaussian distribution N(mean,sigma)
  // truncated at leftCut and rightCut
  //
  Double_t value;
  do{
    value=gRandom->Gaus(mean,sigma);
  }while((value-mean)<-leftCut || (value-mean)>rightCut);
  return value;
}

Double_t AliMathBase::BetheBlochAleph(Double_t bg,
         Double_t kp1,
         Double_t kp2,
         Double_t kp3,
         Double_t kp4,
         Double_t kp5) {
  //
  // This is the empirical ALEPH parameterization of the Bethe-Bloch formula.
  // It is normalized to 1 at the minimum.
  //
  // bg - beta*gamma
  // 
  // The default values for the kp* parameters are for ALICE TPC.
  // The returned value is in MIP units
  //
  Double_t beta = bg/TMath::Sqrt(1.+ bg*bg);

  Double_t aa = TMath::Power(beta,kp4);
  Double_t bb = TMath::Power(1./bg,kp5);

  bb=TMath::Log(kp3+bb);

  return (kp2-aa-bb)*kp1/aa;
}
