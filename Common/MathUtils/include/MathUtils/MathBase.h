// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MATHUTILS_MATHBASE_H_
#define ALICEO2_MATHUTILS_MATHBASE_H_

/// \file   MathBase.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include "Rtypes.h"
#include "TLinearFitter.h"
#include "TVectorD.h"
#include "TMath.h"

namespace o2 {
namespace mathUtils {
namespace mathBase {
  /// fast fit of an array with ranges (histogram) with gaussian function
  /// 
  /// Fitting procedure:
  /// 1. Step - make logarithm
  /// 2. Linear  fit (parabola) - more robust, always converges, fast
  ///
  /// \param[in]  nbins size of the array and number of histogram bins
  /// \param[in]  arr   array with elements
  /// \param[in]  xMin  minimum range of the array
  /// \param[in]  xMax  maximum range of the array
  /// \param[out] param return paramters of the fit (0-Constant, 1-Mean, 2-Sigma, 3-Sum)
  ///
  /// \return chi2 or exit code
  ///          >0: the chi2 returned by TLinearFitter
  ///          -3: only three points have been used for the calculation - no fitter was used
  ///          -2: only two points have been used for the calculation - center of gravity was uesed for calculation
  ///          -1: only one point has been used for the calculation - center of gravity was uesed for calculation
  ///          -4: invalid result!!
  ///
  //template <typename T>
  //Double_t  fitGaus(const size_t nBins, const T *arr, const T xMin, const T xMax, std::vector<T>& param);
  template <typename T>
  Double_t  fitGaus(const size_t nBins, const T *arr, const T xMin, const T xMax, std::vector<T>& param)
  {
    static TLinearFitter fitter(3,"pol2");
    static TMatrixD mat(3,3);
    static Double_t kTol = mat.GetTol();
    fitter.StoreData(kFALSE);
    fitter.ClearPoints();
    TVectorD  par(3);
    TVectorD  sigma(3);
    TMatrixD A(3,3);
    TMatrixD b(3,1);
    T rms = TMath::RMS(nBins,arr);
    T max = TMath::MaxElement(nBins,arr);
    T binWidth = (xMax-xMin)/T(nBins);
  
    Float_t meanCOG = 0;
    Float_t rms2COG = 0;
    Float_t sumCOG  = 0;
  
    Float_t entries = 0;
    Int_t nfilled=0;
  
    param.resize(4);
    param[0] = 0.;
    param[1] = 0.;
    param[2] = 0.;
    param[3] = 0.;
  
    for (Int_t i=0; i<nBins; i++){
        entries+=arr[i];
        if (arr[i]>0) nfilled++;
    }
  
    // TODO: Check why this is needed
    if (max<4) return -4;
    if (entries<12) return -4;

    if (rms<kTol) return -4;
  
    param[3] = entries;
  
    Int_t npoints=0;
    for (Int_t ibin=0;ibin<nBins; ibin++){
        Float_t entriesI = arr[ibin];
      if (entriesI>1){
        Double_t xcenter = xMin+(ibin+0.5)*binWidth;
        Double_t error    = 1./TMath::Sqrt(entriesI);
        Double_t val = TMath::Log(Float_t(entriesI));
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
            chi2 = fitter.GetChisquare()/Double_t(npoints);
        }
        if (TMath::Abs(par[1])<kTol) return -4;
        if (TMath::Abs(par[2])<kTol) return -4;
  
        param[1] = T(par[1]/(-2.*par[2]));
        param[2] = T(1./TMath::Sqrt(TMath::Abs(-2.*par[2])));
        Double_t lnparam0 = par[0]+ par[1]* param[1] +  par[2]*param[1]*param[1];
        if ( lnparam0>307 ) return -4;
        param[0] = TMath::Exp(lnparam0);
  
        return chi2;
    }
  
    if (npoints == 2){
        //use center of gravity for 2 points
        meanCOG/=sumCOG;
        rms2COG /=sumCOG;
        param[0] = max;
        param[1] = meanCOG;
        param[2] = TMath::Sqrt(TMath::Abs(meanCOG*meanCOG-rms2COG));
        chi2=-2.;
    }
    if ( npoints == 1 ){
        meanCOG/=sumCOG;
        param[0] = max;
        param[1] = meanCOG;
        param[2] = binWidth/TMath::Sqrt(12);
        chi2=-1.;
    }
    return chi2;
  
  }

} // namespace mathBase
} // namespace mathUtils
} // namespace o2
#endif
