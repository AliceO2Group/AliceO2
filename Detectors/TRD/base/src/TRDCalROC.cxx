// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  Calibration base class for a single ROC                                  //
//  Contains one UShort_t value per pad                                      //
//  However, values are set and get as float, there are stored internally as //
//  (UShort_t) value * 10000                                                 //
//                                                                           //
//  2019 - Ported from AliRoot to O2 (J. Lopez)                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TMath.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TRobustEstimator.h>

#include "TRDBase/TRDCalROC.h"

using namespace o2::trd;

//_____________________________________________________________________________
TRDCalROC::TRDCalROC(int p, int c)
  : mPla(p),
    mCha(c),
    mNcols(144)
{
  //
  // Constructor that initializes a given pad plane type
  //

  //
  // The pad plane parameter
  //
  switch (p) {
    case 0:
      if (c == 2) {
        // L0C0 type
        mNrows = 12;
      } else {
        // L0C1 type
        mNrows = 16;
      }
      break;
    case 1:
      if (c == 2) {
        // L1C0 type
        mNrows = 12;
      } else {
        // L1C1 type
        mNrows = 16;
      }
      break;
    case 2:
      if (c == 2) {
        // L2C0 type
        mNrows = 12;
      } else {
        // L2C1 type
        mNrows = 16;
      }
      break;
    case 3:
      if (c == 2) {
        // L3C0 type
        mNrows = 12;
      } else {
        // L3C1 type
        mNrows = 16;
      }
      break;
    case 4:
      if (c == 2) {
        // L4C0 type
        mNrows = 12;
      } else {
        // L4C1 type
        mNrows = 16;
      }
      break;
    case 5:
      if (c == 2) {
        // L5C0 type
        mNrows = 12;
      } else {
        // L5C1 type
        mNrows = 16;
      }
      break;
  };

  mNchannels = mNrows * mNcols;
  if (mNchannels != 0) {
    mData = new unsigned short[mNchannels];
  }

  for (int i = 0; i < mNchannels; ++i) {
    mData[i] = 0;
  }
}

//_____________________________________________________________________________
TRDCalROC::~TRDCalROC()
{
  //
  // TRDCalROC destructor
  //

  if (mData) {
    delete[] mData;
    mData = nullptr;
  }
}

//___________________________________________________________________________________
double TRDCalROC::getMean(TRDCalROC* const outlierROC) const
{
  //
  // Calculate the mean
  //

  double* ddata = new double[mNchannels];
  int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if ((!outlierROC) || (!(outlierROC->getValue(i)))) {
      //if(mData[i] > 0.000000000000001){
      ddata[nPoints] = (double)mData[i] / 10000;
      nPoints++;
      //}
    }
  }
  double mean = TMath::Mean(nPoints, ddata);
  delete[] ddata;
  return mean;
}
//___________________________________________________________________________________
double TRDCalROC::getMeanNotNull() const
{
  //
  // Calculate the mean rejecting null value
  //

  double* ddata = new double[mNchannels];
  int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if (mData[i] > 0.000000000000001) {
      ddata[nPoints] = (double)mData[i] / 10000;
      nPoints++;
    }
  }
  if (nPoints < 1) {
    delete[] ddata;
    return -1;
  }
  double mean = TMath::Mean(nPoints, ddata);
  delete[] ddata;
  return mean;
}

//_______________________________________________________________________________________
double TRDCalROC::getMedian(TRDCalROC* const outlierROC) const
{
  //
  // Calculate the median
  //

  double* ddata = new double[mNchannels];
  int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if ((!outlierROC) || (!(outlierROC->getValue(i)))) {
      if (mData[i] > 0.000000000000001) {
        ddata[nPoints] = (double)mData[i] / 10000;
        nPoints++;
      }
    }
  }
  double mean = TMath::Median(nPoints, ddata);
  delete[] ddata;
  return mean;
}

//____________________________________________________________________________________________
double TRDCalROC::getRMS(TRDCalROC* const outlierROC) const
{
  //
  // Calculate the RMS
  //

  double* ddata = new double[mNchannels];
  int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if ((!outlierROC) || (!(outlierROC->getValue(i)))) {
      //if(mData[i] > 0.000000000000001){
      ddata[nPoints] = (double)mData[i] / 10000;
      nPoints++;
      //}
    }
  }
  double mean = TMath::RMS(nPoints, ddata);
  delete[] ddata;
  return mean;
}

//____________________________________________________________________________________________
double TRDCalROC::getRMSNotNull() const
{
  //
  // Calculate the RMS
  //

  double* ddata = new double[mNchannels];
  int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if (mData[i] > 0.000000000000001) {
      ddata[nPoints] = (double)mData[i] / 10000;
      nPoints++;
    }
  }
  if (nPoints < 1) {
    delete[] ddata;
    return -1;
  }
  double mean = TMath::RMS(nPoints, ddata);
  delete[] ddata;
  return mean;
}
//______________________________________________________________________________________________
double TRDCalROC::getLTM(double* sigma, double fraction, TRDCalROC* const outlierROC)
{
  //
  //  Calculate LTM mean and sigma
  //

  double* ddata = new double[mNchannels];
  double mean = 0, lsigma = 0;
  unsigned int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if (!outlierROC || !(outlierROC->getValue(i))) {
      if (mData[i] > 0.000000000000001) {
        ddata[nPoints] = (double)mData[i] / 10000;
        nPoints++;
      }
    }
  }
  int hh = TMath::Min(TMath::Nint(fraction * nPoints), int(nPoints));
  TRobustEstimator tre;
  tre.EvaluateUni(nPoints, ddata, mean, lsigma, hh);
  if (sigma) {
    *sigma = lsigma;
  }
  delete[] ddata;
  return mean;
}

//___________________________________________________________________________________
bool TRDCalROC::add(float c1)
{
  //
  // add constant
  //

  bool result = true;
  for (int idata = 0; idata < mNchannels; idata++) {
    if (((getValue(idata) + c1) <= 6.5535) && ((getValue(idata) + c1) >= 0.0))
      setValue(idata, getValue(idata) + c1);
    else {
      setValue(idata, 0.0);
      result = false;
    }
  }
  return result;
}

//_______________________________________________________________________________________
bool TRDCalROC::multiply(float c1)
{
  //
  // multiply constant
  //

  bool result = true;
  if (c1 < 0)
    return false;
  for (int idata = 0; idata < mNchannels; idata++) {
    if ((getValue(idata) * c1) <= 6.5535)
      setValue(idata, getValue(idata) * c1);
    else {
      setValue(idata, 0.0);
      result = false;
    }
  }
  return result;
}

//____________________________________________________________________________________________
bool TRDCalROC::add(const TRDCalROC* roc, double c1)
{
  //
  // add values
  //

  bool result = true;
  for (int idata = 0; idata < mNchannels; idata++) {
    if (((getValue(idata) + roc->getValue(idata) * c1) <= 6.5535) &&
        ((getValue(idata) + roc->getValue(idata) * c1) >= 0.0))
      setValue(idata, getValue(idata) + roc->getValue(idata) * c1);
    else {
      setValue(idata, 0.0);
      result = false;
    }
  }
  return result;
}

//____________________________________________________________________________________________
bool TRDCalROC::multiply(const TRDCalROC* roc)
{
  //
  // multiply values - per by pad
  //

  bool result = true;
  for (int idata = 0; idata < mNchannels; idata++) {
    if ((getValue(idata) * roc->getValue(idata)) <= 6.5535)
      setValue(idata, getValue(idata) * roc->getValue(idata));
    else {
      setValue(idata, 0.0);
      result = false;
    }
  }
  return result;
}

//______________________________________________________________________________________________
bool TRDCalROC::divide(const TRDCalROC* roc)
{
  //
  // divide values
  //

  bool result = true;
  float eps = 0.00000000000000001;
  for (int idata = 0; idata < mNchannels; idata++) {
    if (TMath::Abs(roc->getValue(idata)) > eps) {
      if ((getValue(idata) / roc->getValue(idata)) <= 6.5535)
        setValue(idata, getValue(idata) / roc->getValue(idata));
      else {
        setValue(idata, 0.0);
        result = false;
      }
    } else {
      result = false;
    }
  }
  return result;
}
//______________________________________________________________________________________________
bool TRDCalROC::unfold()
{
  //
  // Compute the mean value per pad col
  // Divide with this value each pad col
  // This is for the noise study
  // Return false if one or more of the pad col was not normalised
  //

  bool result = true;
  float eps = 0.00000000000000001;
  double mmeannotnull = getMeanNotNull();
  double rmsnotnull = getRMSNotNull();
  //printf("mmeannotnull %f and rmsnotnull %f\n",mmeannotnull,rmsnotnull);
  if ((mmeannotnull < 0.) || (rmsnotnull < 0.)) {
    return false;
  }
  // calcul the mean value per col
  for (int icol = 0; icol < mNcols; icol++) {
    float mean = 0.0;
    float nb = 0.0;
    for (int irow = 0; irow < mNrows; irow++) {
      if (TMath::Abs(getValue(icol, irow) - mmeannotnull) < 5 * rmsnotnull) {
        mean += getValue(icol, irow);
        nb += 1.0;
      }
    }
    if (nb > eps) {
      mean = mean / nb;
      if (mean > eps) {
        for (int irow = 0; irow < mNrows; irow++) {
          float value = getValue(icol, irow);
          setValue(icol, irow, (float)(value / mean));
        }
      } else {
        result = false;
      }
    } else {
      result = false;
    }
  }
  return result;
}
//__________________________________________________________________________________
TH2F* TRDCalROC::makeHisto2D(float min, float max, int type, float mu)
{
  //
  // make 2D histo
  // type -1 = user defined range
  //       0 = nsigma cut nsigma=min
  //       1 = delta cut around median delta=min

  float epsr = 0.005;
  if (type >= 0) {
    if (type == 0) {
      // nsigma range
      float mean = getMean();
      float sigma = getRMS();
      float nsigma = TMath::Abs(min);
      sigma *= nsigma;
      if (sigma < epsr)
        sigma = epsr;
      min = mean - sigma;
      max = mean + sigma;
    }
    if (type == 1) {
      // fixed range
      float mean = getMedian();
      float delta = min;
      min = mean - delta;
      max = mean + delta;
    }
    if (type == 2) {
      double sigma;
      float mean = getLTM(&sigma, max);
      sigma *= min;
      if (sigma < epsr)
        sigma = epsr;
      min = mean - sigma;
      max = mean + sigma;
    }
  }
  char name[1000];
  // snprintf(name, 1000, "%s 2D Plane %d Chamber %d", GetTitle(), fPla, fCha);
  TH2F* his = new TH2F(name, name, mNrows, 0, mNrows, mNcols, 0, mNcols);
  for (int irow = 0; irow < mNrows; irow++) {
    for (int icol = 0; icol < mNcols; icol++) {
      his->Fill(irow + 0.5, icol + 0.5, getValue(icol, irow) * mu);
    }
  }
  his->SetStats(0);
  his->SetMaximum(max);
  his->SetMinimum(min);
  return his;
}

//_______________________________________________________________________________________
TH1F* TRDCalROC::makeHisto1D(float min, float max, int type, float mu)
{
  //
  // make 1D histo
  // type -1 = user defined range
  //       0 = nsigma cut nsigma=min
  //       1 = delta cut around median delta=min

  float epsr = 0.005;
  if (type >= 0) {
    if (type == 0) {
      // nsigma range
      float mean = getMean();
      float sigma = getRMS();
      float nsigma = TMath::Abs(min);
      sigma *= nsigma;
      if (sigma < epsr)
        sigma = epsr;
      min = mean - sigma;
      max = mean + sigma;
    }
    if (type == 1) {
      // fixed range
      float mean = getMedian();
      float delta = min;
      min = mean - delta;
      max = mean + delta;
    }
    if (type == 2) {
      //
      // LTM mean +- nsigma
      //
      double sigma;
      float mean = getLTM(&sigma, max);
      sigma *= min;
      if (sigma < epsr)
        sigma = epsr;
      min = mean - sigma;
      max = mean + sigma;
    }
  }
  char name[1000];
  // snprintf(name, 1000, "%s 1D Plane %d Chamber %d", GetTitle(), fPla, fCha);
  TH1F* his = new TH1F(name, name, 100, min, max);
  for (int irow = 0; irow < mNrows; irow++) {
    for (int icol = 0; icol < mNcols; icol++) {
      his->Fill(getValue(icol, irow) * mu);
    }
  }
  return his;
}
