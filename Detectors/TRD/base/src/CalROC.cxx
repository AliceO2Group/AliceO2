// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
#include <sstream>

#include "TRDBase/CalROC.h"

using namespace o2::trd;

//_____________________________________________________________________________
CalROC::CalROC(int p, int c)
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
  if (mData.size() != mNchannels) {
    mData.resize(mNchannels);
    memset(&mData[0], 0, mData.size() * mNchannels);
  }
}

//___________________________________________________________________________________
double CalROC::getMean(CalROC* const outlierROC) const
{
  //
  // Calculate the mean
  //

  std::vector<double> ddata(mNchannels);
  int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if ((!outlierROC) || (!(outlierROC->getValue(i)))) {
      //if(mData[i] > 0.000000000000001){
      ddata[nPoints] = (double)mData[i] / 10000;
      nPoints++;
      //}
    }
  }
  double mean = TMath::Mean(nPoints, ddata.data());
  return mean;
}
//___________________________________________________________________________________
double CalROC::getMeanNotNull() const
{
  //
  // Calculate the mean rejecting null value
  //

  std::vector<double> ddata(mNchannels);
  int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if (mData[i] > 0.000000000000001) {
      ddata[nPoints] = (double)mData[i] / 10000;
      nPoints++;
    }
  }
  if (nPoints < 1) {
    return -1;
  }
  double mean = TMath::Mean(nPoints, ddata.data());
  return mean;
}

//_______________________________________________________________________________________
double CalROC::getMedian(CalROC* const outlierROC) const
{
  //
  // Calculate the median
  //

  std::vector<double> ddata(mNchannels);
  int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if ((!outlierROC) || (!(outlierROC->getValue(i)))) {
      if (mData[i] > 0.000000000000001) {
        ddata[nPoints] = (double)mData[i] / 10000;
        nPoints++;
      }
    }
  }
  double mean = TMath::Median(nPoints, ddata.data());
  return mean;
}

//____________________________________________________________________________________________
double CalROC::getRMS(CalROC* const outlierROC) const
{
  //
  // Calculate the RMS
  //

  std::vector<double> ddata(mNchannels);
  int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if ((!outlierROC) || (!(outlierROC->getValue(i)))) {
      //if(mData[i] > 0.000000000000001){
      ddata[nPoints] = (double)mData[i] / 10000;
      nPoints++;
      //}
    }
  }
  double mean = TMath::RMS(nPoints, ddata.data());
  return mean;
}

//____________________________________________________________________________________________
double CalROC::getRMSNotNull() const
{
  //
  // Calculate the RMS
  //

  std::vector<double> ddata(mNchannels);
  int nPoints = 0;
  for (int i = 0; i < mNchannels; i++) {
    if (mData[i] > 0.000000000000001) {
      ddata[nPoints] = (double)mData[i] / 10000;
      nPoints++;
    }
  }
  if (nPoints < 1) {
    return -1;
  }
  double mean = TMath::RMS(nPoints, ddata.data());
  return mean;
}
//______________________________________________________________________________________________
double CalROC::getLTM(double* sigma, double fraction, CalROC* const outlierROC)
{
  //
  //  Calculate LTM mean and sigma
  //

  std::vector<double> ddata(mNchannels);
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
  tre.EvaluateUni(nPoints, ddata.data(), mean, lsigma, hh);
  if (sigma) {
    *sigma = lsigma;
  }
  return mean;
}

//___________________________________________________________________________________
bool CalROC::add(float c1)
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
bool CalROC::multiply(float c1)
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
bool CalROC::add(const CalROC* roc, double c1)
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
bool CalROC::multiply(const CalROC* roc)
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
bool CalROC::divide(const CalROC* roc)
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
bool CalROC::unfold()
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
TH2F* CalROC::makeHisto2D(float min, float max, int type, float mu)
{
  //
  // make 2D histo
  // type -1 = user defined range
  //       0 = nsigma cut nsigma=min
  //       1 = delta cut around median delta=min

  if (type >= 0) {
    float epsr = 0.005;
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
  std::stringstream title;
  title << mTitle.c_str() << " 2D Plane " << mPla << " Chamber " << mCha;
  std::string titlestr = title.str();
  TH2F* his = new TH2F(mName.c_str(), titlestr.c_str(), mNrows, 0, mNrows, mNcols, 0, mNcols);
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
TH1F* CalROC::makeHisto1D(float min, float max, int type, float mu)
{
  //
  // make 1D histo
  // type -1 = user defined range
  //       0 = nsigma cut nsigma=min
  //       1 = delta cut around median delta=min

  if (type >= 0) {
    float epsr = 0.005;
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
  std::stringstream title;
  title << mTitle.c_str() << " 1D Plane " << mPla << " Chamber " << mCha;
  std::string titlestr = title.str();
  TH1F* his = new TH1F(mName.c_str(), titlestr.c_str(), 100, min, max);
  for (int irow = 0; irow < mNrows; irow++) {
    for (int icol = 0; icol < mNcols; icol++) {
      his->Fill(getValue(icol, irow) * mu);
    }
  }
  return his;
}
