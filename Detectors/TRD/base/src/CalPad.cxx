// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TMath.h>
#include <TH2F.h>
#include <TH1F.h>
#include <TStyle.h>

#include "TRDBase/CalPad.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/PadPlane.h"

#include "CalROC.h"
#include "CalDet.h"

using namespace o2::trd;
using namespace o2::trd::constants;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which saved per pad                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
CalPad::CalPad()
{
  //
  // CalPad default constructor
  //

  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    fROC[idet] = nullptr;
  }
}

//_____________________________________________________________________________
CalPad::CalPad(const Text_t* name, const Text_t* title)
{
  //
  // CalPad constructor
  //

  for (int isec = 0; isec < NSECTOR; isec++) {
    for (int ipla = 0; ipla < NLAYER; ipla++) {
      for (int icha = 0; icha < NSTACK; icha++) {
        int idet = getDet(ipla, icha, isec);
        fROC[idet] = new CalROC(ipla, icha);
      }
    }
  }
  mName = name;
  mTitle = title;
}

//_____________________________________________________________________________
CalPad::CalPad(const CalPad& c)
{
  //
  // CalPad copy constructor
  //

  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    fROC[idet] = new CalROC(*((CalPad&)c).fROC[idet]);
  }
  mName = c.mName;
  mTitle = c.mTitle;
}

//_____________________________________________________________________________
CalPad::~CalPad()
{
  //
  // CalPad destructor
  //

  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (fROC[idet]) {
      delete fROC[idet];
      fROC[idet] = 0;
    }
  }
}

//_____________________________________________________________________________
CalPad& CalPad::operator=(const CalPad& c)
{
  //
  // Assignment operator
  //

  if (this != &c)
    ((CalPad&)c).Copy(*this);
  return *this;
}

//_____________________________________________________________________________
void CalPad::Copy(TObject& c) const
{
  //
  // Copy function
  //

  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (((CalPad&)c).fROC[idet]) {
      delete ((CalPad&)c).fROC[idet];
    }
    ((CalPad&)c).fROC[idet] = new CalROC();
    if (fROC[idet]) {
      fROC[idet]->Copy(*((CalPad&)c).fROC[idet]);
    }
  }

  c.mName = mName;
  c.mTitle = mTitle;
}

//_____________________________________________________________________________
bool CalPad::scaleROCs(const CalDet* values)
{
  //
  // Scales ROCs of this class with the values from the class <values>
  // Is used if an CalPad object defines local variations of a parameter
  // defined per detector using a CalDet class
  //

  if (!values)
    return kFALSE;
  bool result = kTRUE;
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (fROC[idet]) {
      if (!fROC[idet]->multiply(values->getValue(idet)))
        result = kFALSE;
    }
  }
  return result;
}

//_____________________________________________________________________________
void CalPad::setCalROC(int det, CalROC* calroc)
{
  //
  // Set the CalROC to this one
  //

  if (!calroc)
    return;
  if (fROC[det]) {
    for (int icol = 0; icol < calroc->getNcols(); icol++) {
      for (int irow = 0; irow < calroc->getNrows(); irow++) {
        fROC[det]->SetValue(icol, irow, calroc->getValue(icol, irow));
      }
    }
  }
}
//_____________________________________________________________________________
double CalPad::getMeanRMS(double& rms, const CalDet* calDet, int type)
{
  //
  // Calculate mean an RMS of all rocs
  // If calDet correct the CalROC from the detector coefficient
  // type == 0 for gain and vdrift
  // type == 1 for t0
  //
  double factor = 0.0;
  if (type == 0)
    factor = 1.0;
  double sum = 0, sum2 = 0, n = 0, val;
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (calDet)
      factor = calDet->getValue(idet);
    CalROC* calRoc = fROC[idet];
    if (calRoc) {
      for (int irow = 0; irow < calRoc->getNrows(); irow++) {
        for (int icol = 0; icol < calRoc->getNcols(); icol++) {
          if (type == 0)
            val = calRoc->getValue(icol, irow) * factor;
          else
            val = calRoc->getValue(icol, irow) + factor;
          sum += val;
          sum2 += val * val;
          n++;
        }
      }
    }
  }
  double n1 = 1. / n;
  double mean = sum * n1;
  rms = TMath::Sqrt(TMath::Abs(sum2 * n1 - mean * mean));
  return mean;
}

//_____________________________________________________________________________
double CalPad::getMean(const CalDet* calDet, int type, CalPad* const outlierPad)
{
  //
  // return mean of the mean of all ROCs
  // If calDet correct the CalROC from the detector coefficient
  // type == 0 for gain and vdrift
  // type == 1 for t0
  //
  double factor = 0.0;
  if (type == 0)
    factor = 1.0;
  double arr[MAXCHAMBER];
  int n = 0;
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (calDet)
      factor = calDet->getValue(idet);
    CalROC* calRoc = fROC[idet];
    if (calRoc) {
      CalROC* outlierROC = 0;
      if (outlierPad)
        outlierROC = outlierPad->getCalROC(idet);
      if (type == 0)
        arr[n] = calRoc->getMean(outlierROC) * factor;
      else
        arr[n] = calRoc->getMean(outlierROC) + factor;
      n++;
    }
  }
  return TMath::Mean(n, arr);
}

//_____________________________________________________________________________
double CalPad::getRMS(const CalDet* calDet, int type, CalPad* const outlierPad)
{
  //
  // return mean of the RMS of all ROCs
  // If calDet correct the CalROC from the detector coefficient
  // type == 0 for gain and vdrift
  // type == 1 for t0
  //
  double factor = 0.0;
  if (type == 0)
    factor = 1.0;
  double arr[MAXCHAMBER];
  int n = 0;
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (calDet)
      factor = calDet->getValue(idet);
    CalROC* calRoc = fROC[idet];
    if (calRoc) {
      CalROC* outlierROC = 0;
      if (outlierPad)
        outlierROC = outlierPad->getCalROC(idet);
      if (type == 0)
        arr[n] = calRoc->getRMS(outlierROC) * factor;
      else
        arr[n] = calRoc->getRMS(outlierROC);
      n++;
    }
  }
  return TMath::Mean(n, arr);
}

//_____________________________________________________________________________
double CalPad::getMedian(const CalDet* calDet, int type, CalPad* const outlierPad)
{
  //
  // return mean of the median of all ROCs
  // If CalDet, the correct the CalROC from the detector coefficient
  // type == 0 for gain and vdrift
  // type == 1 for t0
  //
  double factor = 0.0;
  if (type == 0)
    factor = 1.0;
  double arr[MAXCHAMBER];
  int n = 0;
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (calDet)
      factor = calDet->getValue(idet);
    CalROC* calRoc = fROC[idet];
    if (calRoc) {
      CalROC* outlierROC = 0;
      if (outlierPad)
        outlierROC = outlierPad->getCalROC(idet);
      if (type == 0)
        arr[n] = calRoc->getMedian(outlierROC) * factor;
      else
        arr[n] = calRoc->getMedian(outlierROC) + factor;
      n++;
    }
  }
  return TMath::Mean(n, arr);
}

//_____________________________________________________________________________
double CalPad::getLTM(double* sigma, double fraction, const CalDet* calDet, int type, CalPad* const outlierPad)
{
  //
  // return mean of the LTM and sigma of all ROCs
  // If calDet correct the CalROC from the detector coefficient
  // type == 0 for gain and vdrift
  // type == 1 for t0
  //
  double factor = 0.0;
  if (type == 0)
    factor = 1.0;
  double arrm[MAXCHAMBER];
  double arrs[MAXCHAMBER];
  double* sTemp = 0x0;
  int n = 0;

  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (calDet)
      factor = calDet->getValue(idet);
    CalROC* calRoc = fROC[idet];
    if (calRoc) {
      if (sigma)
        sTemp = arrs + n;
      CalROC* outlierROC = 0;
      if (outlierPad)
        outlierROC = outlierPad->getCalROC(idet);
      if (type == 0)
        arrm[n] = calRoc->getLTM(sTemp, fraction, outlierROC) * factor;
      else
        arrm[n] = calRoc->getLTM(sTemp, fraction, outlierROC) + factor;
      n++;
    }
  }
  if (sigma)
    *sigma = TMath::Mean(n, arrs);
  return TMath::Mean(n, arrm);
}

//_____________________________________________________________________________
TH1F* CalPad::makeHisto1D(const CalDet* calDet, int typedet, float min, float max, int type)
{
  //
  // make 1D histo
  // type -1 = user defined range
  //       0 = nsigma cut nsigma=min
  // If calDet correct the CalROC from the detector coefficient
  // typedet == 0 for gain and vdrift
  // typedet == 1 for t0
  //

  double factor = 0.0;
  if (typedet == 0)
    factor = 1.0;

  if (type >= 0) {
    if (type == 0) {
      // nsigma range
      float mean = getMean(calDet, typedet);
      float sigma = 0.0;
      float kEpsilonr = 0.005;
      if (getRMS(calDet, typedet) > kEpsilonr)
        sigma = getRMS(calDet, typedet);
      else {
        double rms = 0.0;
        sigma = getMeanRMS(rms, calDet, typedet);
      }
      float nsigma = TMath::Abs(min);
      sigma *= nsigma;
      if (sigma < kEpsilonr)
        sigma = kEpsilonr;
      min = mean - sigma;
      max = mean + sigma;
    }
    if (type == 1) {
      // fixed range
      float mean = getMedian(calDet, typedet);
      float delta = min;
      min = mean - delta;
      max = mean + delta;
    }
    if (type == 2) {
      //
      // LTM mean +- nsigma
      //
      double sigma;
      float mean = getLTM(&sigma, max, calDet, typedet);
      sigma *= min;
      float kEpsilonr = 0.005;
      if (sigma < kEpsilonr)
        sigma = kEpsilonr;
      min = mean - sigma;
      max = mean + sigma;
    }
  }
  char name[1000];
  snprintf(name, 1000, "%s Pad 1D", getTitle());
  TH1F* his = new TH1F(name, name, 100, min, max);
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (calDet)
      factor = calDet->getValue(idet);
    if (fROC[idet]) {
      for (int irow = 0; irow < fROC[idet]->getNrows(); irow++) {
        for (int icol = 0; icol < fROC[idet]->getNcols(); icol++) {
          if (typedet == 0)
            his->Fill(fROC[idet]->getValue(irow, icol) * factor);
          else
            his->Fill(fROC[idet]->getValue(irow, icol) + factor);
        }
      }
    }
  }
  return his;
}

//_____________________________________________________________________________
TH2F* CalPad::makeHisto2DSmPl(int sm, int pl, const CalDet* calDet, int typedet, float min, float max, int type)
{
  //
  // Make 2D graph
  // sm    - supermodule number
  // pl    - plane number
  // If calDet correct the CalROC from the detector coefficient
  // typedet == 0 for gain and vdrift
  // typedet == 1 for t0
  //
  gStyle->SetPalette(1);
  double factor = 0.0;
  if (typedet == 0)
    factor = 1.0;

  float kEpsilon = 0.000000000001;

  Geometry* trdGeo = new Geometry();

  if (type >= 0) {
    float kEpsilonr = 0.005;
    if (type == 0) {
      // nsigma range
      float mean = getMean(calDet, typedet);
      float sigma = 0.0;
      if (getRMS(calDet, typedet) > kEpsilonr)
        sigma = getRMS(calDet, typedet);
      else {
        double rms = 0.0;
        sigma = getMeanRMS(rms, calDet, typedet);
      }
      float nsigma = TMath::Abs(min);
      sigma *= nsigma;
      if (sigma < kEpsilonr)
        sigma = kEpsilonr;
      min = mean - sigma;
      max = mean + sigma;
    }
    if (type == 1) {
      // fixed range
      float mean = getMedian(calDet, typedet);
      float delta = min;
      min = mean - delta;
      max = mean + delta;
    }
    if (type == 2) {
      //
      // LTM mean +- nsigma
      //
      double sigma;
      float mean = getLTM(&sigma, max, calDet, typedet);
      sigma *= min;
      if (sigma < kEpsilonr)
        sigma = kEpsilonr;
      min = mean - sigma;
      max = mean + sigma;
    }
  }

  AliTRDpadPlane* padPlane0 = trdGeo->getPadPlane(pl, 0);
  double row0 = padPlane0->getRow0();
  double col0 = padPlane0->getCol0();

  char name[1000];
  snprintf(name, 1000, "%s Pad 2D sm %d pl %d", getTitle(), sm, pl);
  TH2F* his = new TH2F(name, name, 76, -TMath::Abs(row0), TMath::Abs(row0), 144, -TMath::Abs(col0), TMath::Abs(col0));

  // Where we begin
  int offsetsmpl = 30 * sm + pl;

  for (int k = 0; k < NSTACK; k++) {
    int det = offsetsmpl + k * 6;
    if (calDet)
      factor = calDet->getValue(det);
    if (fROC[det]) {
      CalROC* calRoc = fROC[det];
      for (int irow = 0; irow < calRoc->getNrows(); irow++) {
        for (int icol = 0; icol < calRoc->getNcols(); icol++) {
          if (TMath::Abs(calRoc->getValue(icol, irow)) > kEpsilon) {
            int binz = 0;
            int kb = NSTACK - 1 - k;
            int krow = calRoc->getNrows() - 1 - irow;
            int kcol = calRoc->getNcols() - 1 - icol;
            if (kb > 2)
              binz = 16 * (kb - 1) + 12 + krow + 1;
            else
              binz = 16 * kb + krow + 1;
            int biny = kcol + 1;
            float value = calRoc->getValue(icol, irow);
            if (typedet == 0)
              his->SetBinContent(binz, biny, value * factor);
            else
              his->SetBinContent(binz, biny, value + factor);
          }
        }
      }
    }
  }
  his->SetXTitle("z (cm)");
  his->SetYTitle("y (cm)");
  his->SetStats(0);
  his->SetMaximum(max);
  his->SetMinimum(min);
  delete trdGeo;
  return his;
}

//_____________________________________________________________________________
TH2F* CalPad::makeHisto2DCh(int ch, const CalDet* calDet, int typedet, float min, float max, int type)
{
  //
  // Make 2D graph mean value in z direction
  // ch    - chamber
  // If calDet correct the CalROC from the detector coefficient
  // typedet == 0 for gain and vdrift
  // typedet == 1 for t0
  //
  gStyle->SetPalette(1);
  double factor = 0.0;
  if (typedet == 0)
    factor = 1.0;

  if (type >= 0) {
    float kEpsilonr = 0.005;
    if (type == 0) {
      // nsigma range
      float mean = getMean(calDet, typedet);
      float sigma = 0.0;
      if (getRMS(calDet, typedet) > kEpsilonr)
        sigma = getRMS(calDet, typedet);
      else {
        double rms = 0.0;
        sigma = getMeanRMS(rms, calDet, typedet);
      }
      float nsigma = TMath::Abs(min);
      sigma *= nsigma;
      if (sigma < kEpsilonr)
        sigma = kEpsilonr;
      min = mean - sigma;
      max = mean + sigma;
    }
    if (type == 1) {
      // fixed range
      float mean = getMedian(calDet, typedet);
      float delta = min;
      min = mean - delta;
      max = mean + delta;
    }
    if (type == 2) {
      //
      // LTM mean +- nsigma
      //
      double sigma;
      float mean = getLTM(&sigma, max, calDet, typedet);
      sigma *= min;
      if (sigma < kEpsilonr)
        sigma = kEpsilonr;
      min = mean - sigma;
      max = mean + sigma;
    }
  }

  Geometry* trdGeo = new Geometry();

  float kEpsilon = 0.000000000001;

  double poslocal[3] = {0.0, 0.0, 0.0};
  double posglobal[3] = {0.0, 0.0, 0.0};

  std::string name;
  name << mTitle << " Pad 2D ch " << ch;
  TH2F* his = new TH2F(name.c_str(), name.c_str(), 400, -400.0, 400.0, 400, -400.0, 400.0);

  // Where we begin
  int offsetch = 6 * ch;

  for (int isec = 0; isec < NSECTOR; isec++) {
    for (int ipl = 0; ipl < NLAYER; ipl++) {
      int det = offsetch + isec * 30 + ipl;
      if (calDet)
        factor = calDet->getValue(det);
      if (fROC[det]) {
        CalROC* calRoc = fROC[det];
        PadPlane* padPlane = trdGeo->getPadPlane(ipl, ch);
        for (int icol = 0; icol < calRoc->getNcols(); icol++) {
          poslocal[0] = trdGeo->getTime0(ipl);
          poslocal[2] = padPlane->getRowPos(0);
          poslocal[1] = padPlane->getColPos(icol);
          trdGeo->rotateBack(det, poslocal, posglobal);
          int binx = 1 + TMath::Nint((posglobal[0] + 400.0) * 0.5);
          int biny = 1 + TMath::Nint((posglobal[1] + 400.0) * 0.5);
          float value = 0.0;
          int nb = 0;
          for (int irow = 0; irow < calRoc->getNrows(); irow++) {
            if (TMath::Abs(calRoc->getValue(icol, irow)) > kEpsilon) {
              value += calRoc->getValue(icol, irow);
              nb++;
            }
          }
          if (nb > 0) {
            value = value / nb;
          }
          if (typedet == 0)
            his->SetBinContent(binx, biny, value * factor);
          else
            his->SetBinContent(binx, biny, value + factor);
        }
      }
    }
  }
  his->SetXTitle("x (cm)");
  his->SetYTitle("y (cm)");
  his->SetStats(0);
  his->SetMaximum(max);
  his->SetMinimum(min);
  delete trdGeo;
  return his;
}

//_____________________________________________________________________________
bool CalPad::add(float c1)
{
  //
  // add constant for all channels of all ROCs
  //

  bool result = kTRUE;
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (fROC[idet]) {
      if (!fROC[idet]->add(c1))
        result = kFALSE;
    }
  }
  return result;
}

//_____________________________________________________________________________
bool CalPad::multiply(float c1)
{
  //
  // multiply constant for all channels of all ROCs
  //
  bool result = kTRUE;
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (fROC[idet]) {
      if (!fROC[idet]->multiply(c1))
        result = kFALSE;
    }
  }
  return result;
}

//_____________________________________________________________________________
bool CalPad::add(const CalPad* pad, double c1, const CalDet* calDet1, const CalDet* calDet2, int type)
{
  //
  // add calpad channel by channel multiplied by c1 - all ROCs
  // If calDet1 and calDet2, the correct the CalROC from the detector coefficient
  // then you have calDet1 and the calPad together
  // calDet2 and pad together
  // typedet == 0 for gain and vdrift
  // typedet == 1 for t0
  //
  float kEpsilon = 0.000000000001;

  double factor1 = 0.0;
  double factor2 = 0.0;
  if (type == 0) {
    factor1 = 1.0;
    factor2 = 1.0;
  }
  bool result = kTRUE;
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (calDet1)
      factor1 = calDet1->getValue(idet);
    if (calDet2)
      factor2 = calDet2->getValue(idet);
    if (fROC[idet]) {
      if (type == 0) {
        if (TMath::Abs(factor1) > kEpsilon) {
          if (!fROC[idet]->add(pad->getCalROC(idet), c1 * factor2 / factor1))
            result = kFALSE;
        } else
          result = kFALSE;
      } else {
        CalROC* croc = new CalROC((const CalROC)*pad->getCalROC(idet));
        if (!croc->add(factor2))
          result = kFALSE;
        if (!fROC[idet]->add(croc, c1))
          result = kFALSE;
      }
    }
  }
  return result;
}

//_____________________________________________________________________________
bool CalPad::multiply(const CalPad* pad, const CalDet* calDet1, const CalDet* calDet2, int type)
{
  //
  // multiply calpad channel by channel - all ROCs
  // If calDet1 and calDet2, the correct the CalROC from the detector coefficient
  // then you have calDet1 and the calPad together
  // calDet2 and pad together
  // typedet == 0 for gain and vdrift
  // typedet == 1 for t0
  //
  float kEpsilon = 0.000000000001;
  bool result = kTRUE;
  double factor1 = 0.0;
  double factor2 = 0.0;
  if (type == 0) {
    factor1 = 1.0;
    factor2 = 1.0;
  }
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (calDet1)
      factor1 = calDet1->getValue(idet);
    if (calDet2)
      factor2 = calDet2->getValue(idet);
    if (fROC[idet]) {
      if (type == 0) {
        if (TMath::Abs(factor1) > kEpsilon) {
          CalROC* croc = new CalROC((const CalROC)*pad->getCalROC(idet));
          if (!croc->multiply(factor2))
            result = kFALSE;
          if (!fROC[idet]->multiply(croc))
            result = kFALSE;
        } else
          result = kFALSE;
      } else {
        CalROC* croc2 = new CalROC((const CalROC)*pad->getCalROC(idet));
        if (!croc2->add(factor2))
          result = kFALSE;
        if (!fROC[idet]->add(factor1))
          result = kFALSE;
        if (!fROC[idet]->multiply(croc2))
          result = kFALSE;
        if (!fROC[idet]->add(-factor1))
          result = kFALSE;
      }
    }
  }
  return result;
}

//_____________________________________________________________________________
bool CalPad::divide(const CalPad* pad, const CalDet* calDet1, const CalDet* calDet2, int type)
{
  //
  // divide calpad channel by channel - all ROCs
  // If calDet1 and calDet2, the correct the CalROC from the detector coefficient
  // then you have calDet1 and the calPad together
  // calDet2 and pad together
  // typedet == 0 for gain and vdrift
  // typedet == 1 for t0
  //
  float kEpsilon = 0.000000000001;
  bool result = kTRUE;
  double factor1 = 0.0;
  double factor2 = 0.0;
  if (type == 0) {
    factor1 = 1.0;
    factor2 = 1.0;
  }
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (calDet1)
      factor1 = calDet1->getValue(idet);
    if (calDet2)
      factor2 = calDet2->getValue(idet);
    if (fROC[idet]) {
      if (type == 0) {
        if (TMath::Abs(factor1) > kEpsilon) {
          CalROC* croc = new CalROC((const CalROC)*pad->getCalROC(idet));
          if (!croc->multiply(factor2))
            result = kFALSE;
          if (!fROC[idet]->divide(croc))
            result = kFALSE;
        } else
          result = kFALSE;
      } else {
        CalROC* croc2 = new CalROC((const CalROC)*pad->getCalROC(idet));
        if (!croc2->add(factor2))
          result = kFALSE;
        if (!fROC[idet]->add(factor1))
          result = kFALSE;
        if (!fROC[idet]->divide(croc2))
          result = kFALSE;
        if (!fROC[idet]->add(-factor1))
          result = kFALSE;
      }
    }
  }
  return result;
}
