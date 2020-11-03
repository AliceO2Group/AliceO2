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
//  TRD calibration class for parameters which saved per detector            //
//  2019 - Ported from AliRoot to O2 (J. Lopez)                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TMath.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TRobustEstimator.h>

#include "TRDBase/PadPlane.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/CalDet.h"
// #include "AliMathBase.h"

using namespace o2::trd;
using namespace o2::trd::constants;

//___________________________________________________________________________________
double CalDet::getMean(CalDet* const outlierDet) const
{
  //
  // Calculate the mean
  //

  if (!outlierDet) {
    return TMath::Mean(MAXCHAMBER, mData.data());
  }
  std::array<double, MAXCHAMBER> ddata;
  int nPoints = 0;
  for (int i = 0; i < MAXCHAMBER; i++) {
    if (!(outlierDet->getValue(i))) {
      ddata[nPoints] = mData[nPoints];
      nPoints++;
    }
  }
  return TMath::Mean(nPoints, ddata.data());
}

//_______________________________________________________________________________________
double CalDet::getMedian(CalDet* const outlierDet) const
{
  //
  // Calculate the median
  //

  if (!outlierDet) {
    return (double)TMath::Median(MAXCHAMBER, mData.data());
  }
  std::array<double, MAXCHAMBER> ddata;
  int nPoints = 0;
  for (int i = 0; i < MAXCHAMBER; i++) {
    if (!(outlierDet->getValue(i))) {
      ddata[nPoints] = mData[nPoints];
      nPoints++;
    }
  }
  return TMath::Median(nPoints, ddata.data());
}

//____________________________________________________________________________________________
double CalDet::getRMS(CalDet* const outlierDet) const
{
  //
  // Calculate the RMS
  //

  if (!outlierDet) {
    return TMath::RMS(MAXCHAMBER, mData.data());
  }
  std::array<double, MAXCHAMBER> ddata;
  int nPoints = 0;
  for (int i = 0; i < MAXCHAMBER; i++) {
    if (!(outlierDet->getValue(i))) {
      ddata[nPoints] = mData[nPoints];
      nPoints++;
    }
  }
  return TMath::RMS(nPoints, ddata.data());
}

//____________________________________________________________________________________________
double CalDet::getRMSRobust(double robust) const
{
  //
  // Calculate the robust RMS
  //

  // sorted
  std::array<int, MAXCHAMBER> index;
  TMath::Sort((int)MAXCHAMBER, mData.data(), index.data());

  // reject
  double reject = (int)(MAXCHAMBER * (1 - robust) / 2.0);
  if (reject <= 0) {
    reject = 0;
  }
  if (reject >= MAXCHAMBER) {
    reject = 0;
  }

  std::array<double, MAXCHAMBER> ddata;
  int nPoints = 0;
  for (int i = 0; i < MAXCHAMBER; i++) {
    bool rej = kFALSE;
    for (int k = 0; k < reject; k++) {
      if (i == index[k]) {
        rej = kTRUE;
      }
      if (i == index[MAXCHAMBER - (k + 1)]) {
        rej = kTRUE;
      }
    }
    if (!rej) {
      ddata[nPoints] = mData[i];
      nPoints++;
    }
  }
  return TMath::RMS(nPoints, ddata.data());
}

//____________________________________________________________________________________________
double CalDet::getMeanRobust(double robust) const
{
  //
  // Calculate the robust mean
  //

  // sorted
  std::array<int, MAXCHAMBER> index;
  TMath::Sort((int)MAXCHAMBER, mData.data(), index.data());

  // reject
  double reject = (int)(MAXCHAMBER * (1 - robust) / 2.0);
  if (reject <= 0) {
    reject = 0;
  }
  if (reject >= MAXCHAMBER) {
    reject = 0;
  }

  std::array<double, MAXCHAMBER> ddata;
  int nPoints = 0;
  for (int i = 0; i < MAXCHAMBER; i++) {
    bool rej = kFALSE;
    for (int k = 0; k < reject; k++) {
      if (i == index[k]) {
        rej = kTRUE;
      }
      if (i == index[MAXCHAMBER - (k + 1)]) {
        rej = kTRUE;
      }
    }
    if (!rej) {
      ddata[nPoints] = mData[i];
      nPoints++;
    }
  }
  return TMath::Mean(nPoints, ddata.data());
}

//______________________________________________________________________________________________
double CalDet::getLTM(double* sigma, double fraction, CalDet* const outlierDet)
{
  //
  //  Calculate LTM mean and sigma
  //

  std::array<double, MAXCHAMBER> ddata;
  double mean = 0, lsigma = 0;
  unsigned int nPoints = 0;
  for (int i = 0; i < MAXCHAMBER; i++) {
    if (!outlierDet || !(outlierDet->getValue(i))) {
      ddata[nPoints] = mData[nPoints];
      nPoints++;
    }
  }
  int hh = TMath::Min(TMath::Nint(fraction * nPoints), int(nPoints));
  // AliMathBase::EvaluateUni(nPoints, ddata, mean, lsigma, hh);
  TRobustEstimator tre;
  tre.EvaluateUni(nPoints, ddata.data(), mean, lsigma, hh);

  if (sigma) {
    *sigma = lsigma;
  }
  return mean;
}

//_________________________________________________________________________
TH1F* CalDet::makeHisto1Distribution(float min, float max, int type)
{
  //
  // make 1D histo
  // type -1 = user defined range
  //       0 = nsigma cut nsigma=min
  //       1 = delta cut around median delta=min
  //

  if (type >= 0) {
    if (type == 0) {
      // nsigma range
      float mean = getMean();
      float sigma = getRMS();
      float nsigma = TMath::Abs(min);
      min = mean - nsigma * sigma;
      max = mean + nsigma * sigma;
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
      min = mean - sigma;
      max = mean + sigma;
    }
  }
  std::stringstream title;
  title << mTitle.c_str() << " CalDet 1Distribution";
  std::string titlestr = title.str();
  TH1F* his = new TH1F(mName.c_str(), titlestr.c_str(), 100, min, max);
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    his->Fill(getValue(idet));
  }
  return his;
}

//________________________________________________________________________________
TH1F* CalDet::makeHisto1DAsFunctionOfDet(float min, float max, int type)
{
  //
  // make 1D histo
  // type -1 = user defined range
  //       0 = nsigma cut nsigma=min
  //       1 = delta cut around median delta=min
  //

  if (type >= 0) {
    if (type == 0) {
      // nsigma range
      float mean = getMean();
      float sigma = getRMS();
      float nsigma = TMath::Abs(min);
      min = mean - nsigma * sigma;
      max = mean + nsigma * sigma;
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
      min = mean - sigma;
      max = mean + sigma;
    }
  }

  std::stringstream title;
  title << mTitle.c_str() << " CalDet as function of det";
  std::string titlestr = title.str();
  TH1F* his = new TH1F(mName.c_str(), titlestr.c_str(), MAXCHAMBER, 0, MAXCHAMBER);
  for (int det = 0; det < MAXCHAMBER; det++) {
    his->Fill(det + 0.5, getValue(det));
  }
  his->SetMaximum(max);
  his->SetMinimum(min);
  return his;
}

//_____________________________________________________________________________
TH2F* CalDet::makeHisto2DCh(int ch, float min, float max, int type)
{
  //
  // Make 2D graph
  // ch    - chamber
  // type -1 = user defined range
  //       0 = nsigma cut nsigma=min
  //       1 = delta cut around median delta=min
  //

  if (type >= 0) {
    if (type == 0) {
      // nsigma range
      float mean = getMean();
      float sigma = getRMS();
      float nsigma = TMath::Abs(min);
      min = mean - nsigma * sigma;
      max = mean + nsigma * sigma;
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
      min = mean - sigma;
      max = mean + sigma;
    }
  }

  Geometry* trdGeo = Geometry::instance();

  float poslocal[3] = {0, 0, 0};
  float posglobal[3] = {0, 0, 0};

  std::stringstream title;
  title << mTitle.c_str() << " Cal AADet 2D ch " << ch;
  std::string titlestr = title.str();
  TH2F* his = new TH2F(mName.c_str(), titlestr.c_str(), 400, -400.0, 400.0, 400, -400.0, 400.0);

  // Where we begin
  int offsetch = 6 * ch;

  for (int isec = 0; isec < NSECTOR; isec++) {
    for (int ipl = 0; ipl < NLAYER; ipl++) {
      int det = offsetch + isec * 30 + ipl;
      const PadPlane* padPlane = trdGeo->getPadPlane(ipl, ch);
      for (int icol = 0; icol < padPlane->getNcols(); icol++) {
        poslocal[0] = trdGeo->getTime0(ipl);
        poslocal[2] = padPlane->getRowPos(0);
        poslocal[1] = padPlane->getColPos(icol);
        trdGeo->rotateBack(det, poslocal, posglobal);
        int binx = 1 + TMath::Nint((posglobal[0] + 400.0) * 0.5);
        int biny = 1 + TMath::Nint((posglobal[1] + 400.0) * 0.5);
        his->SetBinContent(binx, biny, mData[det]);
      }
    }
  }
  his->SetXTitle("x (cm)");
  his->SetYTitle("y (cm)");
  his->SetStats(0);
  his->SetMaximum(max);
  his->SetMinimum(min);
  return his;
}

//_____________________________________________________________________________
TH2F* CalDet::makeHisto2DSmPl(int sm, int pl, float min, float max, int type)
{
  //
  // Make 2D graph
  // sm    - supermodule number
  // pl    - plane number
  // type -1 = user defined range
  //       0 = nsigma cut nsigma=min
  //       1 = delta cut around median delta=min
  //

  if (type >= 0) {
    if (type == 0) {
      // nsigma range
      float mean = getMean();
      float sigma = getRMS();
      float nsigma = TMath::Abs(min);
      min = mean - nsigma * sigma;
      max = mean + nsigma * sigma;
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
      min = mean - sigma;
      max = mean + sigma;
    }
  }

  Geometry* trdGeo = Geometry::instance();
  const PadPlane* padPlane0 = trdGeo->getPadPlane(pl, 0);
  double row0 = padPlane0->getRow0();
  double col0 = padPlane0->getCol0();

  std::stringstream title;
  title << mTitle.c_str() << " CalDet 2D sm " << sm << " and pl " << pl;
  TH2F* his = new TH2F(mName.c_str(), title.str().c_str(), 5, -TMath::Abs(row0), TMath::Abs(row0), 4, -2 * TMath::Abs(col0), 2 * TMath::Abs(col0));

  // Where we begin
  int offsetsmpl = 30 * sm + pl;

  for (int k = 0; k < NSTACK; k++) {
    int det = offsetsmpl + k * 6;
    int kb = NSTACK - 1 - k;
    his->SetBinContent(kb + 1, 2, mData[det]);
    his->SetBinContent(kb + 1, 3, mData[det]);
  }
  his->SetXTitle("z (cm)");
  his->SetYTitle("y (cm)");
  his->SetStats(0);
  his->SetMaximum(max);
  his->SetMinimum(min);
  return his;
}

//_____________________________________________________________________________
void CalDet::add(float c1)
{
  //
  // Add constant for all detectors
  //

  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    mData[idet] += c1;
  }
}

//_____________________________________________________________________________
void CalDet::multiply(float c1)
{
  //
  // multiply constant for all detectors
  //

  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    mData[idet] *= c1;
  }
}

//_____________________________________________________________________________
void CalDet::add(const CalDet* calDet, double c1)
{
  //
  // add caldet channel by channel multiplied by c1
  //

  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    mData[idet] += calDet->getValue(idet) * c1;
  }
}

//_____________________________________________________________________________
void CalDet::multiply(const CalDet* calDet)
{
  //
  // multiply caldet channel by channel
  //

  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    mData[idet] *= calDet->getValue(idet);
  }
}

//_____________________________________________________________________________
void CalDet::divide(const CalDet* calDet)
{
  //
  // divide caldet channel by channel
  //

  float eps = 0.00000000000000001;
  for (int idet = 0; idet < MAXCHAMBER; idet++) {
    if (TMath::Abs(calDet->getValue(idet)) > eps) {
      mData[idet] /= calDet->getValue(idet);
    }
  }
}

//_____________________________________________________________________________
double CalDet::calcMean(bool wghtPads, int& calib)
{
  // Calculate the mean value after rejection of the chambers not calibrated
  // wghPads = kTRUE weighted with the number of pads in case of a AliTRDCalPad created (t0)
  // calib = number of used chambers for the mean calculation

  double sum = 0.0;
  int ndet = 0;
  double meanALL = 0.0;
  double meanWP = 0.0;
  double padsALL = (144 * 16 * 24 + 144 * 12 * 6) * 18;
  std::array<double, 18> meanSM{};
  std::array<double, 18> meanSMWP{};

  int det = 0;
  while (det < 540) {
    float val = mData[det];
    int iSM = (int)(det / (6 * 5));
    double pads = (((int)(det % (6 * 5)) / 6) == 2) ? 144 * 12 : 144 * 16;
    meanALL += val / 540.;
    meanSM[iSM] += val / 30.;
    meanWP += val * (pads / padsALL);
    meanSMWP[iSM] += val * (pads / (padsALL / 18.));
    det++;
  }

  det = 0;
  while (det < 540) {
    float val = mData[det];
    if (((!wghtPads) &&
         (TMath::Abs(val - meanALL) > 0.0001) &&
         (TMath::Abs(val - meanSM[(int)(det / (6 * 5))]) > 0.0001)) ||
        ((wghtPads) &&
         (TMath::Abs(val - meanWP) > 0.0001) &&
         (TMath::Abs(val - meanSMWP[(int)(det / (6 * 5))]) > 0.0001))) {
      if (val <= 50.) { // get rid of exb alternative mean values
        sum += val;
        ndet++;
      }
    }
    det++;
  }
  calib = ndet;
  return (sum != 0.0 ? sum / ndet : -1);
}

//_____________________________________________________________________________
double CalDet::calcMean(bool wghtPads)
{
  // Calculate the mean value after rejection of the chambers not calibrated
  // wghPads = kTRUE weighted with the number of pads in case of a AliTRDCalPad created (t0)
  // calib = number of used chambers for the mean calculation

  int calib = 0;
  return calcMean(wghtPads, calib);
}
//_____________________________________________________________________________
double CalDet::calcRMS(bool wghtPads, int& calib)
{
  // Calculate the RMS value after rejection of the chambers not calibrated
  // wghPads = kTRUE weighted with the number of pads in case of a AliTRDCalPad created (t0)
  // calib = number of used chambers for the mean calculation

  double sum = 0.0;
  int ndet = 0;
  double meanALL = 0.0;
  double meanWP = 0.0;
  double padsALL = (144 * 16 * 24 + 144 * 12 * 6) * 18;
  std::array<double, 18> meanSM{};
  std::array<double, 18> meanSMWP{};

  int det = 0;
  while (det < 540) {
    double pads = 0.0;
    float val = mData[det];
    int iSM = (int)(det / (6 * 5));
    pads = (((int)(det % (6 * 5)) / 6) == 2) ? 144 * 12 : 144 * 16;
    meanALL += val / 540.;
    meanSM[iSM] += val / 30.;
    meanWP += val * (pads / padsALL);
    meanSMWP[iSM] += val * (pads / (padsALL / 18.));
    det++;
  }

  double mean = 0.0;
  if (!wghtPads) {
    mean = meanALL;
  }
  if (wghtPads) {
    mean = meanWP;
  }

  det = 0;
  while (det < 540) {
    float val = mData[det];
    if (((!wghtPads) &&
         (TMath::Abs(val - meanALL) > 0.0001) &&
         (TMath::Abs(val - meanSM[(int)(det / (6 * 5))]) > 0.0001)) ||
        ((wghtPads) &&
         (TMath::Abs(val - meanWP) > 0.0001) &&
         (TMath::Abs(val - meanSMWP[(int)(det / (6 * 5))]) > 0.0001))) {
      if (val <= 50.) { // get rid of exb alternative mean values
        sum += (val - mean) * (val - mean);
        ndet++;
      }
    }
    det++;
  }

  calib = ndet;
  return (sum != 0.0 ? TMath::Sqrt(sum / ndet) : -1);
}
//_____________________________________________________________________________
double CalDet::calcRMS(bool wghtPads)
{
  // Calculate the RMS value after rejection of the chambers not calibrated
  // wghPads = kTRUE weighted with the number of pads in case of a AliTRDCalPad created (t0)
  // calib = number of used chambers for the mean calculation

  int calib = 0;
  return calcRMS(wghtPads, calib);
}
//_____________________________________________________________________________
double CalDet::getMeanSM(bool wghtPads, int sector) const
{
  // Calculate the mean value for given sector
  // wghPads = kTRUE weighted with the number of pads in case of a AliTRDCalPad created (t0)

  double meanALL = 0.0;
  double meanWP = 0.0;
  double padsALL = (144 * 16 * 24 + 144 * 12 * 6) * 18;
  std::array<double, 18> meanSM{};
  std::array<double, 18> meanSMWP{};

  int det = 0;
  while (det < 540) {
    float val = mData[det];
    int iSM = (int)(det / (6 * 5));
    double pads = (((int)(det % (6 * 5)) / 6) == 2) ? 144 * 12 : 144 * 16;
    meanALL += val / 540.;
    meanSM[iSM] += val / 30.;
    meanWP += val * (pads / padsALL);
    meanSMWP[iSM] += val * (pads / (padsALL / 18.));
    det++;
  }

  double mean = 0.0;
  if (!wghtPads) {
    mean = meanSM[sector];
  }
  if (wghtPads) {
    mean = meanSMWP[sector];
  }

  return mean;
}
