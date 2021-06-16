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

#ifndef O2_TRD_CALPAD_H
#define O2_TRD_CALPAD_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which are saved per pad             //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DataFormatsTRD/Constants.h"

class CalROC;
class CalDet;
class TH2F;
class TH1F;

class CalPad
{

 public:

  CalPad();
  CalPad(const std::string& name, const std::String& title);
  CalPad(const CalPad& c);
  ~CalPad();
  CalPad& operator=(const CalPad& c);

  static int getDet(int p, int c, int s) { return p + c * Constants::NLAYER + s * Constants::NLAYER * Constants::NSTACK; };

  CalROC* getCalROC(int d) const { return mROC[d]; };
  CalROC* getCalROC(int p, int c, int s) const
  {
    return mROC[getDet(p, c, s)];
  };

  bool scaleROCs(const CalDet* values);

  void setCalROC(int det, CalROC* calroc);

  // Statistic
  double getMeanRMS(double& rms, const CalDet* calDet = 0, int type = 0);
  double getMean(const CalDet* calDet = 0, int type = 0, CalPad* const outlierPad = 0);
  double getRMS(const CalDet* calDet = 0, int type = 0, CalPad* const outlierPad = 0);
  double getMedian(const CalDet* calDet = 0, int type = 0, CalPad* const outlierPad = 0);
  double getLTM(double* sigma = 0, double fraction = 0.9, const CalDet* calDet = 0, int type = 0, CalPad* const outlierPad = 0);

  // Plot functions
  TH1F* makeHisto1D(const CalDet* calDet = 0, int typedet = 0, float min = 4, float max = -4, int type = 0);
  TH2F* makeHisto2DSmPl(int sm, int pl, const CalDet* calDet = 0, int typedet = 0, float min = 4, float max = -4, int type = 0);
  TH2F* makeHisto2DCh(int ch, const CalDet* calDet = 0, int typedet = 0, float min = 4, float max = -4, int type = 0);

  // Algebra functions
  bool add(float c1);
  bool multiply(float c1);
  bool add(const CalPad* pad, double c1 = 1, const CalDet* calDet1 = 0, const CalDet* calDet2 = 0, int type = 0);
  bool multiply(const CalPad* pad, const CalDet* calDet1 = 0, const CalDet* calDet2 = 0, int type = 0);
  bool divide(const CalPad* pad, const CalDet* calDet1 = 0, const CalDet* calDet2 = 0, int type = 0);

 protected:
  std::vector<CalROC> mROC(Constants::MAXCHAMBER); //  Array of ROC objects which contain the values per pad

  ClassDef(CalPad, 1) //  TRD calibration class for parameters which are saved per pad
};

#endif
