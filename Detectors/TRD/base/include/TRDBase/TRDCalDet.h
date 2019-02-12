#ifndef O2_TRDCALDET_H
#define O2_TRDCALDET_H

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which are saved per detector        //
//  2019 - Ported from AliRoot to O2 (J. Lopez)                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/TRDGeometry.h"
#include "TRDBase/TRDPadPlane.h"

class TH1F;
class TH2F;

namespace o2
{
namespace trd
{
class TRDCalDet
{
 public:
  enum { kNplan = 6,
         kNcham = 5,
         kNsect = 18,
         kNdet = 540 };
  TRDCalDet() = default;
  ~TRDCalDet() = default;
  //
  float getValue(int d) const { return mData[d]; };
  float getValue(int p, int c, int s) const { return mData[TRDGeometry::getDetector(p, c, s)]; };
  void setValue(int d, float value) { mData[d] = value; };
  void setValue(int p, int c, int s, float value) { mData[TRDGeometry::getDetector(p, c, s)] = value; };
  // statistic
  double getMean(TRDCalDet* const outlierDet = nullptr) const;
  double getMeanRobust(double robust = 0.92) const;
  double getRMS(TRDCalDet* const outlierDet = nullptr) const;
  double getRMSRobust(double robust = 0.92) const;
  double getMedian(TRDCalDet* const outlierDet = nullptr) const;
  double getLTM(double* sigma = nullptr, double fraction = 0.9, TRDCalDet* const outlierDet = nullptr);
  double calcMean(bool wghtPads = false);
  double calcMean(bool wghtPads, int& calib);
  double calcRMS(bool wghtPads = false);
  double calcRMS(bool wghtPads, int& calib);
  double getMeanSM(bool wghtPads, int sector) const;
  // Plot functions
  TH1F* makeHisto1Distribution(float min = 4, float max = -4, int type = 0);
  TH1F* makeHisto1DAsFunctionOfDet(float min = 4, float max = -4, int type = 0);
  TH2F* makeHisto2DCh(int ch, float min = 4, float max = -4, int type = 0);
  TH2F* makeHisto2DSmPl(int sm, int pl, float min = 4, float max = -4, int type = 0);
  // algebra functions
  void add(float c1);
  void multiply(float c1);
  void add(const TRDCalDet* calDet, double c1 = 1);
  void multiply(const TRDCalDet* calDet);
  void divide(const TRDCalDet* calDet);

 protected:
  std::array<float, kNdet> mData{}; // Data
};
} // namespace trd
} // namespace o2
#endif
