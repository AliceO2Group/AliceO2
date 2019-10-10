// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_CHAMBERSTATUS_H
#define O2_TRD_CHAMBERSTATUS_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  Store the status of all the readout chambers across the TRD              //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Most things were stored in AliTRDcalROC,AliTRDcalPad, AliTRDcalDet       //
//  Note to old aliroot:
//      This is analgous to
///////////////////////////////////////////////////////////////////////////////

#include <array>
#include "TRDBase/TRDGeometry.h"
#include "TRDBase/TRDSimParam.h"
class TH2D;

namespace o2
{
namespace trd
{

class ChamberStatus
{
 public:
  enum { kGood = 0,
         kNoData = 1,
         kNoDataHalfChamberSideA = 2,
         kNoDataHalfChamberSideB = 3,
         kBadCalibrated = 4,
         kNotCalibrated = 5 };
  enum { kGoodpat = 0,
         kNoDatapat = 1,
         kNoDataHalfChamberSideApat = 4,
         kNoDataHalfChamberSideBpat = 8,
         kBadCalibratedpat = 16,
         kNotCalibratedpat = 32 }; // just 2^(line above)

  ChamberStatus() = default;
  ~ChamberStatus() = default;
  //
  //  char getStatus(int p, int c, int s) const { int roc=TRDGeometry::getDetector(p,c,s); return mStatus[roc];}
  char getStatus(int det) const { return mStatus[det]; }
  //  void setStatus(int p, int c, int s, char status) { int roc=TRDGeometry::getDetector(p,c,s); setStatus(roc,status);}
  void setStatus(int det, char status);
  void setRawStatus(int det, char status) { mStatus[det] = status; };
  void unsetStatusBit(int det, char status);
  bool isGood(int det) const { return (mStatus[det] & kGoodpat); }
  bool isNoData(int det) const { return (mStatus[det] & kNoDatapat); }
  bool isNoDataSideA(int det) const { return (mStatus[det] & kNoDataHalfChamberSideApat); }
  bool isNoDataSideB(int det) const { return (mStatus[det] & kNoDataHalfChamberSideBpat); }
  bool isBadCalibrated(int det) const { return (mStatus[det] & kBadCalibratedpat); }
  bool isNotCalibrated(int det) const { return (mStatus[det] & kNotCalibratedpat); }

  TH2D* plot(int sm, int rphi);              // Plot mStatus for sm and halfchamberside
  TH2D* plotNoData(int sm, int rphi);        // Plot data status for sm and halfchamberside
  TH2D* plotBadCalibrated(int sm, int rphi); // Plot calibration status for sm and halfchamberside
  TH2D* plot(int sm);                        // Plot mStatus for sm
 protected:
  std::array<char, TRDSimParam::kNdet> mStatus{};
};
} // namespace trd
} // namespace o2
#endif
