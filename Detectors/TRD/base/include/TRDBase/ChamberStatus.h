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
#include "Rtypes.h"
#include "DataFormatsTRD/Constants.h"
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
  enum { kGoodpat = 1,
         kNoDatapat = 2,
         kNoDataHalfChamberSideApat = 4,
         kNoDataHalfChamberSideBpat = 8,
         kBadCalibratedpat = 16,
         kNotCalibratedpat = 32 }; // just 2^(line above)

  ChamberStatus() = default;
  ~ChamberStatus() = default;
  //
  char getStatus(int det) const { return mStatus[det]; }
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
 private:
  std::array<char, constants::MAXCHAMBER> mStatus{};
  ClassDefNV(ChamberStatus, 1);
};
} // namespace trd
} // namespace o2
#endif
