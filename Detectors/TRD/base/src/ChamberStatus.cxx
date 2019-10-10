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
//  TRD calibration class for parameters which are saved frequently(/run)    //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Most things were stored in AliTRDcalROC,AliTRDcalPad, AliTRDcalDet       //
///////////////////////////////////////////////////////////////////////////////

//#include <vector>
#include <array>

#include "TRDBase/ChamberStatus.h"
#include "TH2D.h"
using namespace o2::trd;

void ChamberStatus::setStatus(int det, char status)
{

  //
  // set the chamber status
  //
  //

  switch (status) {
    case ChamberStatus::kGood:
      mStatus[det] |= kGoodpat;
      mStatus[det] &= !kNoDatapat;
      mStatus[det] &= !kBadCalibratedpat;
      break;
    case ChamberStatus::kNoData:
      mStatus[det] &= !kGoodpat;
      mStatus[det] |= kNoDatapat;
      mStatus[det] |= kNoDataHalfChamberSideApat;
      mStatus[det] |= kNoDataHalfChamberSideBpat;
      //      mStatus[det] |=  kBadCalibratedpat;
      break;
    case ChamberStatus::kNoDataHalfChamberSideA:
      mStatus[det] |= kNoDataHalfChamberSideApat;
      if (mStatus[det] & kNoDataHalfChamberSideB) {
        mStatus[det] |= kNoDatapat;
        mStatus[det] &= !kGoodpat;
      }
      break;
    case ChamberStatus::kNoDataHalfChamberSideB:
      mStatus[det] |= kNoDataHalfChamberSideBpat;
      if (mStatus[det] & kNoDataHalfChamberSideApat) {
        mStatus[det] &= !kGoodpat;
        mStatus[det] |= kNoDatapat;
      }
      break;
    case ChamberStatus::kBadCalibrated:
      mStatus[det] &= !kGoodpat;
      mStatus[det] |= kBadCalibratedpat;
      break;
    case ChamberStatus::kNotCalibrated:
      mStatus[det] &= !kGoodpat;
      mStatus[det] |= kNotCalibratedpat;
      // mStatus[det] &= !kBadCalibratedpat;
      break;
    default:
      mStatus[det] &= !kGoodpat;
      mStatus[det] &= !kNoDatapat;
      mStatus[det] &= !kNoDataHalfChamberSideApat;
      mStatus[det] &= !kNoDataHalfChamberSideBpat;
      mStatus[det] &= !kBadCalibratedpat;
      mStatus[det] &= !kNotCalibratedpat;
  }
}
//_____________________________________________________________________________
void ChamberStatus::unsetStatusBit(int det, char status)
{

  //
  // unset the chamber status bit
  //
  //

  switch (status) {
    case ChamberStatus::kGood:
      mStatus[det] &= !kGoodpat;
      break;
    case ChamberStatus::kNoData:
      mStatus[det] &= !kNoDatapat;
      break;
    case ChamberStatus::kNoDataHalfChamberSideA:
      mStatus[det] &= !kNoDataHalfChamberSideApat;
      break;
    case ChamberStatus::kNoDataHalfChamberSideB:
      mStatus[det] &= !kNoDataHalfChamberSideBpat;
      break;
    case ChamberStatus::kBadCalibrated:
      mStatus[det] &= !kBadCalibratedpat;
      break;
    case ChamberStatus::kNotCalibrated:
      mStatus[det] &= !kNotCalibratedpat;
      break;
    default:
      mStatus[det] &= !(kGoodpat & kNoDatapat & kNoDataHalfChamberSideApat & kNoDataHalfChamberSideBpat & kBadCalibratedpat & kNotCalibratedpat);
  }
}
//_____________________________________________________________________________
TH2D* ChamberStatus::plot(int sm, int rphi)
{
  //
  // Plot chamber status for supermodule and halfchamberside
  // as a function of layer and stack
  //

  TH2D* h2 = new TH2D(Form("sm_%d_rphi_%d", sm, rphi), Form("sm_%d_rphi_%d", sm, rphi), 5, 0.0, 5.0, 6, 0.0, 6.0);

  h2->SetDirectory(nullptr);
  h2->SetXTitle("stack");
  h2->SetYTitle("layer");

  int start = sm * 30;
  int end = (sm + 1) * 30;

  for (int i = start; i < end; i++) {
    int layer = i % 6;
    int stackn = static_cast<int>((i - start) / 6.);
    int status = getStatus(i);
    h2->Fill(stackn, layer, status);
    if (rphi == 0) {
      if (!(mStatus[i] & kNoDataHalfChamberSideBpat))
        h2->Fill(stackn, layer, status);
    } else if (rphi == 1) {
      if (!(mStatus[i] & kNoDataHalfChamberSideApat))
        h2->Fill(stackn, layer, status);
    }
  }

  return h2;
}
//_____________________________________________________________________________
TH2D* ChamberStatus::plotNoData(int sm, int rphi)
{
  //
  // Plot chamber data status for supermodule and halfchamberside
  // as a function of layer and stack
  //

  TH2D* h2 = new TH2D(Form("sm_%d_rphi_%d_data", sm, rphi), Form("sm_%d_rphi_%d_data", sm, rphi), 5, 0.0, 5.0, 6, 0.0, 6.0);

  h2->SetDirectory(nullptr);
  h2->SetXTitle("stack");
  h2->SetYTitle("layer");

  int start = sm * 30;
  int end = (sm + 1) * 30;

  for (int i = start; i < end; i++) {
    int layer = i % 6;
    int stackn = static_cast<int>((i - start) / 6.);
    if (rphi == 0) {
      if (mStatus[i] & kNoDataHalfChamberSideBpat)
        h2->Fill(stackn, layer, 1);
      if (mStatus[i] & kNoDatapat)
        h2->Fill(stackn, layer, 1);
    } else if (rphi == 1) {
      if (!(mStatus[i] & kNoDataHalfChamberSideApat))
        h2->Fill(stackn, layer, 1);
      if (!(mStatus[i] & kNoDatapat))
        h2->Fill(stackn, layer, 1);
    }
  }

  return h2;
}
//_____________________________________________________________________________
TH2D* ChamberStatus::plotBadCalibrated(int sm, int rphi)
{
  //
  // Plot chamber calibration status for supermodule and halfchamberside
  // as a function of layer and stack
  //

  TH2D* h2 = new TH2D(Form("sm_%d_rphi_%d_calib", sm, rphi), Form("sm_%d_rphi_%d_calib", sm, rphi), 5, 0.0, 5.0, 6, 0.0, 6.0);

  h2->SetDirectory(nullptr);
  h2->SetXTitle("stack");
  h2->SetYTitle("layer");

  int start = sm * 30;
  int end = (sm + 1) * 30;

  for (int i = start; i < end; i++) {
    int layer = i % 6;
    int stackn = static_cast<int>((i - start) / 6.);
    if (rphi == 0) {
      if (mStatus[i] & kBadCalibratedpat)
        h2->Fill(stackn, layer, 1);
    } else if (rphi == 1) {
      if (mStatus[i] & kBadCalibratedpat)
        h2->Fill(stackn, layer, 1);
    }
  }

  return h2;
}
//_____________________________________________________________________________
TH2D* ChamberStatus::plot(int sm)
{
  //
  // Plot chamber status for supermodule and halfchamberside
  // as a function of layer and stack
  //

  TH2D* h2 = new TH2D(Form("sm_%d", sm), Form("sm_%d", sm), 5, 0.0, 5.0, 6, 0.0, 6.0);

  h2->SetDirectory(nullptr);
  h2->SetXTitle("stack");
  h2->SetYTitle("layer");

  int start = sm * 30;
  int end = (sm + 1) * 30;

  for (int i = start; i < end; i++) {
    int layer = i % 6;
    int stackn = static_cast<int>((i - start) / 6.);
    int status = getStatus(i);
    h2->Fill(stackn, layer, status);
  }

  return h2;
}
