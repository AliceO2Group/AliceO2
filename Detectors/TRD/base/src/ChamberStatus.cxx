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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which are saved frequently(/run)    //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Most things were stored in AliTRDcalROC,AliTRDcalPad, AliTRDcalDet       //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/ChamberStatus.h"
#include "TH2D.h"

using namespace o2::trd;

void ChamberStatus::setStatus(int det, int8_t bit)
{

  //
  // set the chamber status
  //
  //

  switch (bit) {
    case Good:
      mStatus[det] = Good;
      break;
    case NoData:
      mStatus[det] &= !Good;
      mStatus[det] |= NoData;
      mStatus[det] |= NoDataHalfChamberSideA;
      mStatus[det] |= NoDataHalfChamberSideB;
      break;
    case NoDataHalfChamberSideA:
      mStatus[det] |= NoDataHalfChamberSideA;
      if (mStatus[det] & NoDataHalfChamberSideB) {
        mStatus[det] |= NoData;
        mStatus[det] &= !Good;
      }
      break;
    case NoDataHalfChamberSideB:
      mStatus[det] |= NoDataHalfChamberSideB;
      if (mStatus[det] & NoDataHalfChamberSideA) {
        mStatus[det] &= !Good;
        mStatus[det] |= NoData;
      }
      break;
    case BadCalibrated:
      mStatus[det] &= !Good;
      mStatus[det] |= BadCalibrated;
      break;
    case NotCalibrated:
      mStatus[det] &= !Good;
      mStatus[det] |= NotCalibrated;
      break;
    default:
      mStatus[det] = 0;
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
      if (!(mStatus[i] & NoDataHalfChamberSideB)) {
        h2->Fill(stackn, layer, status);
      }
    } else if (rphi == 1) {
      if (!(mStatus[i] & NoDataHalfChamberSideA)) {
        h2->Fill(stackn, layer, status);
      }
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
      if (mStatus[i] & NoDataHalfChamberSideB) {
        h2->Fill(stackn, layer, 1);
      }
      if (mStatus[i] & NoData) {
        h2->Fill(stackn, layer, 1);
      }
    } else if (rphi == 1) {
      if (!(mStatus[i] & NoDataHalfChamberSideA)) {
        h2->Fill(stackn, layer, 1);
      }
      if (!(mStatus[i] & NoData)) {
        h2->Fill(stackn, layer, 1);
      }
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
      if (mStatus[i] & BadCalibrated) {
        h2->Fill(stackn, layer, 1);
      }
    } else if (rphi == 1) {
      if (mStatus[i] & BadCalibrated) {
        h2->Fill(stackn, layer, 1);
      }
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
