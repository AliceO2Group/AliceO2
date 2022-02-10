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

#ifndef O2_ZDC_ZDCTDCCORR_H
#define O2_ZDC_ZDCTDCCORR_H

#include "ZDCBase/Constants.h"
#include <array>
#include <Rtypes.h>

/// \file ZDCTDCCorr.h
/// \brief ZDC TDC correction parameters
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct ZDCTDCCorr {
  const static int NParExtS = 6;
  const static int NParExtC = 4;
  const static int NParMidS = 3;
  const static int NParMidC = 1;
  std::array<std::array<std::array<float, NFParA>, NBucket>, NTDCChannels> mAmpSigCorr;                                      // Amplitude correction of signal w/o pile-up
  std::array<std::array<std::array<std::array<std::array<float, NFParT>, NBucket>, NBucket>, NBCAn>, NTDCChannels> mTDCCorr; // TDC correction of signal with pile-up
  std::array<std::array<std::array<std::array<std::array<float, NFParA>, NBucket>, NBucket>, NBCAn>, NTDCChannels> mAmpCorr; // Amplitude correction of signal with pile-up

  std::array<std::array<float, NParExtC>, NTDCChannels> mTSBegC; // TDC time correction, constant, beginning of sequence
  std::array<std::array<float, NParMidC>, NTDCChannels> mTSMidC; // TDC time correction, constant, mid of sequence
  std::array<std::array<float, NParExtC>, NTDCChannels> mTSEndC; // TDC time correction, constant, end of sequence
  std::array<std::array<float, NParExtC>, NTDCChannels> mAFBegC; // TDC amplitude correction, constant, beginning of sequence
  std::array<std::array<float, NParMidC>, NTDCChannels> mAFMidC; // TDC amplitude correction, constant, mid of sequence
  std::array<std::array<float, NParExtC>, NTDCChannels> mAFEndC; // TDC amplitude correction, constant, end of sequence

  void SetAmpSigCorr(int itdc, int ibuk, int ipar, float val)
  {
    mAmpSigCorr[itdc][ibuk][ipar] = val;
  }
  void SetTDCCorr(int itdc, int ibun, int ibukb, int ibuks, int ipar, float val)
  {
    mAmpCorr[itdc][ibun][ibukb][ibuks][ipar] = val;
  }
  void SetAmpCorr(int itdc, int ibun, int ibukb, int ibuks, int ipar, float val)
  {
    mTDCCorr[itdc][ibun][ibukb][ibuks][ipar] = val;
  }

  ZDCTDCCorr()
  {
    clear();
  }
  void clear();
  void print() const;
  void dump() const;
  ClassDefNV(ZDCTDCCorr, 1);
};
} // namespace zdc
} // namespace o2

#endif
