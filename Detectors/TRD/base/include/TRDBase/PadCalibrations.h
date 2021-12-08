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

#ifndef O2_TRD_PADCALIBRATIONS_H
#define O2_TRD_PADCALIBRATIONS_H

////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
//  TRD calibration class for parameters which are stored pad wise (1.2M entries) //
//  2019 - Ported from various bits of AliRoot (SHTM)                             //
//  Similar CalPad                                                                //
////////////////////////////////////////////////////////////////////////////////////

#include <array>

#include "TRDBase/PadParameters.h"
#include "DataFormatsTRD/Constants.h"

namespace o2
{
namespace trd
{

template <class T>
class PadCalibrations
{
 public:
  PadCalibrations();
  ~PadCalibrations() = default;
  //
  const PadParameters<T>& getChamberPads(int roc) { return mReadoutChamber[roc]; }
  T getValue(int roc, int col, int row) const { return mReadoutChamber[roc].getValue(col, row); }
  void setPadValue(int roc, int col, int row, T value) { mReadoutChamber[roc].setValue(col, row, value); }
  void setPadValue(int roc, int channel, T value) { mReadoutChamber[roc].setValue(channel, value); }

 private:
  std::array<PadParameters<T>, constants::MAXCHAMBER> mReadoutChamber;
  ClassDefNV(PadCalibrations, 1);
};

template <class T>
PadCalibrations<T>::PadCalibrations()
{
  //
  // CalPadStatus constructor
  //
  int chamberindex = 0;
  for (auto& roc : mReadoutChamber) { // Range-for!
    roc.init(chamberindex++);
  }
}

} // namespace trd
} // namespace o2
#endif
