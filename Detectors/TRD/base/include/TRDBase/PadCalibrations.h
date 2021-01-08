// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TRDBase/SimParam.h"
#include "DataFormatsTRD/Constants.h"

class Geometry;

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
  int getNChannels(int roc) { return mreadOutChamber[roc].getNChannels(); }
  PadParameters<T>& getChamberPads(int roc) { return mreadOutChamber[roc]; }
  T getValue(int roc, int col, int row) { return mreadOutChamber[roc].getValue(col, row); }
  T getPadValue(int roc, int col, int row) { return mreadOutChamber[roc].getValue(col, row); }
  void setPadValue(int roc, int col, int row, T value) { mreadOutChamber[roc].setValue(col, row, value); }
  void setPadValue(int roc, int channel, T value) { mreadOutChamber[roc].setValue(channel, value); }
  void reset(int roc, int col, int row, std::vector<T>& data);
  void init();
 protected:
  std::array<PadParameters<T>, constants::MAXCHAMBER> mreadOutChamber;
};

template <class T>
PadCalibrations<T>::PadCalibrations()
{
  //
  // CalPadStatus constructor
  //
  //Geometry fgeom;
  int chamberindex = 0;
  for (auto& roc : mreadOutChamber) { // Range-for!
    roc.init(chamberindex++);
  }
}

template <class T>
void PadCalibrations<T>::init()
{
  //
  // CalPadStatus constructor
  //
  int chamberindex = 0;
  for (auto& roc : mreadOutChamber) { // Range-for!
    roc.init(chamberindex++);
  }
}

template <class T>
void PadCalibrations<T>::reset(int roc, int col, int row, std::vector<T>& data)
{
  //reset the readoutchamber
  //primarily here for setting the values incoming from run2 ocdb, but might find other use cases.
  // you need to send it the roc as it is used to calculate internal parameters.
  mreadOutChamber[roc].reset(roc, col, row, data); // it *should* not actually matter as this *should* be set correctly via init.
}

} // namespace trd
} // namespace o2
#endif
