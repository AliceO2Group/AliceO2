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

#ifndef O2_TRD_PADPARAMETERS_H
#define O2_TRD_PADPARAMETERS_H

///////////////////////////////////////////////////////////////////////////////
//  TRD pad calibrations base class                                          //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  This is analagous to the old CalROC but templatized so can store unsigned//
//      int(CalROC) and char SingleChamberStatus amongst others.
//  Most things were stored in AliTRDcalROC,AliTRDcalPad, AliTRDcalDet       //
///////////////////////////////////////////////////////////////////////////////

//
#include <vector>

#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#include "Framework/Logger.h"

namespace o2
{
namespace trd
{

template <class T>
class PadParameters
{
 public:
  PadParameters() = default;
  PadParameters(int iDet);
  void init(int iDet);

  int getChannel(int col, int row) const;
  T getValue(int ich) const { return mData[ich]; };
  T getValue(int col, int row) const { return getValue(getChannel(col, row)); };
  void setValue(int ich, T value) { mData[ich] = value; }
  void setValue(int col, int row, T value) { setValue(getChannel(col, row), value); }

 private:
  std::vector<T> mData{}; /// One element for each pad
  ClassDefNV(PadParameters, 1);
};

template <class T>
PadParameters<T>::PadParameters(int iDet)
{
  init(iDet);
}

template <class T>
void PadParameters<T>::init(int iDet)
{
  auto nRows = HelperMethods::getStack(iDet) == 2 ? constants::NROWC0 : constants::NROWC1;
  auto nChannels = constants::NCOLUMN * nRows;
  mData.resize(nChannels);
}

template <class T>
int PadParameters<T>::getChannel(int col, int row) const
{
  if (mData.empty()) {
    LOG(error) << "Pad parameters not initialized";
  } else if (mData.size() == constants::NROWC0 * constants::NCOLUMN) {
    return row + col * constants::NROWC0;
  } else if (mData.size() == constants::NROWC1 * constants::NCOLUMN) {
    return row + col * constants::NROWC1;
  } else {
    LOG(error) << "Wrong number of channels set: " << mData.size();
  }
  return -1;
}

} // namespace trd
} // namespace o2
#endif
