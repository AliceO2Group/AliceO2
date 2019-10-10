// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "TRDBase/TRDGeometry.h"
#include "TRDBase/TRDSimParam.h"
#include "TRDBase/FeeParam.h"
using namespace std;

namespace o2
{
namespace trd
{

template <class T>
class PadParameters
{
 public:
  PadParameters() = default;
  PadParameters(int chamberindex);
  ~PadParameters() = default;
  int init(int c);
  //

  int getNrows() const { return mNrows; };
  int getNcols() const { return mNcols; };
  int getChannel(int c, int r) const { return r + c * mNrows; };
  int getNchannels() const { return mNchannels; };
  T getValue(int ich) const { return mData[ich]; };
  T getValue(int col, int row) const { return getValue(getChannel(col, row)); };
  void setValue(int ich, T value) { mData[ich] = value; }
  void setValue(int col, int row, T value) { setValue(getChannel(col, row), value); }
  void debug() { cout << "in padparameters debug with this:" << this << " with mPlane:mChamber:mNrows:mNchannels::" << mPlane << ":" << mNrows << ":" << mNchannels << "   and a vector size of : " << mData.size() << endl; };
  void reset(int chamberindex, int col, int row, std::vector<T>& data);
  void dumpNonZeroValues(int roc)
  {
    int count = 0;
    for (auto& pad : mData) { // Range-for!
      if (pad > 0) {
        cout << "roc:pad:value" << roc << ":" << count << ":" << pad << endl;
      }
      count++;
    }
  }

 protected:
  int mPlane{0};                 //  Plane number
  int mChamber{0};               //  Chamber number
  int mNrows{0};                 //  Number of rows
  int mNcols{FeeParam::mgkNcol}; //  Number of columns
  int mNchannels;                //  Number of channels = rows*columns
  std::vector<T> mData;          // Size is mNchannels
};

//
template <class T>
PadParameters<T>::PadParameters(int chamberindex)
{
  init(chamberindex);
}

template <class T>
int PadParameters<T>::init(int chamberindex)
{
  mPlane = TRDGeometry::getLayer(chamberindex);
  mChamber = TRDGeometry::getStack(chamberindex);
  if (mChamber == 2)
    mNrows = FeeParam::mgkNrowC0;
  else
    mNrows = FeeParam::mgkNrowC1;
  // the FeeParam variables need to be unprotected, and dont want to change FeeParam in this PR.
  mNchannels = mNrows * mNcols;
  mData.resize(mNchannels);
  if (mData.size() != mNchannels || mData.size() == 0)
    LOG(FATAL) << "PadParamaters initialised with a size of " << mData.size() << " and mNchannels of " << mNchannels;
  return 0;
}

template <class T>
void PadParameters<T>::reset(int chamberindex, int cols, int rows, std::vector<T>& data)
{
  mPlane = TRDGeometry::getLayer(chamberindex);
  mChamber = TRDGeometry::getStack(chamberindex);
  mNrows = rows;
  mNcols = cols;
  // the FeeParam variables need to be unprotected, and dont want to change FeeParam in this PR.
  mNchannels = mNrows * mNcols;
  if (mData.size() != mNchannels)
    LOG(info) << "reseting a readoutchamber size which should not be needed given the detectors have not changed in run2 to run3";
  mData.resize(mNchannels);
  if (mData.size() != mNchannels || mData.size() == 0)
    LOG(FATAL) << "PadParamaters initialised with a size of " << mData.size() << " and mNchannels of " << mNchannels;

  // now reset the data of the pads.
  int counter = 0;
  for (auto pad : mData) {
    pad = data[counter];
    counter++;
  }
}

} // namespace trd
} // namespace o2
#endif
