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

using namespace std;
namespace o2
{
namespace trd
{

template <class T>
class PadParameters
{
 public:
  enum { kNplan = 6,
         kNcham = 5,
         kNsect = 18,
         kNdet = 540 };
  enum { kVdrift = 0,
         kGainFactor = 1,
         kT0 = 2,
         kExB = 3,
         kLocalGainFactor = 4 };
  PadParameters()=default;
  PadParameters(int p, int c);
  ~PadParameters() = default;
  int init(int c, std::vector<T>& data);
  //

  int getNrows() const { return mNrows; };
  int getNcols() const { return mNcols; };
  int getChannel(int c, int r) const { return r + c * mNrows; };
  int getNchannels() const { return mNchannels; };
  T getValue(int ich) const { return mData[ich]; };
  T getValue(int col, int row) { return getValue(getChannel(col, row)); };
  void setValue(int ich, T value) { mData[ich] = value; };
  void setValue(int col, int row, T value) { setValue(getChannel(col, row), value); };

 protected:
  int mPlane{0};        //  Plane number
  int mChamber{0};      //  Chamber number
  int mNrows{0};        //  Number of rows
  int mNcols{0};        //  Number of columns
  int mNchannels{0};    //  Number of channels = rows*columns
  std::vector<T> mData; // Size is mNchannels
};


//
template <class T>
PadParameters<T>::PadParameters(int p, int c)
{

  init(c, nullptr);
}

template <class T>
int PadParameters<T>::init(int c, std::vector<T>& data)
{
  if (c == 2)
    mNrows = 12;
  else
    mNrows = 16;
  mPlane = p;   //  Plane number
  mChamber = c; //  Chamber number
  mNcols = 144; //  Number of columns TODO look this up somewhere else.
  mNchannels = mNrows * mNcols;
  mData = data;
  if (mData.size() != mNchannels)
    LOG(FATAL) << "PadParamaters initialised with a size of " << mData.size() << " != " << mNchannels;

  return 0;
}


} // namespace trd
} // namespace o2
#endif
