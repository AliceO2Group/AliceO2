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
//  TRD pad calibrations base class                                          //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  This is analagous to the old CalROC but templatized so can store unsigned//
//      int(CalROC) and char SingleChamberStatus amongst others.
//  Most things were stored in AliTRDcalROC,AliTRDcalPad, AliTRDcalDet       //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/TRDGeometry.h"
#include "TRDBase/PadParameters.h"

using namespace o2::trd;

//
template <class T>
PadParameters<T>::PadParameters(int p, int c)
{

  init(p, c, nullptr);
}

template <class T>
int PadParameters<T>::init(int p, int c, std::vector<T>& data)
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

//
// algebra
template <class T>
bool PadParameters<T>::add(float c1)
{

  for (int i : mData) {
    mData[i] += c1;
  }
}

template <class T>
bool PadParameters<T>::multiply(float c1)
{
  for (int i : mData) {
    mData[i] *= c1;
  }
}
