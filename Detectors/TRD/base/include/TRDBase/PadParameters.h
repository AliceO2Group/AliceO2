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
  PadParameters(int chamberindex);
  ~PadParameters() = default;
  int init(int c);
  //

  int getNrows() const { return mNrows; };
  int getNcols() const { return mNcols; };
  int getChannel(int c, int r) const { return r + c * mNrows; };
  int getNchannels() const { return mNchannels; };
  T getValue(int ich) const { return mData[ich]; };
  T getValue(int col, int row) { cout <<" getting value from col:" << col <<" and row : " << row <<" with mNrows of :" << mNrows << " and mNchannels of : "<< mNchannels << " and channel of : " << row+col*mNrows << " and a vector size of : " << mData.size() << endl; return getValue(getChannel(col, row)); };
  void setValue(int ich, T value) { if(mData.size()==0) cout << " error setting a value on an empty array !&&$%!$!^#%^&*$*%^!%!@$%@#$%#$%^%" << endl; else mData[ich] = value; };
  void setValue(int col, int row, T value) { cout << "called ("<< col<<","<< row<<","<< value << ")"<< endl; cout << "size " << mData.size() << "p,c,r" << mPlane<<","<<mChamber<<","<<mNrows<<","<<mNchannels<<endl; cout << " and a base pointer of : " << this << endl; cout << "getChannel returns : " << getChannel(col,row) << endl; if(mData.size()==0) cout <<"error setting a value on an empty array @#$%#&^$&^#&@%^@#$%@#$%@#$%"<< endl; else setValue(getChannel(col, row), value); };
  void const debug(){ cout <<"in padparameters debug with this:" << this << " with mPlane:mChamber:mNrows:mNchannels::" << mPlane<<":"<<mNrows << ":"<< mNchannels << "   and a vector size of : " << mData.size() << endl; };
 protected:
  int mPlane{0};        //  Plane number
  int mChamber{0};      //  Chamber number
  int mNrows{0};        //  Number of rows
  int mNcols{144};        //  Number of columns
  int mNchannels;    //  Number of channels = rows*columns
  std::vector<T> mData; // Size is mNchannels
};


//
template <class T>
PadParameters<T>::PadParameters(int chamberindex)
{
cout << " init from the constructor of PadParameters" << endl;
  init(chamberindex);
}

template <class T>
int PadParameters<T>::init(int chamberindex)
{
    cout <<" ======================================== " << chamberindex << " =================================" << endl;
cout <<" mPlane before : " << mPlane <<" mChamber : " << mChamber;
  mPlane = TRDGeometry::getLayer(chamberindex);
  mChamber= TRDGeometry::getStack(chamberindex);
  if (mChamber == 2)
    mNrows = 12; // FeeParam::mgkNrowC0;
  else
    mNrows = 16; //FeeParam::mgkNrowC1;
  mNcols = 144; // FeeParam::mgkNcols; //  Number of columns TODO look this up somewhere else.
  // the FeeParam variables need to be unprotected, and dont want to change FeeParam in this PR.
  mNchannels = mNrows * mNcols;
  cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ mData size before resize  : " << mData.size() << " and going to reset it to " << mNchannels << endl;
  mData.resize(mNchannels);
  cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ mData size after resize  : " << mData.size() << " and going to reset it to " << mNchannels << endl;
  if (mData.size() != mNchannels)
    LOG(FATAL) << "PadParamaters initialised with a size of " << mData.size() << " != " << mNchannels;
  if(mNchannels ==0 || mData.size()==0) cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  cout << "PadParamaters initialised with a size of " << mData.size() << " with channels of " << mNchannels << " and a chamber index of : " << chamberindex << " plane, chamber rows, cols : " << mPlane << "::"<<mChamber<<"::"<<mNrows<<"::"<<mNcols<<endl;
  cout <<" mPlane before : " << mPlane <<" mChamber : " << mChamber;
  cout <<" ======================================== " << chamberindex << " =================================" << endl;
  return 0;
}


} // namespace trd
} // namespace o2
#endif
