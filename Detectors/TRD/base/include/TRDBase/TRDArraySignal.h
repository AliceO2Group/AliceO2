// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDARRAYSIGNAL_H
#define O2_TRDARRAYSIGNAL_H

/////////////////////////////////////////////
//                                         //
// Container Class for Signals             //
//                                         //
/////////////////////////////////////////////
#include <vector>
#include "Rtypes.h"
namespace o2
{
namespace trd
{

class TRDArraySignal
{
 public:
  TRDArraySignal();
  TRDArraySignal(int nrow, int ncol, int ntime);
  TRDArraySignal(const TRDArraySignal& d);
  ~TRDArraySignal();
  TRDArraySignal& operator=(const TRDArraySignal& d);

  void allocate(int nrow, int ncol, int ntime);
  void setNdet(int ndet) { mNdet = ndet; };
  int getNdet() const { return mNdet; };
  int getNrow() const { return mNrow; };
  int getNcol() const { return mNcol; };
  int getNtime() const { return mNtime; };
  float getDataByAdcCol(int row, int col, int time) const
  {
    return mSignal[(row * mNumberOfChannels + col) * mNtime + time];
  };
  void setDataByAdcCol(int row, int col, int time, float value)
  {
    mSignal[(row * mNumberOfChannels + col) * mNtime + time] = value;
  };
  bool hasData() const { return mNtime ? 1 : 0; };
  int getDim() const { return mNdim; };
  int getOverThreshold(float threshold) const;
  void compress(float minval);
  void expand();
  void reset();
  float getData(int nrow, int ncol, int ntime) const;
  void setData(int nrow, int ncol, int ntime, float value);

 protected:
  int mNdet{ 0 };             //ID number of the chamber
  int mNrow{ 0 };             //Number of rows of the chamber
  int mNcol{ 0 };             //Number of columns of the chamber
  int mNumberOfChannels{ 0 }; //  Number of MCM channels per row
  int mNtime{ 0 };            //Number of time bins
  int mNdim{ 0 };             //Dimension of the array
  std::vector<float> mSignal; //[fNdim]  //Pointer to signals

  ClassDefNV(TRDArraySignal, 1) //Signal container class
};
} //namespace trd
} //namespace o2
#endif
