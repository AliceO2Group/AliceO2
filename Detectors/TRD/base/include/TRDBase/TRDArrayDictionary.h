// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDARRAYDICTIONARY_H
#define O2_TRDARRAYDICTIONARY_H

///////////////////////////////////////////////////
//                                               //
// Container Class for Dictionary Info           //
//                                               //
///////////////////////////////////////////////////

#include "GPUCommonRtypes.h" // for ClassDef

#include <vector>

namespace o2
{
namespace trd
{

class TRDArrayDictionary
{

 public:
  TRDArrayDictionary();
  TRDArrayDictionary(int nrow, int ncol, int ntime);
  TRDArrayDictionary(const TRDArrayDictionary& a);
  ~TRDArrayDictionary();
  TRDArrayDictionary& operator=(const TRDArrayDictionary& a);

  void allocate(int nrow, int ncol, int ntime);
  void setNdet(int ndet) { mNdet = ndet; };
  int getNdet() const { return mNdet; };
  void setDataByAdcCol(int nrow, int ncol, int ntime, int value)
  {
    mDictionary[(nrow * mNumberOfChannels + ncol) * mNtime + ntime] = value;
  };
  int getDataByAdcCol(int nrow, int ncol, int ntime) const
  {
    return mDictionary[(nrow * mNumberOfChannels + ncol) * mNtime + ntime];
  };
  int getDim() const { return mNDdim; };
  void compress();
  void expand();
  void reset();
  int getData(int nrow, int ncol, int ntime) const;
  void setData(int nrow, int ncol, int ntime, int value);
  static void createLut();
  bool wasExpandCalled() const { return mFlag; };

 protected:
  int mNdet{ 0 };               //ID number of the chamber
  int mNrow{ 0 };               //Number of rows
  int mNcol{ 0 };               //Number of columns
  int mNumberOfChannels{ 0 };   //  Number of MCM channels per row
  int mNtime{ 0 };              //Number of time bins
  int mNDdim{ 0 };              //Dimension of the Dictionary array
  bool mFlag{ 0 };              //! Has Expand() being called before?
  std::vector<int> mDictionary; //[mNDdim]  //Pointer to integers array

  ClassDefNV(TRDArrayDictionary, 1) //Dictionary container class
};

} // namespace trd
} // namespace o2
#endif
