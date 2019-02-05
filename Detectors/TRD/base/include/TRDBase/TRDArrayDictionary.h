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

#include "AliTPCCommonRtypes.h" // for ClassDef


namespace o2
{
namespace trd
{


class TRDArrayDictionary
{

 public:

  TRDArrayDictionary();
  TRDArrayDictionary(int nrow, int ncol, int ntime);
  TRDArrayDictionary(const TRDArrayDictionary &a);
  ~TRDArrayDictionary();
  TRDArrayDictionary &operator=(const TRDArrayDictionary &a);

  void  allocate(int nrow, int ncol, int ntime);
  void  setNdet(int ndet) {fNdet=ndet;};  
  int getNdet()  const {return fNdet;};
  void  setDataByAdcCol(int nrow, int ncol, int ntime, int value)
                       {mDictionary[(nrow*mNumberOfChannels+ncol)*mNtime+ntime]=value;};
  int getDataByAdcCol(int nrow, int ncol, int ntime) const
               {return mDictionary[(nrow*mNumberOfChannels+ncol)*mNtime+ntime];};
  int getDim() const {return mNDdim;};
  void  compress();
  void  expand();
  void  reset();
  int getData(int nrow, int ncol, int ntime) const;
  void  setData(int nrow, int ncol, int ntime, int value);
  static  void    createLut();
  bool wasExpandCalled() const {return fFlag;};

 protected:

  int   mNdet{0};        //ID number of the chamber
  int   mNrow{0};        //Number of rows
  int   mNcol{0};        //Number of columns
  int   mNumberOfChannels{0};  //  Number of MCM channels per row
  int   mNtime{0};       //Number of time bins
  int   mNDdim{0};       //Dimension of the Dictionary array
  bool  mFlag{kFalse};        //! Has Expand() being called before?
  std::vector<int>  mDictionary;  //[fNDdim]  //Pointer to integers array
  bool mgLutPadNumberExists{kFalse};
  std::vector<short> mgLutPadNumbering;   //  [fNcol] Look Up Table


  ClassDefNV(TRDArrayDictionary,1) //Dictionary container class
    
};

}// trd namespace
}// o2 namespace
#endif
