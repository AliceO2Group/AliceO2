// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDARRAYADC_H
#define O2_TRDARRAYADC_H

namespace o2
{
namespace trd
{

///////////////////////////////////////////////
//                                           //
// Container class for ADC values            //
//                                           //
///////////////////////////////////////////////

// ported from cblume original code in AliRoot, possibly some modifications
//
#include "Rtypes.h"

class TRDSignalIndex;

class TRDArrayADC
{
 public:
  TRDArrayADC();
  TRDArrayADC(int nrow, int ncol, int ntime);
  TRDArrayADC(const TRDArrayADC& b);
  ~TRDArrayADC();
  TRDArrayADC& operator=(const TRDArrayADC& b);

  void allocate(int nrow, int ncol, int ntime);
  void setNdet(int ndet) { mNdet = ndet; };
  int getNdet() const { return mNdet; };
  void setDataByAdcCol(int nrow, int ncol, int ntime, short value)
  {
    mADC[(nrow * mNumberOfChannels + ncol) * mNtime + ntime] = value;
  }
  bool HasData() const { return mNtime ? 1 : 0; };
  short getDataByAdcCol(int nrow, int ncol, int ntime) const
  {
    return mADC[(nrow * mNumberOfChannels + ncol) * mNtime + ntime];
  };
  inline void getData(int r, int c, int t, int n, short* vals) const;
  short getDataBits(int nrow, int ncol, int ntime) const;
  unsigned char getPadStatus(int nrow, int ncol, int ntime) const;
  void setPadStatus(int nrow, int ncol, int ntime, unsigned char status);
  bool IsPadCorrupted(int nrow, int ncol, int ntime);
  void compress();
  void expand();
  int getNtime() const { return mNtime; };
  int getNrow() const { return mNrow; };
  int getNcol() const { return mNcol; };
  int getDim() const { return mNAdim; };
  void deleteNegatives();
  void reset();
  void conditionalReset(TRDSignalIndex* idx);
  inline short* getDataAddress(int nrow, int ncol, int ntime = 0) const;
  inline short getData(int nrow, int ncol, int ntime) const;
  inline void setData(int nrow, int ncol, int ntime, short value);
  static void createLut();

 protected:
  int mNdet{ 0 };                  //ID number of the chamber
  int mNrow{ 0 };                  //Number of rows
  int mNcol{ 0 };                  //Number of columns(pads)
  int mNumberOfChannels{ 0 };      //  Number of MCM channels per row
  int mNtime{ 0 };                 //Number of time bins
  int mNAdim{ 0 };                 //Dimension of the ADC array
                                     //  std::vector<short> mADC;  //[mNAdim]   //Pointer to adc values
  short* mADC = nullptr;           //[mNAdim]   //Pointer to adc values
  static short* mgLutPadNumbering; //  [mNcol] Look Up Table
  ClassDefNV(TRDArrayADC, 1)         //ADC container class
};

} // end of namespace trd
} // end of namespace o2
#endif
