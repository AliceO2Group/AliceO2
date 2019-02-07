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
  TRDArrayADC(Int_t nrow, Int_t ncol, Int_t ntime);
  TRDArrayADC(const TRDArrayADC& b);
  ~TRDArrayADC();
  TRDArrayADC& operator=(const TRDArrayADC& b);

  void allocate(Int_t nrow, Int_t ncol, Int_t ntime);
  void setNdet(Int_t ndet) { mNdet = ndet; };
  Int_t getNdet() const { return mNdet; };
  void setDataByAdcCol(Int_t nrow, Int_t ncol, Int_t ntime, Short_t value)
  {
    mADC[(nrow * mNumberOfChannels + ncol) * mNtime + ntime] = value;
  }
  Bool_t hasData() const { return mNtime ? 1 : 0; };
  Short_t getDataByAdcCol(Int_t nrow, Int_t ncol, Int_t ntime) const
  {
    return mADC[(nrow * mNumberOfChannels + ncol) * mNtime + ntime];
  };
  inline void getData(Int_t r, Int_t c, Int_t t, Int_t n, Short_t* vals) const;
  Short_t getDataBits(Int_t nrow, Int_t ncol, Int_t ntime) const;
  UChar_t getPadStatus(Int_t nrow, Int_t ncol, Int_t ntime) const;
  void setPadStatus(Int_t nrow, Int_t ncol, Int_t ntime, UChar_t status);
  Bool_t isPadCorrupted(Int_t nrow, Int_t ncol, Int_t ntime);
  void compress();
  void expand();
  Int_t getNtime() const { return mNtime; };
  Int_t getNrow() const { return mNrow; };
  Int_t getNcol() const { return mNcol; };
  Int_t getDim() const { return mNAdim; };
  void deleteNegatives();
  void reset();
  void conditionalReset(TRDSignalIndex* idx);
  inline Short_t* getDataAddress(Int_t nrow, Int_t ncol, Int_t ntime = 0) const;
  inline Short_t getData(Int_t nrow, Int_t ncol, Int_t ntime) const;
  inline void setData(Int_t nrow, Int_t ncol, Int_t ntime, Short_t value);
  static void createLut();

 protected:
  Int_t mNdet{ 0 };                  //ID number of the chamber
  Int_t mNrow{ 0 };                  //Number of rows
  Int_t mNcol{ 0 };                  //Number of columns(pads)
  Int_t mNumberOfChannels{ 0 };      //  Number of MCM channels per row
  Int_t mNtime{ 0 };                 //Number of time bins
  Int_t mNAdim{ 0 };                 //Dimension of the ADC array
                                     //  std::vector<Short_t> mADC;  //[mNAdim]   //Pointer to adc values
  Short_t* mADC = nullptr;           //[mNAdim]   //Pointer to adc values
  static Short_t* mgLutPadNumbering; //  [mNcol] Look Up Table
  ClassDefNV(TRDArrayADC, 1)         //ADC container class
};

} // end of namespace trd
} // end of namespace o2
#endif
