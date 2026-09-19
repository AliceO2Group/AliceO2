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

#ifndef O2_TRD_PADPLANE_H
#define O2_TRD_PADPLANE_H

// Forwards to standard header with protection for GPU compilation
#include "GPUCommonRtypes.h" // for ClassDef
#include "GPUCommonDef.h"
#include "GPUCommonDouble.h"

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  TRD pad plane class                                                   //
//                                                                        //
//  Contains the information on ideal pad positions, pad dimensions,      //
//  tilting angle, etc.                                                   //
//  It also provides methods to identify the current pad number from      //
//  local tracking coordinates.                                           //
//                                                                        //
////////////////////////////////////////////////////////////////////////////
namespace o2
{
namespace trd
{
class PadPlane
{
 public:
  PadPlane() = default;
  PadPlane(int layer, int stack) : mLayer(layer), mStack(stack){};
  PadPlane(const PadPlane& p) = delete;
  PadPlane& operator=(const PadPlane& p) = delete;
  ~PadPlane() = default;

#ifndef GPUCA_GPUCODE_DEVICE
  void setLayer(int l) { mLayer = l; };
  void setStack(int s) { mStack = s; };
  void setRowSpacing(o2::gpu::GPUdoubleValue s) { mRowSpacing = s; };
  void setColSpacing(o2::gpu::GPUdoubleValue s) { mColSpacing = s; };
  void setLengthRim(o2::gpu::GPUdoubleValue l) { mLengthRim = l; };
  void setWidthRim(o2::gpu::GPUdoubleValue w) { mWidthRim = w; };
  void setNcols(int n);
  void setNrows(int n);
  void setPadCol(int ic, o2::gpu::GPUdoubleValue c)
  {
    if (ic < mNcols) {
      mPadCol[ic] = c;
    }
  };
  void setPadRow(int ir, o2::gpu::GPUdoubleValue r)
  {
    if (ir < mNrows) {
      mPadRow[ir] = r;
    }
  };
  void setLength(o2::gpu::GPUdoubleValue l) { mLength = l; };
  void setWidth(o2::gpu::GPUdoubleValue w) { mWidth = w; };
  void setLengthOPad(o2::gpu::GPUdoubleValue l)
  {
    mLengthOPad = l;
    mInverseLengthOPad = 1.0 / l;
  };
  void setWidthOPad(o2::gpu::GPUdoubleValue w)
  {
    mWidthOPad = w;
    mInverseWidthOPad = 1.0 / w;
  };
  void setLengthIPad(o2::gpu::GPUdoubleValue l)
  {
    mLengthIPad = l;
    mInverseLengthIPad = 1.0 / l;
  };
  void setWidthIPad(o2::gpu::GPUdoubleValue w)
  {
    mWidthIPad = w;
    mInverseWidthIPad = 1.0 / w;
  };
  void setPadRowSMOffset(o2::gpu::GPUdoubleValue o) { mPadRowSMOffset = o; };
  void setAnodeWireOffset(float o) { mAnodeWireOffset = o; };
  void setTiltingAngle(o2::gpu::GPUdoubleValue t);
#endif

  GPUd() int getPadRowNumber(o2::gpu::GPUdoubleValue z) const
  {
    //
    // Finds the pad row number for a given z-position in local supermodule system
    //
    int row = 0;
    int nabove = 0;
    int nbelow = 0;
    int middle = 0;

    if ((z > getRow0()) || (z < getRowEnd())) {
      row = -1;

    } else {
      nabove = mNrows + 1;
      nbelow = 0;
      while (nabove - nbelow > 1) {
        middle = (nabove + nbelow) / 2;
        if (z == (o2::gpu::GPUdoubleGet(mPadRow[middle - 1]) + o2::gpu::GPUdoubleGet(mPadRowSMOffset))) {
          row = middle;
        }
        if (z > (o2::gpu::GPUdoubleGet(mPadRow[middle - 1]) + o2::gpu::GPUdoubleGet(mPadRowSMOffset))) {
          nabove = middle;
        } else {
          nbelow = middle;
        }
      }
      row = nbelow - 1;
    }

    return row;
  };

  GPUd() int getPadRowNumberROC(o2::gpu::GPUdoubleValue z) const;
  GPUd() o2::gpu::GPUdoubleValue getPadRow(o2::gpu::GPUdoubleValue z) const;
  GPUd() int getPadColNumber(o2::gpu::GPUdoubleValue rphi) const;
  GPUd() o2::gpu::GPUdoubleValue getPad(o2::gpu::GPUdoubleValue y, o2::gpu::GPUdoubleValue z) const;

  GPUd() o2::gpu::GPUdoubleValue getTiltOffset(int row, o2::gpu::GPUdoubleValue rowOffset) const
  {
    if (row == 0 || row == mNrows - 1) {
      return o2::gpu::GPUdoubleGet(mTiltingTan) * (rowOffset - 0.5 * o2::gpu::GPUdoubleGet(mLengthOPad));
    } else {
      return o2::gpu::GPUdoubleGet(mTiltingTan) * (rowOffset - 0.5 * o2::gpu::GPUdoubleGet(mLengthIPad));
    }
  };
  GPUd() o2::gpu::GPUdoubleValue getPadRowOffset(int row, o2::gpu::GPUdoubleValue z) const
  {
    if ((row < 0) || (row >= mNrows)) {
      return -1.0;
    } else {
      return o2::gpu::GPUdoubleGet(mPadRow[row]) + o2::gpu::GPUdoubleGet(mPadRowSMOffset) - z;
    }
  };
  GPUd() o2::gpu::GPUdoubleValue getPadRowOffsetROC(int row, o2::gpu::GPUdoubleValue z) const
  {
    if ((row < 0) || (row >= mNrows)) {
      return -1.0;
    } else {
      return o2::gpu::GPUdoubleGet(mPadRow[row]) - z;
    }
  };

  GPUd() o2::gpu::GPUdoubleValue getPadColOffset(int col, o2::gpu::GPUdoubleValue rphi) const
  {
    if ((col < 0) || (col >= mNcols)) {
      return -1.0;
    } else {
      return rphi - o2::gpu::GPUdoubleGet(mPadCol[col]);
    }
  };

  GPUd() o2::gpu::GPUdoubleValue getTiltingAngle() const { return o2::gpu::GPUdoubleGet(mTiltingAngle); };
  GPUd() int getNrows() const { return mNrows; };
  GPUd() int getNcols() const { return mNcols; };
  GPUd() o2::gpu::GPUdoubleValue getRow0() const { return o2::gpu::GPUdoubleGet(mPadRow[0]) + o2::gpu::GPUdoubleGet(mPadRowSMOffset); };
  GPUd() o2::gpu::GPUdoubleValue getRow0ROC() const { return o2::gpu::GPUdoubleGet(mPadRow[0]); };
  GPUd() o2::gpu::GPUdoubleValue getCol0() const { return o2::gpu::GPUdoubleGet(mPadCol[0]); };
  GPUd() o2::gpu::GPUdoubleValue getRowEnd() const { return o2::gpu::GPUdoubleGet(mPadRow[mNrows - 1]) - o2::gpu::GPUdoubleGet(mLengthOPad) + o2::gpu::GPUdoubleGet(mPadRowSMOffset); };
  GPUd() o2::gpu::GPUdoubleValue getRowEndROC() const { return o2::gpu::GPUdoubleGet(mPadRow[mNrows - 1]) - o2::gpu::GPUdoubleGet(mLengthOPad); };
  GPUd() o2::gpu::GPUdoubleValue getColEnd() const { return o2::gpu::GPUdoubleGet(mPadCol[mNcols - 1]) + o2::gpu::GPUdoubleGet(mWidthOPad); };
  GPUd() o2::gpu::GPUdoubleValue getRowPos(int row) const { return o2::gpu::GPUdoubleGet(mPadRow[row]) + o2::gpu::GPUdoubleGet(mPadRowSMOffset); };
  GPUd() o2::gpu::GPUdoubleValue getRowPosROC(int row) const { return o2::gpu::GPUdoubleGet(mPadRow[row]); };
  GPUd() o2::gpu::GPUdoubleValue getColPos(int col) const { return o2::gpu::GPUdoubleGet(mPadCol[col]); };
  GPUd() o2::gpu::GPUdoubleValue getRowSize(int row) const
  {
    if ((row == 0) || (row == mNrows - 1)) {
      return o2::gpu::GPUdoubleGet(mLengthOPad);
    } else {
      return o2::gpu::GPUdoubleGet(mLengthIPad);
    }
  };
  GPUd() o2::gpu::GPUdoubleValue getColSize(int col) const
  {
    if ((col == 0) || (col == mNcols - 1)) {
      return o2::gpu::GPUdoubleGet(mWidthOPad);
    } else {
      return o2::gpu::GPUdoubleGet(mWidthIPad);
    }
  };

  GPUd() o2::gpu::GPUdoubleValue getLengthRim() const { return o2::gpu::GPUdoubleGet(mLengthRim); };
  GPUd() o2::gpu::GPUdoubleValue getWidthRim() const { return o2::gpu::GPUdoubleGet(mWidthRim); };
  GPUd() o2::gpu::GPUdoubleValue getRowSpacing() const { return o2::gpu::GPUdoubleGet(mRowSpacing); };
  GPUd() o2::gpu::GPUdoubleValue getColSpacing() const { return o2::gpu::GPUdoubleGet(mColSpacing); };
  GPUd() o2::gpu::GPUdoubleValue getLengthOPad() const { return o2::gpu::GPUdoubleGet(mLengthOPad); };
  GPUd() o2::gpu::GPUdoubleValue getLengthIPad() const { return o2::gpu::GPUdoubleGet(mLengthIPad); };
  GPUd() o2::gpu::GPUdoubleValue getWidthOPad() const { return o2::gpu::GPUdoubleGet(mWidthOPad); };
  GPUd() o2::gpu::GPUdoubleValue getWidthIPad() const { return o2::gpu::GPUdoubleGet(mWidthIPad); };
  GPUd() o2::gpu::GPUdoubleValue getAnodeWireOffset() const { return o2::gpu::GPUdoubleGet(mAnodeWireOffset); };

 private:
  static GPUglobalconstexpr() int MAXCOLS = 144;
  static GPUglobalconstexpr() int MAXROWS = 16;

  int mLayer; //  Layer number
  int mStack; //  Stack number

  o2::gpu::GPUdoubleStore mLength; //  Length of pad plane in z-direction (row)
  o2::gpu::GPUdoubleStore mWidth;  //  Width of pad plane in rphi-direction (col)

  o2::gpu::GPUdoubleStore mLengthRim; //  Length of the rim in z-direction (row)
  o2::gpu::GPUdoubleStore mWidthRim;  //  Width of the rim in rphi-direction (col)

  o2::gpu::GPUdoubleStore mLengthOPad; //  Length of an outer pad in z-direction (row)
  o2::gpu::GPUdoubleStore mWidthOPad;  //  Width of an outer pad in rphi-direction (col)

  o2::gpu::GPUdoubleStore mLengthIPad; //  Length of an inner pad in z-direction (row)
  o2::gpu::GPUdoubleStore mWidthIPad;  //  Width of an inner pad in rphi-direction (col)

  o2::gpu::GPUdoubleStore mRowSpacing; //  Spacing between the pad rows
  o2::gpu::GPUdoubleStore mColSpacing; //  Spacing between the pad columns

  int mNrows; //  Number of rows
  int mNcols; //  Number of columns

  o2::gpu::GPUdoubleStore mTiltingAngle; //  Pad tilting angle
  o2::gpu::GPUdoubleStore mTiltingTan;   //  Tangens of pad tilting angle

  o2::gpu::GPUdoubleStore mPadRow[MAXROWS]; //  Pad border positions in row direction
  o2::gpu::GPUdoubleStore mPadCol[MAXCOLS]; //  Pad border positions in column direction

  o2::gpu::GPUdoubleStore mPadRowSMOffset; //  To be added to translate local ROC system to local SM system

  o2::gpu::GPUdoubleStore mAnodeWireOffset; //  Distance of first anode wire from pad edge

  o2::gpu::GPUdoubleStore mInverseLengthIPad; // 1 / mLengthIPad
  o2::gpu::GPUdoubleStore mInverseLengthOPad; // 1 / mLengthOPad

  o2::gpu::GPUdoubleStore mInverseWidthIPad; // 1 / mWidthIPad
  o2::gpu::GPUdoubleStore mInverseWidthOPad; // 1 / mWidthOPad

  ClassDefNV(PadPlane, 2); //  TRD ROC pad plane
};
} // namespace trd
} // namespace o2
#endif
