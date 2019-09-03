// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDPADPLANE_H
#define O2_TRDPADPLANE_H

//Forwards to standard header with protection for GPU compilation
#include "GPUCommonRtypes.h" // for ClassDef

#include "GPUCommonDef.h"

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
class TRDPadPlane
{
 public:
  TRDPadPlane() = default;
  TRDPadPlane(int layer, int stack) : mLayer(layer), mStack(stack){};
  TRDPadPlane(const TRDPadPlane& p) = delete;
  TRDPadPlane& operator=(const TRDPadPlane& p) = delete;
  ~TRDPadPlane() = default;

  void setLayer(int l) { mLayer = l; };
  void setStack(int s) { mStack = s; };
  void setRowSpacing(double s) { mRowSpacing = s; };
  void setColSpacing(double s) { mColSpacing = s; };
  void setLengthRim(double l) { mLengthRim = l; };
  void setWidthRim(double w) { mWidthRim = w; };
  void setNcols(int n);
  void setNrows(int n);
  void setPadCol(int ic, double c)
  {
    if (ic < mNcols)
      mPadCol[ic] = c;
  };
  void setPadRow(int ir, double r)
  {
    if (ir < mNrows)
      mPadRow[ir] = r;
  };
  void setLength(double l) { mLength = l; };
  void setWidth(double w) { mWidth = w; };
  void setLengthOPad(double l) { mLengthOPad = l; };
  void setWidthOPad(double w) { mWidthOPad = w; };
  void setLengthIPad(double l) { mLengthIPad = l; };
  void setWidthIPad(double w) { mWidthIPad = w; };
  void setPadRowSMOffset(double o) { mPadRowSMOffset = o; };
  void setAnodeWireOffset(float o) { mAnodeWireOffset = o; };
  void setTiltingAngle(double t);

  GPUd() int getPadRowNumber(double z) const;
  GPUd() int getPadRowNumberROC(double z) const;
  GPUd() int getPadColNumber(double rphi) const;

  GPUd() double getTiltOffset(double rowOffset) const { return mTiltingTan * (rowOffset - 0.5 * mLengthIPad); };
  GPUd() double getPadRowOffset(int row, double z) const
  {
    if ((row < 0) || (row >= mNrows))
      return -1.0;
    else
      return mPadRow[row] + mPadRowSMOffset - z;
  };
  GPUd() double getPadRowOffsetROC(int row, double z) const
  {
    if ((row < 0) || (row >= mNrows))
      return -1.0;
    else
      return mPadRow[row] - z;
  };

  GPUd() double getPadColOffset(int col, double rphi) const
  {
    if ((col < 0) || (col >= mNcols))
      return -1.0;
    else
      return rphi - mPadCol[col];
  };

  GPUd() double getTiltingAngle() const { return mTiltingAngle; };
  GPUd() int getNrows() const { return mNrows; };
  GPUd() int getNcols() const { return mNcols; };
  GPUd() double getRow0() const { return mPadRow[0] + mPadRowSMOffset; };
  GPUd() double getRow0ROC() const { return mPadRow[0]; };
  GPUd() double getCol0() const { return mPadCol[0]; };
  GPUd() double getRowEnd() const { return mPadRow[mNrows - 1] - mLengthOPad + mPadRowSMOffset; };
  GPUd() double getRowEndROC() const { return mPadRow[mNrows - 1] - mLengthOPad; };
  GPUd() double getColEnd() const { return mPadCol[mNcols - 1] + mWidthOPad; };
  GPUd() double getRowPos(int row) const { return mPadRow[row] + mPadRowSMOffset; };
  GPUd() double getRowPosROC(int row) const { return mPadRow[row]; };
  GPUd() double getColPos(int col) const { return mPadCol[col]; };
  GPUd() double getRowSize(int row) const
  {
    if ((row == 0) || (row == mNrows - 1))
      return mLengthOPad;
    else
      return mLengthIPad;
  };
  GPUd() double getColSize(int col) const
  {
    if ((col == 0) || (col == mNcols - 1))
      return mWidthOPad;
    else
      return mWidthIPad;
  };

  GPUd() double getLengthRim() const { return mLengthRim; };
  GPUd() double getWidthRim() const { return mWidthRim; };
  GPUd() double getRowSpacing() const { return mRowSpacing; };
  GPUd() double getColSpacing() const { return mColSpacing; };
  GPUd() double getLengthOPad() const { return mLengthOPad; };
  GPUd() double getLengthIPad() const { return mLengthIPad; };
  GPUd() double getWidthOPad() const { return mWidthOPad; };
  GPUd() double getWidthIPad() const { return mWidthIPad; };
  GPUd() double getAnodeWireOffset() const { return mAnodeWireOffset; };

 protected:
  static constexpr int MAXCOLS = 144;
  static constexpr int MAXROWS = 16;

  int mLayer; //  Layer number
  int mStack; //  Stack number

  double mLength; //  Length of pad plane in z-direction (row)
  double mWidth;  //  Width of pad plane in rphi-direction (col)

  double mLengthRim; //  Length of the rim in z-direction (row)
  double mWidthRim;  //  Width of the rim in rphi-direction (col)

  double mLengthOPad; //  Length of an outer pad in z-direction (row)
  double mWidthOPad;  //  Width of an outer pad in rphi-direction (col)

  double mLengthIPad; //  Length of an inner pad in z-direction (row)
  double mWidthIPad;  //  Width of an inner pad in rphi-direction (col)

  double mRowSpacing; //  Spacing between the pad rows
  double mColSpacing; //  Spacing between the pad columns

  int mNrows; //  Number of rows
  int mNcols; //  Number of columns

  double mTiltingAngle; //  Pad tilting angle
  double mTiltingTan;   //  Tangens of pad tilting angle

  double mPadRow[MAXROWS]; //  Pad border positions in row direction
  double mPadCol[MAXCOLS]; //  Pad border positions in column direction

  double mPadRowSMOffset; //  To be added to translate local ROC system to local SM system

  double mAnodeWireOffset; //  Distance of first anode wire from pad edge

 private:
  ClassDefNV(TRDPadPlane, 1); //  TRD ROC pad plane
};
} // namespace trd
} // namespace o2
#endif
