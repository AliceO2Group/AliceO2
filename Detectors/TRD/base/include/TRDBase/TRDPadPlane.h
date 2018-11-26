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

#include <Rtypes.h> // for ClassDef

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
  TRDPadPlane();
  TRDPadPlane(int layer, int stack);
  TRDPadPlane(const TRDPadPlane& p) = delete;
  TRDPadPlane& operator=(const TRDPadPlane& p) = delete;
  ~TRDPadPlane();
  // virtual void       Copy(TObject &p) const;

  void setLayer(int l) { mLayer = l; };
  void setStack(int s) { mStack = s; };
  void setRowSpacing(double s) { mRowSpacing = s; };
  void setColSpacing(double s) { mColSpacing = s; };
  void setLengthRim(double l) { mLengthRim = l; };
  void setWidthRim(double w) { mWidthRim = w; };
  void setNcols(int n)
  {
    mNcols = n;
    if (mPadCol)
      delete[] mPadCol;
    mPadCol = new double[mNcols];
  };
  void setNrows(int n)
  {
    mNrows = n;
    if (mPadRow)
      delete[] mPadRow;
    mPadRow = new double[mNrows];
  };
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

  int getPadRowNumber(double z) const;
  int getPadRowNumberROC(double z) const;
  int getPadColNumber(double rphi) const;

  double getTiltOffset(double rowOffset) const { return mTiltingTan * (rowOffset - 0.5 * mLengthIPad); };
  double getPadRowOffset(int row, double z) const
  {
    if ((row < 0) || (row >= mNrows))
      return -1.0;
    else
      return mPadRow[row] + mPadRowSMOffset - z;
  };
  double getPadRowOffsetROC(int row, double z) const
  {
    if ((row < 0) || (row >= mNrows))
      return -1.0;
    else
      return mPadRow[row] - z;
  };

  double getPadColOffset(int col, double rphi) const
  {
    if ((col < 0) || (col >= mNcols))
      return -1.0;
    else
      return rphi - mPadCol[col];
  };

  double getTiltingAngle() const { return mTiltingAngle; };
  int getNrows() const { return mNrows; };
  int getNcols() const { return mNcols; };
  double getRow0() const { return mPadRow[0] + mPadRowSMOffset; };
  double getRow0ROC() const { return mPadRow[0]; };
  double getCol0() const { return mPadCol[0]; };
  double getRowEnd() const { return mPadRow[mNrows - 1] - mLengthOPad + mPadRowSMOffset; };
  double getRowEndROC() const { return mPadRow[mNrows - 1] - mLengthOPad; };
  double getColEnd() const { return mPadCol[mNcols - 1] + mWidthOPad; };
  double getRowPos(int row) const { return mPadRow[row] + mPadRowSMOffset; };
  double getRowPosROC(int row) const { return mPadRow[row]; };
  double getColPos(int col) const { return mPadCol[col]; };
  double getRowSize(int row) const
  {
    if ((row == 0) || (row == mNrows - 1))
      return mLengthOPad;
    else
      return mLengthIPad;
  };
  double getColSize(int col) const
  {
    if ((col == 0) || (col == mNcols - 1))
      return mWidthOPad;
    else
      return mWidthIPad;
  };

  double getLengthRim() const { return mLengthRim; };
  double getWidthRim() const { return mWidthRim; };
  double getRowSpacing() const { return mRowSpacing; };
  double getColSpacing() const { return mColSpacing; };
  double getLengthOPad() const { return mLengthOPad; };
  double getLengthIPad() const { return mLengthIPad; };
  double getWidthOPad() const { return mWidthOPad; };
  double getWidthIPad() const { return mWidthIPad; };
  double getAnodeWireOffset() const { return mAnodeWireOffset; };
 protected:
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

  double* mPadRow; //  Pad border positions in row direction
  double* mPadCol; //  Pad border positions in column direction

  double mPadRowSMOffset; //  To be added to translate local ROC system to local SM system

  double mAnodeWireOffset; //  Distance of first anode wire from pad edge

 private:
  ClassDefNV(TRDPadPlane, 1) //  TRD ROC pad plane
};
}
}
#endif
