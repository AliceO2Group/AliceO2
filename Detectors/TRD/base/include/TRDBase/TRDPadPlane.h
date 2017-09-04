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
  ~TRDPadPlane();
  // virtual void       Copy(TObject &p) const;

  void SetLayer(int l) { mLayer = l; };
  void SetStack(int s) { mStack = s; };
  void SetRowSpacing(double s) { mRowSpacing = s; };
  void SetColSpacing(double s) { mColSpacing = s; };
  void SetLengthRim(double l) { mLengthRim = l; };
  void SetWidthRim(double w) { mWidthRim = w; };
  void SetNcols(int n)
  {
    mNcols = n;
    if (mPadCol)
      delete[] mPadCol;
    mPadCol = new double[mNcols];
  };
  void SetNrows(int n)
  {
    mNrows = n;
    if (mPadRow)
      delete[] mPadRow;
    mPadRow = new double[mNrows];
  };
  void SetPadCol(int ic, double c)
  {
    if (ic < mNcols)
      mPadCol[ic] = c;
  };
  void SetPadRow(int ir, double r)
  {
    if (ir < mNrows)
      mPadRow[ir] = r;
  };
  void SetLength(double l) { mLength = l; };
  void SetWidth(double w) { mWidth = w; };
  void SetLengthOPad(double l) { mLengthOPad = l; };
  void SetWidthOPad(double w) { mWidthOPad = w; };
  void SetLengthIPad(double l) { mLengthIPad = l; };
  void SetWidthIPad(double w) { mWidthIPad = w; };
  void SetPadRowSMOffset(double o) { mPadRowSMOffset = o; };
  void SetAnodeWireOffset(float o) { mAnodeWireOffset = o; };
  void SetTiltingAngle(double t);

  int GetPadRowNumber(double z) const;
  int GetPadRowNumberROC(double z) const;
  int GetPadColNumber(double rphi) const;

  double GetTiltOffset(double rowOffset) const { return mTiltingTan * (rowOffset - 0.5 * mLengthIPad); };
  double GetPadRowOffset(int row, double z) const
  {
    if ((row < 0) || (row >= mNrows))
      return -1.0;
    else
      return mPadRow[row] + mPadRowSMOffset - z;
  };
  double GetPadRowOffsetROC(int row, double z) const
  {
    if ((row < 0) || (row >= mNrows))
      return -1.0;
    else
      return mPadRow[row] - z;
  };

  double GetPadColOffset(int col, double rphi) const
  {
    if ((col < 0) || (col >= mNcols))
      return -1.0;
    else
      return rphi - mPadCol[col];
  };

  double GetTiltingAngle() const { return mTiltingAngle; };
  int GetNrows() const { return mNrows; };
  int GetNcols() const { return mNcols; };
  double GetRow0() const { return mPadRow[0] + mPadRowSMOffset; };
  double GetRow0ROC() const { return mPadRow[0]; };
  double GetCol0() const { return mPadCol[0]; };
  double GetRowEnd() const { return mPadRow[mNrows - 1] - mLengthOPad + mPadRowSMOffset; };
  double GetRowEndROC() const { return mPadRow[mNrows - 1] - mLengthOPad; };
  double GetColEnd() const { return mPadCol[mNcols - 1] + mWidthOPad; };
  double GetRowPos(int row) const { return mPadRow[row] + mPadRowSMOffset; };
  double GetRowPosROC(int row) const { return mPadRow[row]; };
  double GetColPos(int col) const { return mPadCol[col]; };
  double GetRowSize(int row) const
  {
    if ((row == 0) || (row == mNrows - 1))
      return mLengthOPad;
    else
      return mLengthIPad;
  };
  double GetColSize(int col) const
  {
    if ((col == 0) || (col == mNcols - 1))
      return mWidthOPad;
    else
      return mWidthIPad;
  };

  double GetLengthRim() const { return mLengthRim; };
  double GetWidthRim() const { return mWidthRim; };
  double GetRowSpacing() const { return mRowSpacing; };
  double GetColSpacing() const { return mColSpacing; };
  double GetLengthOPad() const { return mLengthOPad; };
  double GetLengthIPad() const { return mLengthIPad; };
  double GetWidthOPad() const { return mWidthOPad; };
  double GetWidthIPad() const { return mWidthIPad; };
  double GetAnodeWireOffset() const { return mAnodeWireOffset; };
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
  TRDPadPlane(const TRDPadPlane& p);
  TRDPadPlane& operator=(const TRDPadPlane& p);

  ClassDefNV(TRDPadPlane, 1) //  TRD ROC pad plane
};
}
}
#endif
