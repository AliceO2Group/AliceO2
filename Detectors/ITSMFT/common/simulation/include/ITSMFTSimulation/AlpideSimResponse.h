// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AlpideSimResponse.h
/// \brief Definition of the ITSMFT Alpide simulated response parametrization

#ifndef ALICEO2_ITSSMFT_ALPIDESIMRESPONSE_H
#define ALICEO2_ITSSMFT_ALPIDESIMRESPONSE_H

#include <array>
#include <string>
#include <vector>
#include <Rtypes.h>

namespace o2
{
namespace itsmft
{
/*
 * AlpideRespSimMat : class to access the response: probability to collect electron 
 * in MNPix*MNPix cells. 
 */
class AlpideRespSimMat
{
 public:
  static int constexpr NPix = 5;              /// side of quadrant (pixels) with non-0 response
  static int constexpr MatSize = NPix * NPix; /// number of pixels in the quadrant
  static int constexpr getNPix() { return NPix; }

  AlpideRespSimMat() = default;
  ~AlpideRespSimMat() = default;

  void adopt(const AlpideRespSimMat& src, bool flipRow = false, bool flipCol = false)
  {
    // copy constructor with option of channels flipping
    for (int iRow = NPix; iRow--;) {
      int rw = flipRow ? NPix - 1 - iRow : iRow;
      for (int iCol = NPix; iCol--;) {
        int bDest = rw * NPix + (flipCol ? NPix - 1 - iCol : iCol);
        data[bDest] = src.data[iRow * NPix + iCol];
      }
    }
  }

  /// probability to find an electron in pixel ix,iy,iz
  float getValue(int iRow, int iCol) const { return data[iRow * NPix + iCol]; }
  float getValue(int iRow, int iCol, bool flipRow, bool flipCol) const
  {
    int bin = (flipRow ? NPix - 1 - iRow : iRow) * NPix + (flipCol ? NPix - 1 - iCol : iCol);
    return data[bin];
  }

  /// pointer on underlying array
  std::array<float, MatSize>* getArray() { return &data; }

  /// print values
  void print(bool flipRow = false, bool flipCol = false) const;

 private:
  std::array<float, MatSize> data;

  ClassDefNV(AlpideRespSimMat, 1);
};

/*
 * AlpideSimResponse: container for Alpide simulates parameterized response matrices
 * Based on the Miljenko Šuljić standalone code and needs as an input text matrices
 * from simulation. 
 * Provides for the electron injected to point X(columns direction),Y (rows direction) 
 * (with respect to pixel center) and Z (depth, with respect to epitaxial layer inner 
 * serface!!! i.e. touching the substrate) the probability to be collected in every 
 * of NPix*NPix pixels with reference pixel in the center. 
 */

class AlpideSimResponse
{

 private:
  int getColBin(float pos) const;
  int getRowBin(float pos) const;
  int getDepthBin(float pos) const;
  std::string composeDataName(int colBin, int rowBin);

  int mNBinCol = 0;                    /// number of bins in X(col direction)
  int mNBinRow = 0;                    /// number of bins in Y(row direction)
  int mNBinDpt = 0;                    /// number of bins in Z(sensor dept)
  int mMaxBinCol = 0;                  /// max allowed Xb (to avoid subtraction)
  int mMaxBinRow = 0;                  /// max allowed Yb (to avoid subtraction)
  float mColMax = 14.62e-4;            /// upper boundary of Col
  float mRowMax = 13.44e-4;            /// upper boundary of Row
  float mDptMin = 0.f;                 /// lower boundary of Dpt
  float mDptMax = 0.f;                 /// upper boundary of Dpt
  float mDptShift = 0.f;               /// shift of the depth center wrt 0
  float mStepInvCol = 0;               /// inverse step of the Col grid
  float mStepInvRow = 0;               /// inverse step of the Row grid
  float mStepInvDpt = 0;               /// inverse step of the Dpt grid
  std::vector<AlpideRespSimMat> mData; /// response data
  /// path to look for data file
  std::string mDataPath = "$(O2_ROOT)/share/Detectors/ITSMFT/data/alpideResponseData";
  std::string mGridColName = "grid_list_x.txt";             /// name of the file with grid in Col
  std::string mGridRowName = "grid_list_y.txt";             /// name of the file with grid in Row
  std::string mColRowDataFmt = "data_pixels_%.2f_%.2f.txt"; /// format to read the data for given Col,Row

 public:
  AlpideSimResponse() = default;
  ~AlpideSimResponse() = default;

  void initData();

  bool getResponse(float vRow, float vCol, float cDepth, AlpideRespSimMat& dest) const;
  const AlpideRespSimMat* getResponse(float vRow, float vCol, float vDepth, bool& flipRow, bool& flipCol) const;
  static int constexpr getNPix() { return AlpideRespSimMat::getNPix(); }
  int getNBinCol() const { return mNBinCol; }
  int getNBinRow() const { return mNBinRow; }
  int getNBinDepth() const { return mNBinDpt; }
  float getColMax() const { return mColMax; }
  float getRowMax() const { return mRowMax; }
  float getDepthMin() const { return mDptMin; }
  float getDepthMax() const { return mDptMax; }
  float getDepthShift() const { return mDptShift; }
  float getStepCol() const { return mStepInvCol ? 1. / mStepInvCol : 0.f; }
  float getStepRow() const { return mStepInvRow ? 1. / mStepInvRow : 0.f; }
  float getStepDepth() const { return mStepInvDpt ? 1. / mStepInvDpt : 0.f; }
  void setDataPath(const std::string pth) { mDataPath = pth; }
  void setGridColName(const std::string nm) { mGridColName = nm; }
  void setGridRowName(const std::string nm) { mGridRowName = nm; }
  void setColRowDataFmt(const std::string nm) { mColRowDataFmt = nm; }
  const std::string& getDataPath() const { return mDataPath; }
  const std::string& getGridColName() const { return mGridColName; }
  const std::string& getGridRowName() const { return mGridRowName; }
  const std::string& getColRowDataFmt() const { return mColRowDataFmt; }
  void print() const;

  ClassDef(AlpideSimResponse, 1);
};

//-----------------------------------------------------
inline int AlpideSimResponse::getColBin(float pos) const
{
  /// get column bin w/o checking for over/under flow. pos MUST be >=0
  int i = pos * mStepInvCol + 0.5f;
  return i < mNBinCol ? i : mMaxBinCol;
}

//-----------------------------------------------------
inline int AlpideSimResponse::getRowBin(float pos) const
{
  /// get row bin w/o checking for over/under flow. pos MUST be >=0
  int i = pos * mStepInvRow + 0.5f;
  return i < mNBinRow ? i : mMaxBinRow;
}

//-----------------------------------------------------
inline int AlpideSimResponse::getDepthBin(float pos) const
{
  /// get depth bin w/o checking for over/under flow. pos is with respect of the beginning
  /// of epitaxial layer
  int i = (mDptMax - pos) * mStepInvDpt;
  return i < 0 ? 0 : i; // depth bin
}

} // namespace itsmft
} // namespace o2

#endif
