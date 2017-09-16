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
namespace ITSMFT
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

  void adopt(const AlpideRespSimMat& src, bool flipX=false, bool flipY=false)
  {
    // copy constructor with option of channels flipping
    for (int i=NPix;i--;) {
      for (int j=NPix;j--;) {
	int bDest = (flipY ? NPix-1-i : i)*NPix + (flipX ? NPix-1-j : j);
	data[bDest] = src.data[i*NPix+j];
      }
    }
  }
  
  /// probability to find an electron in pixel ix,iy,iz 
  float getValue(int ix, int iy) const { return data[ix * NPix + iy]; }
  float getValue(int ix, int iy, bool flipX, bool flipY) const
  {
    int bin = (flipY ? NPix-1-ix : ix)*NPix + (flipX ? NPix-1-iy : iy);
    return data[bin];
  }

  /// pointer on underlying array
  std::array<float, MatSize>* getArray() { return &data; }

  /// print values
  void print() const;

 private:
  std::array<float, MatSize> data;

  ClassDefNV(AlpideRespSimMat, 1)
};


/*
 * AlpideSimResponse: container for Alpide simulates parameterized response matrices
 * Based on the Miljenko Šuljić standalone code and needs as an input text matrices
 * from simulation. 
 * Provides for the electron injected to point X,Y (with respect to pixel center) 
 * and Z (with respect to epitaxial layer inner serface!!! i.e. touching the substrate) 
 * the probability to be collected in every of 5x5 pixels with reference pixel in the
 * center. 
 */

class AlpideSimResponse
{

 private:
  int getXBin(float xpos) const;
  int getYBin(float ypos) const;
  int getZBin(float zpos) const;
  std::string composeDataName(int xbin, int ybin);

  int mNBinX = 0;                /// number of bins in X
  int mNBinY = 0;                /// number of bins in Y
  int mNBinZ = 0;                /// number of bins in Z (sensor dept)
  int mMaxBinX = 0;              /// max allowed Xb (to avoid subtraction)
  int mMaxBinY = 0;              /// max allowed Yb (to avoid subtraction)  
  float mXMax = 14.62e-4;        /// upper boundary of X
  float mYMax = 13.44e-4;        /// upper boundary of Y
  float mZMin = 0.f;             /// lower boundary of Z
  float mZMax = 0.f;             /// upper boundary of Z
  float mStepInvX = 0;           /// inverse step of the X grid
  float mStepInvY = 0;           /// inverse step of the Y grid
  float mStepInvZ = 0;           /// inverse step of the Z grid
  std::vector<AlpideRespSimMat> mData; /// response data
  /// path to look for data file
  std::string mDataPath  = "$(O2_ROOT)/share/Detectors/ITSMFT/data/alpideResponseData";
  std::string mGridXName = "grid_list_x.txt";           /// name of the file with grid in X
  std::string mGridYName = "grid_list_y.txt";           /// name of the file with grid in Y
  std::string mXYDataFmt = "data_pixels_%.2f_%.2f.txt"; /// format to read the data for given X,Y

 public:
  AlpideSimResponse() = default;
  ~AlpideSimResponse() = default;

  void initData();

  bool getResponse(float x, float y, float z, AlpideRespSimMat& dest) const;
  const AlpideRespSimMat* getResponse(float x, float y, float z, bool& flipX, bool& flipY) const;
  static int constexpr getNPix() { return AlpideRespSimMat::getNPix(); }
  int getNBinX() const { return mNBinX; }
  int getNBinY() const { return mNBinY; }
  int getNBinZ() const { return mNBinZ; }
  float getXMax() const { return mXMax; }
  float getYMax() const { return mYMax; }
  float getZMin() const { return mZMin; }
  float getZMax() const { return mZMax; }
  float getStepX() const { return mStepInvX ? 1./mStepInvX : 0.f; }
  float getStepY() const { return mStepInvY ? 1./mStepInvY : 0.f; }
  float getStepZ() const { return mStepInvZ ? 1./mStepInvZ : 0.f; }
  void setDataPath(const std::string pth) { mDataPath = pth; }
  void setGridXName(const std::string nm) { mGridXName = nm; }
  void setGridYName(const std::string nm) { mGridYName = nm; }
  void setXYDataFmt(const std::string nm) { mXYDataFmt = nm; }
  const std::string& getDataPath() const { return mDataPath; }
  const std::string& getGridXName() const { return mGridXName; }
  const std::string& getGridYName() const { return mGridYName; }
  const std::string& getXYDataFmt() const { return mXYDataFmt; }
  void print() const;

  ClassDef(AlpideSimResponse, 1)
};

//-----------------------------------------------------
inline int AlpideSimResponse::getXBin(float xpos) const
{
  /// get x bin w/o checking for over/under flow. xpos MUST be >=0
  int ix = xpos * mStepInvX + 0.5f;
  return ix<mNBinX ? ix:mMaxBinX;
}

//-----------------------------------------------------
inline int AlpideSimResponse::getYBin(float ypos) const
{
  /// get y bin w/o checking for over/under flow. ypos MUST be >=0
  int iy = ypos * mStepInvY + 0.5f;
  return iy<mNBinY ? iy:mMaxBinY;
}

//-----------------------------------------------------
inline int AlpideSimResponse::getZBin(float zpos) const
{
  /// get z bin w/o checking for over/under flow. zpos is with respect of the beginning
  /// of epitaxial layer
  int iz = (mZMax - zpos) * mStepInvZ;
  return iz<0 ? 0:iz; // depth bin
}
}
}

#endif
