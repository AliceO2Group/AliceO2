// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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
 * RespSimMat : class to access the response: probability to collect electron 
 * in MNPix*MNPix cells. 
 */
class RespSimMat
{
 public:
  static int constexpr NPix = 5;              /// side of quadrant (pixels) with non-0 response
  static int constexpr MatSize = NPix * NPix; /// number of pixels in the quadrant
  static int constexpr getNPix() { return NPix; }
  
  RespSimMat() = default;
  ~RespSimMat() = default;

  /// probability to find an electron in pixel ix,iy,iz 
  float getValue(int ix, int iy) const { return data[ix * NPix + iy]; }

  /// pointer on underlying array
  std::array<float, MatSize>* getArray() { return &data; }

  /// print values
  void print() const;

 private:
  std::array<float, MatSize> data;

  ClassDefNV(RespSimMat, 1)
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
  float mXMax = 14.62e-4;        /// upper boundary of X
  float mYMax = 13.44e-4;        /// upper boundary of Y
  float mZMin = 0.f;             /// lower boundary of Z
  float mZMax = 0.f;             /// upper boundary of Z
  float mStepInvX = 0;           /// inverse step of the X grid
  float mStepInvY = 0;           /// inverse step of the Y grid
  float mStepInvZ = 0;           /// inverse step of the Z grid
  std::vector<RespSimMat> mData; /// response data
  /// path to look for data file
  std::string mDataPath  = "$(O2_ROOT)/share/Detectors/ITSMFT/data/alpideResponseData";
  std::string mGridXName = "grid_list_x.txt";           /// name of the file with grid in X
  std::string mGridYName = "grid_list_y.txt";           /// name of the file with grid in Y
  std::string mXYDataFmt = "data_pixels_%.2f_%.2f.txt"; /// format to read the data for given X,Y

 public:
  AlpideSimResponse() = default;
  ~AlpideSimResponse() = default;

  void initData();

  const RespSimMat* getResponse(float x, float y, float z) const;
  static int constexpr getNPix() { return RespSimMat::getNPix(); }
  int getNBinX() const { return mNBinX; }
  int getNBinY() const { return mNBinY; }
  int getNBinZ() const { return mNBinZ; }
  float getXMax() const { return mXMax; }
  float getYMax() const { return mYMax; }
  float getZMin() const { return mZMin; }
  float getZMax() const { return mZMax; }
  float getStepX() const { return mStepInvX ? mStepInvX : 0.f; }
  float getStepY() const { return mStepInvY ? mStepInvY : 0.f; }
  float getStepZ() const { return mStepInvZ ? mStepInvZ : 0.f; }
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
  return xpos * mStepInvX + 0.5f;
}

//-----------------------------------------------------
inline int AlpideSimResponse::getYBin(float ypos) const
{
  /// get y bin w/o checking for over/under flow. ypos MUST be >=0
  return ypos * mStepInvY + 0.5f;
}

//-----------------------------------------------------
inline int AlpideSimResponse::getZBin(float zpos) const
{
  /// get z bin w/o checking for over/under flow. zpos is with respect of the beginning
  /// of epitaxial layer
  return (mZMax - zpos) * mStepInvZ; // hights bin
}
}
}

#endif
