// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AlpideSimResponse.cxx
/// \brief Implementation of the ITSMFT Alpide simulated response parametrization

#include "ITSMFTSimulation/AlpideSimResponse.h"
#include <TSystem.h>
#include <cstdio>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include "FairLogger.h"

using namespace o2::ITSMFT;
using namespace std;

ClassImp(o2::ITSMFT::AlpideSimResponse)
ClassImp(o2::ITSMFT::AlpideRespSimMat)

constexpr float micron2cm = 1e-4;

void AlpideSimResponse::initData()
{
  /*
   * read grid parameters and load data
   */
  if (mData.size()) {
    cout << "Object already initialized" << endl;
    print();
    return;
  }

  const float kTiny = 1e-6; // to check 0 values

  // if needed, append path with slash
  if (mDataPath.length() && mDataPath.back() != '/') {
    mDataPath.push_back('/');
  }
  mDataPath = gSystem->ExpandPathName(mDataPath.data());
  string inpfname = mDataPath + mGridColName;
  std::ifstream inpGrid;

  // read X grid
  inpGrid.open(inpfname, std::ifstream::in);
  if (inpGrid.fail()) {
    LOG(FATAL) << "Failed to open file " << inpfname << FairLogger::endl;
  }

  while (inpGrid >> mStepInvCol && inpGrid.good()) {
    mNBinCol++;
  }
  
  if (!mNBinCol || mStepInvCol < kTiny) {
    LOG(FATAL) << "Failed to read X(col) binning from " << inpfname << FairLogger::endl;
  }
  mMaxBinCol = mNBinCol-1;
  mStepInvCol = mMaxBinCol / mStepInvCol; // inverse of the X bin width
  inpGrid.close();

  // read Y grid
  inpfname = mDataPath + mGridRowName;
  inpGrid.open(inpfname, std::ifstream::in);
  if (inpGrid.fail()) {
    LOG(FATAL) << "Failed to open file " << inpfname << FairLogger::endl;
  }

  while (inpGrid >> mStepInvRow && inpGrid.good())
    mNBinRow++;
  if (!mNBinRow || mStepInvRow < kTiny) {
    LOG(FATAL) << "Failed to read Y(row) binning from " << inpfname << FairLogger::endl;
  }
  mMaxBinRow = mNBinRow - 1;
  mStepInvRow = mMaxBinRow / mStepInvRow; // inverse of the Row bin width
  inpGrid.close();

  // load response data
  int nz = 0;
  size_t cnt = 0;
  float val, gx, gy, gz;
  int lost, untrck, dead, nele;
  size_t dataSize = 0;
  mDptMax = -2.e9;
  mDptMin = 2.e9;
  const int npix = AlpideRespSimMat::getNPix();

  for (int ix = 0; ix < mNBinCol; ix++) {
    for (int iy = 0; iy < mNBinRow; iy++) {
      inpfname = composeDataName(ix, iy);
      inpGrid.open(inpfname, std::ifstream::in);
      if (inpGrid.fail()) {
        LOG(FATAL) << "Failed to open file " << inpfname << FairLogger::endl;
      }
      inpGrid >> nz;
      if (cnt == 0) {
        mNBinDpt = nz;
        dataSize = mNBinCol * mNBinRow * mNBinDpt;
        mData.reserve(dataSize); // reserve space for data
      } else if (nz != mNBinDpt) {
        LOG(FATAL) << "Mismatch in Nz slices of bin X(col): " << ix << " Y(row): " << iy
                   << " wrt bin 0,0. File " << inpfname << FairLogger::endl;
      }

      // load data
      for (int iz = 0; iz < nz; iz++) {
        AlpideRespSimMat mat;

        std::array<float, AlpideRespSimMat::MatSize>* arr = mat.getArray();
        for (int ip = 0; ip < npix * npix; ip++) {
          inpGrid >> val;
          (*arr)[ip] = val;
          cnt++;
        }
        inpGrid >> lost >> dead >> untrck >> nele >> gx >> gy >> gz;

        if (inpGrid.bad()) {
          LOG(FATAL) << "Failed reading data for depth(Z) slice " << iz << " from "
                     << inpfname << FairLogger::endl;
        }
        if (!nele) {
          LOG(FATAL) << "Wrong normalization Nele=" << nele << "for  depth(Z) slice "
                     << iz << " from " << inpfname
                     << FairLogger::endl;
        }

        if (mDptMax < -1e9) mDptMax = gz;
        if (mDptMin > gz)   mDptMin = gz;

        // normalize
        float norm = 1.f/nele;
        for (int ip = npix * npix; ip--;) (*arr)[ip] *= norm;
        mData.push_back(mat); // store in the final container
      }                       // loop over z

      inpGrid.close();

    } // loop over y
  }   // loop over x

  // final check
  if (dataSize != mData.size()) {
    LOG(FATAL) << "Mismatch between expected " << dataSize << " and loaded " << mData.size()
               << " number of bins" << FairLogger::endl;
  }

  // normalize Dpt boundaries

  mStepInvCol /= micron2cm;
  mStepInvRow /= micron2cm;

  mDptMin *= micron2cm;
  mDptMax *= micron2cm;
  mStepInvDpt = (mNBinDpt - 1) / (mDptMax - mDptMin);
  mDptMin -= 0.5 / mStepInvDpt;
  mDptMax += 0.5 / mStepInvDpt;
  mDptShift = 0.5*(mDptMax+mDptMin);
  print();
}

//-----------------------------------------------------
void AlpideSimResponse::print() const
{
  /*
   * print itself
   */
  printf("Alpide response object of %zu matrices to map chagre in xyz to %dx%d pixels\n",
         mData.size(), getNPix(),getNPix());
  printf("X(col) range: %+e : %+e | step: %e | Nbins: %d\n", 0.f, mColMax, 1.f / mStepInvCol, mNBinCol);
  printf("Y(row) range: %+e : %+e | step: %e | Nbins: %d\n", 0.f, mRowMax, 1.f / mStepInvRow, mNBinRow);
  printf("Z(dpt) range: %+e : %+e | step: %e | Nbins: %d\n", mDptMin, mDptMax, 1.f / mStepInvDpt, mNBinDpt);
}

//-----------------------------------------------------
string AlpideSimResponse::composeDataName(int colBin, int rowBin)
{
  /*
   * compose the file-name to read data for bin colBin,rowBin
   */

  // ugly but safe way to compose the file name
  float vcol = colBin / mStepInvCol, vrow = rowBin / mStepInvRow;
  size_t size = snprintf(nullptr, 0, mColRowDataFmt.data(), vcol, vrow) + 1;
  unique_ptr<char[]> tmp(new char[size]);
  snprintf(tmp.get(), size, mColRowDataFmt.data(), vcol, vrow);
  return mDataPath + string(tmp.get(), tmp.get() + size - 1);
}

//____________________________________________________________
bool AlpideSimResponse::getResponse(float vRow, float vCol, float vDepth, AlpideRespSimMat& dest) const
{
  /*
   * get linearized NPix*NPix matrix for response at point vRow(sensor local X, along row)
   * vCol(sensor local Z, along columns) and vDepth (sensor local Y, i.e. depth)
   */
  if (!mNBinDpt) {
    LOG(FATAL) << "response object is not initialized" << FairLogger::endl;
  }
  bool flipCol = false, flipRow = false;
  if (vDepth < mDptMin || vDepth > mDptMax) return false;
  if (vCol < 0) {
    vCol = -vCol;
    flipCol = true;
  }
  if (vCol > mColMax) return false;
  if (vRow < 0) {
    vRow = -vRow;
    flipRow = true;
  }
  if (vRow > mRowMax) return false;

  size_t bin = getDepthBin(vDepth) + mNBinDpt * (getRowBin(vRow) + mNBinRow * getColBin(vCol));
  if (bin >= mData.size()) {
    // this should not happen
    LOG(FATAL) << "requested bin " << bin << "row/col/depth: " << getRowBin(vRow) << ":" << getColBin(vCol) 
               << ":" << getDepthBin(vDepth) << ")" <<">= maxBin " << mData.size()
               << " for X(row)=" << vRow << " Z(col)=" << vCol << " Y(depth)=" << vDepth << FairLogger::endl;
  }
  // printf("bin %d %d %d\n",getColBin(vCol),getRowBin(vRow),getDepthBin(vDepth));
  //  return &mData[bin];
  dest.adopt( mData[bin], flipRow, flipCol);
  return true;
}

//____________________________________________________________
const AlpideRespSimMat* AlpideSimResponse::getResponse(float vRow, float vCol, float vDepth, bool& flipRow, bool& flipCol) const
{
  /*
   * get linearized NPix*NPix matrix for response at point vRow(sensor local X, along row)
   * vCol(sensor local Z, along columns) and vDepth (sensor local Y, i.e. depth)
   */
  if (!mNBinDpt) {
    LOG(FATAL) << "response object is not initialized" << FairLogger::endl;
  }
  if (vDepth < mDptMin || vDepth > mDptMax) return nullptr;
  if (vCol < 0) {
    vCol = -vCol;
    flipCol = true;
  }
  else {
    flipCol = false;
  }
  if (vCol > mColMax) return nullptr;
  if (vRow < 0) {
    vRow = -vRow;
    flipRow = true;
  }
  else {
    flipRow = false;
  }
  if (vRow > mRowMax) return nullptr;

  size_t bin = getDepthBin(vDepth) + mNBinDpt * (getRowBin(vRow) + mNBinRow * getColBin(vCol));
  if (bin >= mData.size()) {
    // this should not happen
    LOG(FATAL) << "requested bin " << bin << "row/col/depth: " << getRowBin(vRow) << ":" << getColBin(vCol) 
               << ":" << getDepthBin(vDepth) << ")" <<">= maxBin " << mData.size()
               << " for X(row)=" << vRow << " Z(col)=" << vCol << " Y(depth)=" << vDepth << FairLogger::endl;
  }
  return &mData[bin];

}

//__________________________________________________
void AlpideRespSimMat::print(bool flipRow,bool flipCol) const
{
  /*
   * print the response matrix
   */
  for (int iRow = 0; iRow < NPix; iRow++) {
    for (int iCol = 0; iCol < NPix; iCol++) {
      printf("%+e ", getValue(iRow, iCol, flipRow,flipCol));
    }
    printf("\n");
  }
}
