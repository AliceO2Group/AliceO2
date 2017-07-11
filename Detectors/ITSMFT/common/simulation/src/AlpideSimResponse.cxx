// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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
ClassImp(o2::ITSMFT::RespSimMat)

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
  string inpfname = mDataPath + mGridXName;
  std::ifstream inpGrid;

  // read X grid
  inpGrid.open(inpfname, std::ifstream::in);
  if (inpGrid.fail()) {
    LOG(FATAL) << "Failed to open file " << inpfname << FairLogger::endl;
  }

  while (inpGrid >> mStepInvX && inpGrid.good()) {
    mNBinX++;
  }
  
  if (!mNBinX || mStepInvX < kTiny) {
    LOG(FATAL) << "Failed to read X binning from " << inpfname << FairLogger::endl;
  }
  mStepInvX = (mNBinX - 1) / mStepInvX; // inverse of the X bin width
  inpGrid.close();

  // read Y grid
  inpfname = mDataPath + mGridYName;
  inpGrid.open(inpfname, std::ifstream::in);
  if (inpGrid.fail()) {
    LOG(FATAL) << "Failed to open file " << inpfname << FairLogger::endl;
  }

  while (inpGrid >> mStepInvY && inpGrid.good())
    mNBinY++;
  if (!mNBinY || mStepInvY < kTiny) {
    LOG(FATAL) << "Failed to read Y binning from " << inpfname << FairLogger::endl;
  }
  mStepInvY = (mNBinY - 1) / mStepInvY; // inverse of the Y bin width
  inpGrid.close();

  // load response data
  int nz = 0;
  size_t cnt = 0;
  float val, gx, gy, gz;
  int lost, untrck, dead, nele;
  size_t dataSize = 0;
  mZMax = -2.e9;
  mZMin = 2.e9;
  const int npix = RespSimMat::getNPix();

  for (int ix = 0; ix < mNBinX; ix++) {
    for (int iy = 0; iy < mNBinY; iy++) {
      inpfname = composeDataName(ix, iy);
      inpGrid.open(inpfname, std::ifstream::in);
      if (inpGrid.fail()) {
        LOG(FATAL) << "Failed to open file " << inpfname << FairLogger::endl;
      }
      inpGrid >> nz;
      if (cnt == 0) {
        mNBinZ = nz;
        dataSize = mNBinX * mNBinY * mNBinZ;
        mData.reserve(dataSize); // reserve space for data
      } else if (nz != mNBinZ) {
        LOG(FATAL) << "Mismatch in Nz slices of bin X: " << ix << " Y: " << iy
		   << " wrt bin 0,0. File " << inpfname << FairLogger::endl;
      }

      // load data
      for (int iz = 0; iz < nz; iz++) {
        RespSimMat mat;

        std::array<float, RespSimMat::MatSize>* arr = mat.getArray();
        for (int ip = 0; ip < npix * npix; ip++) {
          inpGrid >> val;
          (*arr)[ip] = val;
          cnt++;
        }
        inpGrid >> lost >> dead >> untrck >> nele >> gx >> gy >> gz;

        if (inpGrid.bad()) {
          LOG(FATAL) << "Failed reading data for Z slice " << iz << " from "
		     << inpfname << FairLogger::endl;
        }
        if (!nele) {
          LOG(FATAL) << "Wrong normalization Nele=" << nele << "for  Z slice "
		     << iz << " from " << inpfname
		     << FairLogger::endl;
        }

        if (mZMax < -1e9) mZMax = gz;
        if (mZMin > gz)   mZMin = gz;

        // normalize
        float norm = 1. / nele;
        for (int ip = 0; ip < npix * npix; ip++) (*arr)[ip] *= norm;
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

  // normalize Z boundaries

  mStepInvX /= micron2cm;
  mStepInvY /= micron2cm;

  mZMin *= micron2cm;
  mZMax *= micron2cm;
  mStepInvZ = (mNBinZ - 1) / (mZMax - mZMin);
  mZMin -= 0.5 / mStepInvZ;
  mZMax += 0.5 / mStepInvZ;

  print();
}

//-----------------------------------------------------
void AlpideSimResponse::print() const
{
  /*
   * print itself
   */
  printf("Alpide response object of %zu matrices to map chagre in XYZ to %dx%d pixels\n",
	 mData.size(), getNPix(),getNPix());
  printf("X range: %+e : %+e | step: %e | Nbins: %d\n", 0.f, mXMax, 1.f / mStepInvX, mNBinX);
  printf("Y range: %+e : %+e | step: %e | Nbins: %d\n", 0.f, mYMax, 1.f / mStepInvY, mNBinY);
  printf("Z range: %+e : %+e | step: %e | Nbins: %d\n", mZMin, mZMax, 1.f / mStepInvZ, mNBinZ);
}

//-----------------------------------------------------
string AlpideSimResponse::composeDataName(int xbin, int ybin)
{
  /*
   * compose the file-name to read data for bin xbin,ybin
   */

  // ugly but safe way to compose the file name
  float x = xbin / mStepInvX, y = ybin / mStepInvY;
  size_t size = snprintf(nullptr, 0, mXYDataFmt.data(), x, y) + 1;
  unique_ptr<char[]> tmp(new char[size]);
  snprintf(tmp.get(), size, mXYDataFmt.data(), x, y);
  return mDataPath + string(tmp.get(), tmp.get() + size - 1);
}

//-----------------------------------------------------
const RespSimMat* AlpideSimResponse::getResponse(float x, float y, float z) const
{
  /*
   * get linearized NPix*NPix matrix for response at point x,y,z
   */
  if (!mNBinZ) {
    LOG(FATAL) << "response object is not initialized" << FairLogger::endl;
  }
  if (z < mZMin || z > mZMax) return nullptr;
  if (x < 0) x = -x;
  if (x > mXMax) return nullptr;
  if (y < 0) y = -y;
  if (y > mYMax) return nullptr;

  size_t bin = getZBin(z) + mNBinZ * (getYBin(y) + mNBinY * getXBin(x));
  if (bin >= mData.size()) {
    // this should not happen
    LOG(FATAL) << "requested bin " << bin << ">= maxBin " << mData.size() << "for X="
	       << x << " Y=" << y << " Z=" << z << FairLogger::endl;
  }
  return &mData[bin];
}

//__________________________________________________
void RespSimMat::print() const
{
  /*
   * print the response matrix
   */
  for (int ix = 0; ix < NPix; ix++) {
    for (int iy = 0; iy < NPix; iy++) {
      printf("%+e ", getValue(ix, iy));
    }
    printf("\n");
  }
}
