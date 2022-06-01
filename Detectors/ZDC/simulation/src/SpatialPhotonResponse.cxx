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

#include "ZDCSimulation/SpatialPhotonResponse.h"
#include "ZDCBase/Constants.h"
#include <cmath>
#include <iostream>

using namespace o2::zdc;

SpatialPhotonResponse::SpatialPhotonResponse(int Nx, int Ny, double lowerx,
                                             double lowery, double lengthx, double lengthy) : mNx{Nx},
                                                                                              mNy{Ny},
                                                                                              mLxOfCell{lengthx / Nx},
                                                                                              mLyOfCell{lengthy / Ny},
                                                                                              mInvLxOfCell{1. / mLxOfCell},
                                                                                              mInvLyOfCell{1. / mLyOfCell},
                                                                                              mLowerX{lowerx},
                                                                                              mLowerY{lowery}
{
  mImageData.resize(Nx);
  for (int x = 0; x < Nx; ++x) {
    mImageData[x].resize(Ny);
  }
  // now the image should be null initialized
  std::cout << " Image initialized with " << mNx << " x " << mNy << " pixels ";
}

void SpatialPhotonResponse::addPhoton(double x, double y, int nphotons)
{
  const int xpixel = (int)(std::floor((x - mLowerX) * mInvLxOfCell));
  const int ypixel = (int)(std::floor((y - mLowerY) * mInvLyOfCell));
  if (nphotons < 0) {
    std::cerr << "negative photon number\n";
    return;
  }
  if (xpixel < 0 || xpixel >= mNx) {
    std::cerr << "X-PIXEL OUT OF RANGE " << xpixel << " " << x << " , " << y << "\n";
    return;
  }
  if (ypixel < 0 || ypixel >= mNy) {
    std::cerr << "Y-PIXEL OUT OF RANGE " << ypixel << " " << x << " , " << y << "\n";
    return;
  }
  mImageData[xpixel][ypixel] += nphotons;
  mPhotonSum += nphotons;
}

void SpatialPhotonResponse::addPhotonByPixel(int xpixel, int ypixel, int nphotons)
{
  mImageData[xpixel][ypixel] += nphotons;
  mPhotonSum += nphotons;
}

// will print pixel 0 == (0,0) at the lower left corner
void SpatialPhotonResponse::printToScreen() const
{
  std::cout << "Response START " << mPhotonSum << " ----\n";
  for (int y = mNy - 1; y >= 0; --y) {
    {
      for (int x = 0; x < mNx; ++x) {
        const auto val = mImageData[x][y];
        if (val < 0) {
          std::cerr << "SHIT\n";
        }
        std::cout << ((val < 10) ? std::to_string(val) : "x");
      }
      std::cout << "\n";
    }
  }
  std::cout << "Response END ----\n";
}

void SpatialPhotonResponse::reset()
{
  mPhotonSum = 0;
  for (int x = 0; x < mNx; ++x) {
    {
      for (int y = 0; y < mNy; ++y) {
        mImageData[x][y] = 0;
      }
    }
  }
  mTime = 0;
  mDetectorID = -1;
}

std::array<int, 5> SpatialPhotonResponse::getPhotonsPerChannel() const
{
  std::array<int, 5> photonsum = {0, 0, 0, 0, 0};
  if (mPhotonSum == 0) {
    return photonsum;
  }

  if (mDetectorID == -1) {
    std::cerr << "SpatialPhotonResponse has no detectorID ";
    return photonsum;
  }

  auto determineChannel = [](int detector, int x, int y, int Nx, int Ny) {
    if ((x + y) % 2 == 0) {
      return (int)ChannelTypeZNP::Common;
    }

    if (detector == DetectorID::ZNA || detector == DetectorID::ZNC) {
      if (x < Nx / 2) {
        if (y < Ny / 2) {
          return (int)ChannelTypeZNP::Ch1;
        } else {
          return (int)ChannelTypeZNP::Ch3;
        }
      } else {
        if (y >= Ny / 2) {
          return (int)ChannelTypeZNP::Ch4;
        } else {
          return (int)ChannelTypeZNP::Ch2;
        }
      }
    }

    if (detector == DetectorID::ZPA || detector == DetectorID::ZPC) {
      auto i = (int)(4.f * x / Nx);
      return (int)(i + 1);
    }
    return -1;
  };

  int sum = 0;
  for (int x = 0; x < mNx; ++x) {
    // loop over y = rows
    for (int y = 0; y < mNy; ++y) {
      // get channel
      int channel = determineChannel(mDetectorID, x, y, mNx, mNy);
      photonsum[channel] += mImageData[x][y];
      sum += 0;
    }
  }
  // assert (mPhotonSum == photonsum[0] + photonsum[1] + photonsum[2] + photonsum[3] + photonsum[4]);

  return photonsum;
}

void SpatialPhotonResponse::printErrMsg(const char* str) const
{
  std::cerr << str << "\n";
}
