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

#include "ITS3Simulation/ChipSimResponse.h"
#include <vector>
#include <algorithm>

using namespace o2::its3;

ClassImp(o2::its3::ChipSimResponse);

void ChipSimResponse::initData(int tableNumber, std::string dataPath, const bool quiet)
{
  AlpideSimResponse::initData(tableNumber, dataPath, quiet);
  computeCentreFromData();
}

void ChipSimResponse::computeCentreFromData()
{
  const int npix = o2::itsmft::AlpideRespSimMat::getNPix();
  std::vector<float> zVec, effVec;
  zVec.reserve(mNBinDpt);
  effVec.reserve(mNBinDpt);

  for (int iz = 0; iz < mNBinDpt; ++iz) {
    int rev = mNBinDpt - 1 - iz;
    float z = mDptMin + iz / mStepInvDpt;
    float sum = 0.f;
    const auto& mat = mData[rev];
    for (int ix = 0; ix < npix; ++ix) {
      for (int iy = 0; iy < npix; ++iy) {
        sum += mat.getValue(ix, iy);
      }
    }
    zVec.push_back(z);
    effVec.push_back(sum);
  }

  struct Bin {
    float z0, z1, q0, q1, dq;
  };
  std::vector<Bin> bins;
  bins.reserve(zVec.size() - 1);

  float totQ = 0.f;
  for (size_t i = 0; i + 1 < zVec.size(); ++i) {
    float z0 = zVec[i], z1 = zVec[i + 1];
    float q0 = effVec[i], q1 = effVec[i + 1];
    float dq = 0.5f * (q0 + q1) * (z1 - z0);
    bins.push_back({z0, z1, q0, q1, dq});
    totQ += dq;
  }

  if (totQ <= 0.f) {
    mRespCentreDep = mDptMin;
    return;
  }

  float halfQ = 0.5f * totQ;
  float cumQ = 0.f;
  for (auto& b : bins) {
    if (cumQ + b.dq < halfQ) {
      cumQ += b.dq;
      continue;
    }
    float dz = b.z1 - b.z0;
    float slope = (b.q1 - b.q0) / dz;
    float disc = b.q0 * b.q0 - 2.f * slope * (cumQ - halfQ);

    float x;
    if (disc >= 0.f && std::abs(slope) > 1e-6f) {
      x = (-b.q0 + std::sqrt(disc)) / slope;
    } else {
      x = (halfQ - cumQ) / b.q0;
    }
    x = std::clamp(x, 0.f, dz);
    mRespCentreDep = b.z0 + x;
    return;
  }

  mRespCentreDep = mDptMax;
}
