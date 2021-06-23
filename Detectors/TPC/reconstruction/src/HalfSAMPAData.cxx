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

/// \file HalfSAMPAData.cxx
/// \author Sebastian Klewin

#include "TPCReconstruction/HalfSAMPAData.h"
#include "FairLogger.h"

using namespace o2::tpc;

HalfSAMPAData::HalfSAMPAData()
  : HalfSAMPAData(-1, true)
{
}

HalfSAMPAData::HalfSAMPAData(int id, bool low)
  : mID(id), mLow(low)
//  , mData(16,0)
{
}

HalfSAMPAData::HalfSAMPAData(int id, bool low, std::array<short, 16>& data)
  : mID(id), mLow(low)
{
  //  if (data.size() != 16)
  //    LOG(ERROR) << "Vector does not contain 16 elements.";

  mData = data;
}

HalfSAMPAData::~HalfSAMPAData() = default;

std::ostream& HalfSAMPAData::Print(std::ostream& output) const
{
  //  for (int i = mLow ? 0 : 16 ; i < (mLow ? 16 : 32); ++i)
  //  {
  //    output << "Channel " << i << ": " << mData[i] << std::endl;
  //  }

  output
    << mData[0] << "\t"
    << mData[1] << "\t"
    << mData[2] << "\t"
    << mData[3] << "\t"
    << mData[4] << "\t"
    << mData[5] << "\t"
    << mData[6] << "\t"
    << mData[7] << "\t"
    << mData[8] << "\t"
    << mData[9] << "\t"
    << mData[10] << "\t"
    << mData[11] << "\t"
    << mData[12] << "\t"
    << mData[13] << "\t"
    << mData[14] << "\t"
    << mData[15];
  return output;
}
