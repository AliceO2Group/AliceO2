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

#include "TPCCalibration/IDCFourierTransformBase.h"
#include <fftw3.h>

void o2::tpc::IDCFourierTransformAggregator::setIDCs(IDCOne&& oneDIDCs, std::vector<unsigned int>&& integrationIntervalsPerTF)
{
  mIDCOne[mBufferIndex] = std::move(oneDIDCs);
  mIntegrationIntervalsPerTF[mBufferIndex] = std::move(integrationIntervalsPerTF);
  mBufferIndex = !mBufferIndex;
}

void o2::tpc::IDCFourierTransformAggregator::setIDCs(const IDCOne& oneDIDCs, const std::vector<unsigned int>& integrationIntervalsPerTF)
{
  mIDCOne[mBufferIndex] = oneDIDCs;
  mIntegrationIntervalsPerTF[mBufferIndex] = integrationIntervalsPerTF;
  mBufferIndex = !mBufferIndex;
}

std::vector<unsigned int> o2::tpc::IDCFourierTransformAggregator::getLastIntervals(o2::tpc::Side) const
{
  std::vector<unsigned int> endIndex;
  endIndex.reserve(mTimeFrames);
  endIndex.emplace_back(0);
  for (unsigned int interval = 1; interval < mTimeFrames; ++interval) {
    endIndex.emplace_back(endIndex[interval - 1] + mIntegrationIntervalsPerTF[!mBufferIndex][interval]);
  }
  return endIndex;
}

std::vector<float> o2::tpc::IDCFourierTransformAggregator::getExpandedIDCOne(const o2::tpc::Side side) const
{
  std::vector<float> val1DIDCs = mIDCOne[!mBufferIndex].mIDCOne[side]; // just copy the elements
  if (useLastBuffer()) {
    val1DIDCs.insert(val1DIDCs.begin(), mIDCOne[mBufferIndex].mIDCOne[side].end() - mRangeIDC + mIntegrationIntervalsPerTF[!mBufferIndex][0], mIDCOne[mBufferIndex].mIDCOne[side].end());
  }
  return val1DIDCs;
}

float* o2::tpc::IDCFourierTransformAggregator::allocMemFFTW(const o2::tpc::Side side) const
{
  const unsigned int nElementsLastBuffer = useLastBuffer() ? mRangeIDC - mIntegrationIntervalsPerTF[!mBufferIndex][0] : 0;
  const unsigned int nElementsAll = mIDCOne[!mBufferIndex].getNIDCs(side) + nElementsLastBuffer;
  return fftwf_alloc_real(nElementsAll);
}
