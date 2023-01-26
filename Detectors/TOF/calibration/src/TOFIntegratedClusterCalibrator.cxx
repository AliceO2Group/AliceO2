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

/// \file TOFIntegratedClusterCalibrator.cxx
///
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jan 21, 2023

#include "TOFCalibration/TOFIntegratedClusterCalibrator.h"
#include "CommonUtils/TreeStreamRedirector.h"

namespace o2
{
namespace tof
{
using Slot = o2::calibration::TimeSlot<o2::tof::TOFIntegratedClusters>;

void TOFIntegratedClusters::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

void TOFIntegratedClusters::fill(const o2::calibration::TFType tfID, const std::vector<float>& iTOFCNcl, const std::vector<float>& iTOFCqTot)
{
  // check if size is same
  if (iTOFCNcl.size() != iTOFCqTot.size()) {
    LOGP(warning, "Received data with different size. iTOFCNcl.size {}   iTOFCqTot.size {}", iTOFCNcl.size(), iTOFCqTot.size());
    return;
  }

  if (iTOFCNcl.empty()) {
    LOGP(info, "Empty data received. Returning");
    return;
  }

  // initialize when first data is received
  if (mInitialize) {
    initData(iTOFCqTot);
  }

  // check if all data is already received (this should never happen)
  if (mRemainingData == 0) {
    LOGP(warning, "All packages already received. Returning");
    return;
  }

  if (iTOFCNcl.size() != mNValuesPerTF) {
    LOGP(info, "Received data with size {} expected size {} (expected size can be ignored if merging was performed)", iTOFCNcl.size(), mNValuesPerTF);
  }

  const unsigned int posIndex = (tfID - mTFFirst) * mNValuesPerTF;
  if (posIndex + iTOFCNcl.size() > mCurrents.mITOFCNCl.size()) {
    LOGP(warning, "Index for TF {} is larger {} than expected max index {} with {} values per package", tfID, posIndex, mCurrents.mITOFCNCl.size(), mNValuesPerTF);
    return;
  }

  // copy data to buffer
  std::copy(iTOFCNcl.begin(), iTOFCNcl.end(), mCurrents.mITOFCNCl.begin() + posIndex);
  std::copy(iTOFCqTot.begin(), iTOFCqTot.end(), mCurrents.mITOFCQ.begin() + posIndex);

  mRemainingData -= iTOFCNcl.size();
  LOGP(debug, "Processed TF {} at index {} with first TF {} and {} expected currents per TF. Remaining data {}", tfID, posIndex, mTFFirst, mNValuesPerTF, mRemainingData);
}

void TOFIntegratedClusters::merge(const TOFIntegratedClusters* prev)
{
  LOGP(info, "Printing last object...");
  prev->print();
  LOGP(info, "Printing current object...");
  print();

  const auto tfMin = std::min(mTFFirst, prev->mTFFirst);
  const auto tfMax = std::max(mTFLast, prev->mTFLast);

  if (prev->mInitialize) {
    // if last object was not initialized adjust tf range
    mTFFirst = tfMin;
    mTFLast = tfMax;
    if (!mInitialize) {
      // current buffer is already initialized just add empty values in front of buffer
      LOGP(info, "Adding dummy data to front");
      std::vector<float> vecTmp(mNValuesPerTF * ((prev->mTFLast - prev->mTFFirst) + 1), 0);
      mCurrents.mITOFCNCl.insert(mCurrents.mITOFCNCl.begin(), vecTmp.begin(), vecTmp.end());
      mCurrents.mITOFCQ.insert(mCurrents.mITOFCQ.begin(), vecTmp.begin(), vecTmp.end());
    }
    LOGP(info, "Do not merge last object since it was not initialized. Adjusting TF range:");
    print();
    return;
  }

  // check if current object needs to be initialized
  if (mInitialize) {
    // init with dummy vector
    initData(std::vector<float>(prev->mNValuesPerTF));
  }

  // creating new temporary object with the range of current and last object
  const unsigned int totalTFs = (tfMax - tfMin) + 1; // add 1 since first<= t <=last
  const unsigned int nTotal = mNValuesPerTF * totalTFs;
  TOFIntegratedClusters dataTmp(tfMin, tfMax);
  dataTmp.mInitialize = false; // do no initialization as it is done manually
  dataTmp.mNValuesPerTF = mNValuesPerTF;
  dataTmp.mRemainingData = -1;
  dataTmp.mCurrents.mITOFCNCl.resize(nTotal);
  dataTmp.mCurrents.mITOFCQ.resize(nTotal);

  // fill buffered values to tmp objects
  dataTmp.fill(mTFFirst, mCurrents.mITOFCNCl, mCurrents.mITOFCQ);
  dataTmp.fill(prev->mTFFirst, prev->mCurrents.mITOFCNCl, prev->mCurrents.mITOFCQ);

  dataTmp.mRemainingData = mRemainingData;
  *this = std::move(dataTmp);
  LOGP(info, "Merging done", totalTFs);
  print();
}

void TOFIntegratedClusters::initData(const std::vector<float>& vec)
{
  mInitialize = false;
  const unsigned int totalTFs = (mTFLast - mTFFirst) + 1; // add 1 since first<= t <=last
  mNValuesPerTF = vec.size();
  const unsigned int nTotal = mNValuesPerTF * totalTFs;
  mCurrents.mITOFCNCl.resize(nTotal);
  mCurrents.mITOFCQ.resize(nTotal);
  mRemainingData = nTotal;
  LOGP(info, "Init: Expecting {} packages with {} values per package with {} total values", mRemainingData, mNValuesPerTF, nTotal);
}

void TOFIntegratedClusters::dumpToTree(const char* outFileName)
{
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream << "tree"
           << "currents=" << mCurrents
           << "firstTF=" << mTFFirst
           << "lastTF=" << mTFLast
           << "remainingData=" << mRemainingData
           << "valuesPerTF=" << mNValuesPerTF
           << "\n";
}

void TOFIntegratedClusterCalibrator::initOutput()
{
  mIntervals.clear();
  mCalibs.clear();
  mTimeIntervals.clear();
}

void TOFIntegratedClusterCalibrator::finalizeSlot(Slot& slot)
{
  const TFType startTF = slot.getTFStart();
  const TFType endTF = slot.getTFEnd();
  LOGP(info, "Finalizing slot {} <= TF <= {}", startTF, endTF);

  auto& TOFIntegratedClusters = *slot.getContainer();
  if (mDebug) {
    TOFIntegratedClusters.dumpToFile(fmt::format("TOFIntegratedClusters_TF_{}_{}_TS_{}_{}.root", startTF, endTF, slot.getStartTimeMS(), slot.getEndTimeMS()).data());
  }
  mCalibs.emplace_back(TOFIntegratedClusters.getITOFCCurrents());
  mIntervals.emplace_back(startTF, endTF);
  mTimeIntervals.emplace_back(slot.getStartTimeMS(), slot.getEndTimeMS());
}

/// Creates new time slot
Slot& TOFIntegratedClusterCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<TOFIntegratedClusters>(tstart, tend));
  return slot;
}

} // end namespace tof
} // end namespace o2
