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

/// \file IntegratedClusterCalibrator.cxx
///
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jan 21, 2023

#include "DetectorsCalibration/IntegratedClusterCalibrator.h"
#include "CommonUtils/TreeStreamRedirector.h"

namespace o2
{
namespace calibration
{

template <typename DataT>
void IntegratedClusters<DataT>::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

template <typename DataT>
void IntegratedClusters<DataT>::fill(const o2::calibration::TFType tfID, const DataT& currentsContainer)
{
  // check if size is same
  if (!currentsContainer.areSameSize()) {
    LOGP(warning, "Received data with different size. Returning");
    return;
  }

  if (currentsContainer.isEmpty()) {
    LOGP(info, "Empty data received. Returning");
    return;
  }

  // initialize when first data is received
  const unsigned int entries = currentsContainer.getEntries();
  if (mInitialize) {
    initData(entries);
  }

  // check if all data is already received (this should never happen)
  if (mRemainingData == 0) {
    LOGP(warning, "All packages already received. Returning");
    return;
  }

  if (entries != mNValuesPerTF) {
    LOGP(info, "Received data with size {} expected size {} (expected size can be ignored if merging was performed)", entries, mNValuesPerTF);
  }

  const unsigned int posIndex = (tfID - mTFFirst) * mNValuesPerTF;
  if (posIndex + entries > mCurrents.getEntries()) {
    LOGP(warning, "Index for TF {} is larger {} than expected max index {} with {} values per package", tfID, posIndex, mCurrents.getEntries(), mNValuesPerTF);
    return;
  }

  // copy data to buffer
  mCurrents.fill(posIndex, currentsContainer);

  mRemainingData -= entries;
  LOGP(debug, "Processed TF {} at index {} with first TF {} and {} expected currents per TF. Remaining data {}", tfID, posIndex, mTFFirst, mNValuesPerTF, mRemainingData);
}

template <typename DataT>
void IntegratedClusters<DataT>::merge(const IntegratedClusters* prev)
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
      const unsigned int nDummyValues = mNValuesPerTF * ((prev->mTFLast - prev->mTFFirst) + 1);
      mCurrents.insert(nDummyValues);
    }
    LOGP(info, "Do not merge last object since it was not initialized. Adjusting TF range:");
    print();
    return;
  }

  // check if current object needs to be initialized
  if (mInitialize) {
    // init with dummy vector
    initData(prev->mNValuesPerTF);
  }

  // creating new temporary object with the range of current and last object
  const unsigned int totalTFs = (tfMax - tfMin) + 1; // add 1 since first<= t <=last
  const unsigned int nTotal = mNValuesPerTF * totalTFs;
  IntegratedClusters dataTmp(tfMin, tfMax);
  dataTmp.mInitialize = false; // do no initialization as it is done manually
  dataTmp.mNValuesPerTF = mNValuesPerTF;
  dataTmp.mRemainingData = -1;
  dataTmp.mCurrents.resize(nTotal);

  // fill buffered values to tmp objects
  dataTmp.fill(mTFFirst, mCurrents);
  dataTmp.fill(prev->mTFFirst, prev->mCurrents);

  dataTmp.mRemainingData = mRemainingData;
  *this = std::move(dataTmp);
  LOGP(info, "Merging done", totalTFs);
  print();
}

template <typename DataT>
void IntegratedClusters<DataT>::initData(const unsigned int valuesPerTF)
{
  mInitialize = false;
  const unsigned int totalTFs = (mTFLast - mTFFirst) + 1; // add 1 since first<= t <=last
  mNValuesPerTF = valuesPerTF;
  const unsigned int nTotal = mNValuesPerTF * totalTFs;
  mCurrents.resize(nTotal);
  mRemainingData = nTotal;
  LOGP(info, "Init: Expecting {} packages with {} values per package with {} total values", mRemainingData, mNValuesPerTF, nTotal);
}

template <typename DataT>
void IntegratedClusters<DataT>::dumpToTree(const char* outFileName)
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

template <typename DataT>
void IntegratedClusterCalibrator<DataT>::initOutput()
{
  mIntervals.clear();
  mCalibs.clear();
  mTimeIntervals.clear();
}

template <typename DataT>
void IntegratedClusterCalibrator<DataT>::finalizeSlot(o2::calibration::TimeSlot<o2::calibration::IntegratedClusters<DataT>>& slot)
{
  const TFType startTF = slot.getTFStart();
  const TFType endTF = slot.getTFEnd();
  LOGP(info, "Finalizing slot {} <= TF <= {}", startTF, endTF);

  auto& integratedClusters = *slot.getContainer();
  if (mDebug) {
    integratedClusters.dumpToFile(fmt::format("IntegratedClusters_TF_{}_{}_TS_{}_{}.root", startTF, endTF, slot.getStartTimeMS(), slot.getEndTimeMS()).data());
  }
  mCalibs.emplace_back(std::move(integratedClusters).getCurrents());

  mIntervals.emplace_back(startTF, endTF);
  mTimeIntervals.emplace_back(slot.getStartTimeMS(), slot.getEndTimeMS());
}

/// Creates new time slot
template <typename DataT>
o2::calibration::TimeSlot<o2::calibration::IntegratedClusters<DataT>>& IntegratedClusterCalibrator<DataT>::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = this->getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<IntegratedClusters<DataT>>(tstart, tend));
  return slot;
}

template class IntegratedClusters<o2::tof::ITOFC>;
template class IntegratedClusterCalibrator<o2::tof::ITOFC>;

template class IntegratedClusters<o2::fit::IFT0C>;
template class IntegratedClusterCalibrator<o2::fit::IFT0C>;

template class IntegratedClusters<o2::fit::IFV0C>;
template class IntegratedClusterCalibrator<o2::fit::IFV0C>;

template class IntegratedClusters<o2::tpc::ITPCC>;
template class IntegratedClusterCalibrator<o2::tpc::ITPCC>;

} // end namespace calibration
} // end namespace o2
