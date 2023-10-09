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
  integratedClusters.setStartTime(slot.getStartTimeMS());
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

template class IntegratedClusters<o2::tpc::TimeSeriesITSTPC>;
template class IntegratedClusterCalibrator<o2::tpc::TimeSeriesITSTPC>;

template class IntegratedClusters<o2::fit::IFDDC>;
template class IntegratedClusterCalibrator<o2::fit::IFDDC>;

} // end namespace calibration

void tpc::TimeSeriesITSTPC::dumpToTree(const char* outFileName, const int nHBFPerTF)
{
  TFile f(outFileName, "RECREATE");
  TTree tree("timeSeries", "timeSeries");
  mTSTPC.dumpToTree(tree, "TPC_", nHBFPerTF);
  mTSITSTPC.dumpToTree(tree, "ITSTPC_", nHBFPerTF);
  mITSTPCAll.dumpToTree(tree, "ITSTPC_", nHBFPerTF, mTSTPC);
  mITSTPCStandalone.dumpToTree(tree, "ITS_SA_", nHBFPerTF, mTSTPC);
  mITSTPCAfterburner.dumpToTree(tree, "ITS_AB_", nHBFPerTF, mTSTPC);
  f.WriteObject(&tree, "timeSeries");
}

void tpc::ITSTPC_Matching::dumpToTree(TTree& tree, const char* prefix, const int nHBFPerTF, const TimeSeries& timeSeriesRef) const
{
  // adding matching efficiency
  const auto nValues = timeSeriesRef.mDCAr_A_Median.size();
  const int nPointsPerTF = timeSeriesRef.getNBins();
  const int nTotPoints = nValues / nPointsPerTF;

  const int nBinTypes = 4;
  for (int type = 0; type < nBinTypes; ++type) {
    std::array<int, nBinTypes> nBinsAllTypes{timeSeriesRef.mNBinsPhi, timeSeriesRef.mNBinsTgl, timeSeriesRef.mNBinsqPt, 1};
    std::array<std::string, nBinTypes> nBinsAllTypesNames{"phi", "tgl", "qPt", "int"};
    int nBins = nBinsAllTypes[type];

    std::vector<std::vector<float>> matching_eff_A(nBins);
    std::vector<std::vector<float>> matching_eff_C(nBins);
    std::vector<std::vector<float>> chi2Match_A(nBins);
    std::vector<std::vector<float>> chi2Match_C(nBins);

    std::vector<TBranch*> listBranches;
    listBranches.reserve(4 * nBins);
    for (int i = 0; i < nBins; ++i) {
      const std::string brSuf = (type == nBinTypes - 1) ? nBinsAllTypesNames[type] : fmt::format("{}_{}", nBinsAllTypesNames[type], i);
      listBranches.emplace_back(tree.Branch(fmt::format("{}mITSTPC_A_MatchEff_{}", prefix, brSuf).data(), &matching_eff_A[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mITSTPC_C_MatchEff_{}", prefix, brSuf).data(), &matching_eff_C[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mITSTPC_A_Chi2Match_{}", prefix, brSuf).data(), &chi2Match_A[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mITSTPC_C_Chi2Match_{}", prefix, brSuf).data(), &chi2Match_C[i]));
    }

    // alloc memory
    for (int i = 0; i < nBins; ++i) {
      matching_eff_A[i].resize(nTotPoints);
      matching_eff_C[i].resize(nTotPoints);
      chi2Match_A[i].resize(nTotPoints);
      chi2Match_C[i].resize(nTotPoints);
    }

    for (int j = 0; j < nTotPoints; ++j) {
      for (int i = 0; i < nBins; ++i) {
        int idx = timeSeriesRef.getIndexPhi(i, j); // phi bins
        if (type == 1) {
          idx = timeSeriesRef.getIndexTgl(i, j); // tgl bins
        } else if (type == 2) {
          idx = timeSeriesRef.getIndexqPt(i, j); // qPt
        } else if (type == 3) {
          idx = timeSeriesRef.getIndexInt(j);
        }
        matching_eff_A[i][j] = mITSTPC_A_MatchEff[idx];
        matching_eff_C[i][j] = mITSTPC_C_MatchEff[idx];
        chi2Match_A[i][j] = mITSTPC_A_Chi2Match[idx];
        chi2Match_C[i][j] = mITSTPC_C_Chi2Match[idx];
      }
    }
    // fill branches
    for (auto br : listBranches) {
      br->Fill();
    }
  }
  tree.SetEntries(1);
}

void tpc::TimeSeries::dumpToTree(TTree& tree, const char* prefix, const int nHBFPerTF) const
{
  const auto nValues = mDCAr_A_Median.size();
  const int nPointsPerTF = getNBins();
  const int nTotPoints = nValues / nPointsPerTF;

  const int nBinTypes = 4;
  for (int type = 0; type < nBinTypes; ++type) {
    std::array<int, nBinTypes> nBinsAllTypes{mNBinsPhi, mNBinsTgl, mNBinsqPt, 1};
    std::array<std::string, nBinTypes> nBinsAllTypesNames{"phi", "tgl", "qPt", "int"};
    int nBins = nBinsAllTypes[type];

    std::vector<std::vector<float>> dcar_A_Median(nBins);
    std::vector<std::vector<float>> dcar_C_Median(nBins);
    std::vector<std::vector<float>> dcar_A_WeightedMean(nBins);
    std::vector<std::vector<float>> dcar_C_WeightedMean(nBins);
    std::vector<std::vector<float>> dcar_A_RMS(nBins);
    std::vector<std::vector<float>> dcar_C_RMS(nBins);
    std::vector<std::vector<float>> dcaz_A_Median(nBins);
    std::vector<std::vector<float>> dcaz_C_Median(nBins);
    std::vector<std::vector<float>> dcaz_A_WeightedMean(nBins);
    std::vector<std::vector<float>> dcaz_C_WeightedMean(nBins);
    std::vector<std::vector<float>> dcaz_A_RMS(nBins);
    std::vector<std::vector<float>> dcaz_C_RMS(nBins);
    std::vector<std::vector<float>> nTracksDCAr_A(nBins);
    std::vector<std::vector<float>> nTracksDCAr_C(nBins);
    std::vector<std::vector<float>> nTracksDCAz_A(nBins);
    std::vector<std::vector<float>> nTracksDCAz_C(nBins);
    std::vector<std::vector<float>> mipdEdxRatioA(nBins);
    std::vector<std::vector<float>> mipdEdxRatioC(nBins);
    std::vector<std::vector<float>> tpcChi2A(nBins);
    std::vector<std::vector<float>> tpcChi2C(nBins);
    std::vector<std::vector<float>> tpcNClA(nBins);
    std::vector<std::vector<float>> tpcNClC(nBins);

    std::vector<TBranch*> listBranches;
    listBranches.reserve(22 * nBins);
    for (int i = 0; i < nBins; ++i) {
      const std::string brSuf = (type == nBinTypes - 1) ? nBinsAllTypesNames[type] : fmt::format("{}_{}", nBinsAllTypesNames[type], i);
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAr_A_Median_{}", prefix, brSuf).data(), &dcar_A_Median[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAr_C_Median_{}", prefix, brSuf).data(), &dcar_C_Median[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAr_A_WeightedMean_{}", prefix, brSuf).data(), &dcar_A_WeightedMean[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAr_C_WeightedMean_{}", prefix, brSuf).data(), &dcar_C_WeightedMean[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAr_A_RMS_{}", prefix, brSuf).data(), &dcar_A_RMS[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAr_C_RMS_{}", prefix, brSuf).data(), &dcar_C_RMS[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAz_A_Median_{}", prefix, brSuf).data(), &dcaz_A_Median[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAz_C_Median_{}", prefix, brSuf).data(), &dcaz_C_Median[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAz_A_WeightedMean_{}", prefix, brSuf).data(), &dcaz_A_WeightedMean[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAz_C_WeightedMean_{}", prefix, brSuf).data(), &dcaz_C_WeightedMean[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAz_A_RMS_{}", prefix, brSuf).data(), &dcaz_A_RMS[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAz_C_RMS_{}", prefix, brSuf).data(), &dcaz_C_RMS[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAr_A_NTracks_{}", prefix, brSuf).data(), &nTracksDCAr_A[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAr_C_NTracks_{}", prefix, brSuf).data(), &nTracksDCAr_C[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAz_A_NTracks_{}", prefix, brSuf).data(), &nTracksDCAz_A[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mDCAz_C_NTracks_{}", prefix, brSuf).data(), &nTracksDCAz_C[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mMIPdEdxRatioA_{}", prefix, brSuf).data(), &mipdEdxRatioA[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mMIPdEdxRatioC_{}", prefix, brSuf).data(), &mipdEdxRatioC[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mTPCChi2A_{}", prefix, brSuf).data(), &tpcChi2A[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mTPCChi2C_{}", prefix, brSuf).data(), &tpcChi2C[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mTPCNClA_{}", prefix, brSuf).data(), &tpcNClA[i]));
      listBranches.emplace_back(tree.Branch(fmt::format("{}mTPCNClC_{}", prefix, brSuf).data(), &tpcNClC[i]));
    }

    // alloc memory
    for (int i = 0; i < nBins; ++i) {
      dcar_A_Median[i].resize(nTotPoints);
      dcar_C_Median[i].resize(nTotPoints);
      dcar_A_WeightedMean[i].resize(nTotPoints);
      dcar_C_WeightedMean[i].resize(nTotPoints);
      dcar_A_RMS[i].resize(nTotPoints);
      dcar_C_RMS[i].resize(nTotPoints);
      dcaz_A_Median[i].resize(nTotPoints);
      dcaz_C_Median[i].resize(nTotPoints);
      dcaz_A_WeightedMean[i].resize(nTotPoints);
      dcaz_C_WeightedMean[i].resize(nTotPoints);
      dcaz_A_RMS[i].resize(nTotPoints);
      dcaz_C_RMS[i].resize(nTotPoints);
      nTracksDCAr_A[i].resize(nTotPoints);
      nTracksDCAr_C[i].resize(nTotPoints);
      nTracksDCAz_A[i].resize(nTotPoints);
      nTracksDCAz_C[i].resize(nTotPoints);
      mipdEdxRatioA[i].resize(nTotPoints);
      mipdEdxRatioC[i].resize(nTotPoints);
      tpcChi2A[i].resize(nTotPoints);
      tpcChi2C[i].resize(nTotPoints);
      tpcNClA[i].resize(nTotPoints);
      tpcNClC[i].resize(nTotPoints);
    }

    std::vector<double> time(nTotPoints);
    if (type == 0) {
      listBranches.emplace_back(tree.Branch("time", &time));
    }

    for (int j = 0; j < nTotPoints; ++j) {
      for (int i = 0; i < nBins; ++i) {
        int idx = getIndexPhi(i, j);
        if (type == 1) {
          idx = getIndexTgl(i, j);
        } else if (type == 2) {
          idx = getIndexqPt(i, j);
        } else if (type == 3) {
          idx = getIndexInt(j);
        }
        dcar_A_Median[i][j] = mDCAr_A_Median[idx];
        dcar_C_Median[i][j] = mDCAr_C_Median[idx];
        dcar_A_WeightedMean[i][j] = mDCAr_A_WeightedMean[idx];
        dcar_C_WeightedMean[i][j] = mDCAr_C_WeightedMean[idx];
        dcar_A_RMS[i][j] = mDCAr_A_RMS[idx];
        dcar_C_RMS[i][j] = mDCAr_C_RMS[idx];
        dcaz_A_Median[i][j] = mDCAz_A_Median[idx];
        dcaz_C_Median[i][j] = mDCAz_C_Median[idx];
        dcaz_A_WeightedMean[i][j] = mDCAz_A_WeightedMean[idx];
        dcaz_C_WeightedMean[i][j] = mDCAz_C_WeightedMean[idx];
        dcaz_A_RMS[i][j] = mDCAz_A_RMS[idx];
        dcaz_C_RMS[i][j] = mDCAz_C_RMS[idx];
        nTracksDCAr_A[i][j] = mDCAr_A_NTracks[idx];
        nTracksDCAr_C[i][j] = mDCAr_C_NTracks[idx];
        nTracksDCAz_A[i][j] = mDCAz_A_NTracks[idx];
        nTracksDCAz_C[i][j] = mDCAz_C_NTracks[idx];
        mipdEdxRatioA[i][j] = mMIPdEdxRatioA[idx];
        mipdEdxRatioC[i][j] = mMIPdEdxRatioC[idx];
        tpcChi2A[i][j] = mTPCChi2A[idx];
        tpcChi2C[i][j] = mTPCChi2C[idx];
        tpcNClA[i][j] = mTPCNClA[idx];
        tpcNClC[i][j] = mTPCNClC[idx];
      }
      // add time only once
      if (type == 0) {
        time[j] = mTimeMS + j * nHBFPerTF * o2::constants::lhc::LHCOrbitMUS / 1000;
      }
    }
    // fill branches
    for (auto br : listBranches) {
      br->Fill();
    }
  }
  tree.SetEntries(1);
}

} // end namespace o2
