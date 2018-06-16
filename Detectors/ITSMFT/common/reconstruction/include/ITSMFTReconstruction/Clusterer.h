// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.h
/// \brief Definition of the ITS cluster finder
#ifndef ALICEO2_ITS_CLUSTERER_H
#define ALICEO2_ITS_CLUSTERER_H

#define _PERFORM_TIMING_

#include <utility>
#include <vector>
#include "ITSMFTBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Rtypes.h"

#ifdef _PERFORM_TIMING_
#include <TStopwatch.h>
#endif

class TTree;

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace ITSMFT
{
class Clusterer
{
  using PixelReader = o2::ITSMFT::PixelReader;
  using PixelData = o2::ITSMFT::PixelData;
  using ChipPixelData = o2::ITSMFT::ChipPixelData;
  using Cluster = o2::ITSMFT::Cluster;
  using Label = o2::MCCompLabel;
  using MCTruth = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

 public:
  Clusterer();
  ~Clusterer() = default;

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  void process(PixelReader& r, std::vector<Cluster>& clusters, MCTruth* labelsCl = 0);

  // provide the common ITSMFT::GeometryTGeo to access matrices
  void setGeometry(const o2::ITSMFT::GeometryTGeo* gm) { mGeometry = gm; }

  void setMaskOverflowPixels(bool v) { mMaskOverflowPixels = v; }
  bool isMaskOverflowPixels() const { return mMaskOverflowPixels; }

  UInt_t getCurrROF() const { return mCurrROF; }
  UShort_t getCurrChipID() const { return mCurrChipID; }

  void print() const;
  void clear();

  void setOutputTree(TTree* tr) { mClusTree = tr; }

  void setNChips(int n)
  {
    mChips.resize(n);
    mChipsOld.resize(n);
  }

 private:
  enum { kMaxRow = 650 }; // Anything larger than the real number of rows (512 for ALPIDE)
  void initChip(UInt_t first);
  void updateChip(UInt_t ip);
  void finishChip(std::vector<Cluster>& clusters, const MCTruth* labelsDig, MCTruth* labelsClus = nullptr);
  void fetchMCLabels(int digID, const MCTruth* labelsDig, int& nfilled);

  ///< flush cluster data accumulated so far into the tree
  void flushClusters(std::vector<Cluster>& clusters, MCTruth* labels)
  {
#ifdef _PERFORM_TIMING_
    mTimer.Stop();
#endif
    mClusTree->Fill();
#ifdef _PERFORM_TIMING_
    mTimer.Start(kFALSE);
#endif
    clusters.clear();
    if (labels) {
      labels->clear();
    }
  }

  ChipPixelData* mChipData = nullptr; //! pointer on the current single chip data provided by the reader

  ///< array of chips, at the moment index corresponds to chip ID.
  ///< for the processing of fraction of chips only consider mapping of IDs range on mChips
  std::vector<ChipPixelData> mChips;
  std::vector<ChipPixelData> mChipsOld;

  bool mMaskOverflowPixels = true; ///< flag to mask oveflow pixels (fired from hit in prev. ROF)
  Int_t mColumn1[kMaxRow + 2];
  Int_t mColumn2[kMaxRow + 2];
  Int_t *mCurr, *mPrev;

  UInt_t mCurrROF = o2::ITSMFT::PixelData::DummyROF;
  UShort_t mCurrChipID = o2::ITSMFT::PixelData::DummyChipID;
  using NextIndex = int;
  std::vector<std::pair<NextIndex, UInt_t>> mPixels;
  using FirstIndex = Int_t;
  std::vector<FirstIndex> mPreClusterHeads;
  std::vector<Int_t> mPreClusterIndices;
  UShort_t mCol = 0xffff; ///< Column being processed

  const o2::ITSMFT::GeometryTGeo* mGeometry = nullptr; //! ITS OR MFT upgrade geometry

  TTree* mClusTree = nullptr;                                      //! externally provided tree to write output (if needed)
  std::array<Label, Cluster::maxLabels> mLabelsBuff;               //! temporary buffer for building cluster labels
  std::array<PixelData, Cluster::kMaxPatternBits * 2> mPixArrBuff; //! temporary buffer for pattern calc.

#ifdef _PERFORM_TIMING_
  TStopwatch mTimer;
#endif
};

} // namespace ITSMFT
} // namespace o2
#endif /* ALICEO2_ITS_CLUSTERER_H */
