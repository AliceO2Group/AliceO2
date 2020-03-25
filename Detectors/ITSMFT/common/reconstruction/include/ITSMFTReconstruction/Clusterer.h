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
#include <cstring>
#include "ITSMFTBase/GeometryTGeo.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTReconstruction/LookUp.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonConstants/LHCConstants.h"
#include "Rtypes.h"
#include "TTree.h"

#ifdef _PERFORM_TIMING_
#include <TStopwatch.h>
#endif

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace itsmft
{
class Clusterer
{
  using PixelReader = o2::itsmft::PixelReader;
  using PixelData = o2::itsmft::PixelData;
  using ChipPixelData = o2::itsmft::ChipPixelData;
  using Cluster = o2::itsmft::Cluster;
  using CompCluster = o2::itsmft::CompCluster;
  using CompClusterExt = o2::itsmft::CompClusterExt;
  using Label = o2::MCCompLabel;
  using MCTruth = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

  using BCData = o2::InteractionRecord;

 public:
  Clusterer();
  ~Clusterer() = default;

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  void process(PixelReader& r, std::vector<Cluster>* fullClus,
               std::vector<CompClusterExt>* compClus,
               MCTruth* labelsCl = nullptr,
               std::vector<o2::itsmft::ROFRecord>* vecROFRec = nullptr);

  // provide the common itsmft::GeometryTGeo to access matrices
  void setGeometry(const o2::itsmft::GeometryTGeo* gm) { mGeometry = gm; }

  bool isContinuousReadOut() const { return mContinuousReadout; }
  void setContinuousReadOut(bool v) { mContinuousReadout = v; }

  int getMaxBCSeparationToMask() const { return mMaxBCSeparationToMask; }
  void setMaxBCSeparationToMask(int n) { mMaxBCSeparationToMask = n; }

  void setWantFullClusters(bool v) { mWantFullClusters = v; }
  void setWantCompactClusters(bool v) { mWantCompactClusters = v; }

  bool getWantFullClusters() const { return mWantFullClusters; }
  bool getWantCompactClusters() const { return mWantCompactClusters; }

  UInt_t getCurrROF() const { return mROFRef.getROFrame(); }

  void print() const;
  void clear();

  void setOutputTree(TTree* tr) { mClusTree = tr; }

  void setNChips(int n)
  {
    mChips.resize(n);
    mChipsOld.resize(n);
  }

  ///< load the dictionary of cluster topologies
  void loadDictionary(std::string fileName)
  {
    mPattIdConverter.loadDictionary(fileName);
  }

  void setPatterns(std::vector<o2::itsmft::ClusterPattern>* patt) { mPatterns = patt; }
  std::vector<o2::itsmft::ClusterPattern>* getPatterns() const { return mPatterns; }

 private:
  void initChip(UInt_t first);

  ///< add new precluster at given row of current column for the fired pixel with index ip in the ChipPixelData
  void addNewPrecluster(UInt_t ip, UShort_t row)
  {
    mPreClusterHeads.push_back(mPixels.size());
    // new head does not point yet (-1) on other pixels, store just the entry of the pixel in the ChipPixelData
    mPixels.emplace_back(-1, ip);
    int lastIndex = mPreClusterIndices.size();
    mPreClusterIndices.push_back(lastIndex);
    mCurr[row] = lastIndex; // store index of the new precluster in the current column buffer
  }

  ///< add cluster at row (entry ip in the ChipPixeData) to the precluster with given index
  void expandPreCluster(UInt_t ip, UShort_t row, int preClusIndex)
  {
    auto& firstIndex = mPreClusterHeads[mPreClusterIndices[preClusIndex]];
    mPixels.emplace_back(firstIndex, ip);
    firstIndex = mPixels.size() - 1;
    mCurr[row] = preClusIndex;
  }

  ///< recalculate min max row and column of the cluster accounting for the position of pix
  void adjustBoundingBox(const o2::itsmft::PixelData pix, UShort_t& rMin, UShort_t& rMax,
                         UShort_t& cMin, UShort_t& cMax) const
  {
    if (pix.getRowDirect() < rMin) {
      rMin = pix.getRowDirect();
    }
    if (pix.getRowDirect() > rMax) {
      rMax = pix.getRowDirect();
    }
    if (pix.getCol() < cMin) {
      cMin = pix.getCol();
    }
    if (pix.getCol() > cMax) {
      cMax = pix.getCol();
    }
  }

  ///< swap current and previous column buffers
  void swapColumnBuffers()
  {
    int* tmp = mCurr;
    mCurr = mPrev;
    mPrev = tmp;
  }

  ///< reset column buffer, for the performance reasons we use memset
  void resetColumn(int* buff)
  {
    std::memset(buff, -1, sizeof(int) * SegmentationAlpide::NRows);
    //std::fill(buff, buff + SegmentationAlpide::NRows, -1);
  }

  void updateChip(UInt_t ip);
  void finishChip(std::vector<Cluster>* fullClus, std::vector<CompClusterExt>* compClus,
                  const MCTruth* labelsDig, MCTruth* labelsClus = nullptr);
  void fetchMCLabels(int digID, const MCTruth* labelsDig, int& nfilled);

  ///< flush cluster data accumulated so far into the tree
  void flushClusters(std::vector<Cluster>* fullClus, std::vector<CompClusterExt>* compClus, MCTruth* labels)
  {
#ifdef _PERFORM_TIMING_
    mTimer.Stop();
#endif

    mClusTree->Fill();
#ifdef _PERFORM_TIMING_
    mTimer.Start(kFALSE);
#endif
    if (fullClus) {
      fullClus->clear();
    }
    if (compClus) {
      compClus->clear();
    }
    if (labels) {
      labels->clear();
    }
    mClustersCount = 0;
  }

  // clusterization options
  bool mContinuousReadout = true;    ///< flag continuous readout
  bool mWantFullClusters = true;     ///< request production of full clusters with pattern and coordinates
  bool mWantCompactClusters = false; ///< request production of compact clusters with patternID and corner address

  ///< mask continuosly fired pixels in frames separated by less than this amount of BCs (fired from hit in prev. ROF)
  int mMaxBCSeparationToMask = 6000. / o2::constants::lhc::LHCBunchSpacingNS + 10;

  // aux data for clusterization
  ChipPixelData* mChipData = nullptr; //! pointer on the current single chip data provided by the reader

  ///< array of chips, at the moment index corresponds to chip ID.
  ///< for the processing of fraction of chips only consider mapping of IDs range on mChips
  std::vector<ChipPixelData> mChips;    // currently processed chips data
  std::vector<ChipPixelData> mChipsOld; // previously processed chips data (for masking)

  // buffers for entries in mPreClusterIndices in 2 columns, to avoid boundary checks, we reserve
  // extra elements in the beginning and the end
  int mColumn1[SegmentationAlpide::NRows + 2];
  int mColumn2[SegmentationAlpide::NRows + 2];
  int* mCurr; // pointer on the 1st row of currently processed mColumnsX
  int* mPrev; // pointer on the 1st row of previously processed mColumnsX

  o2::itsmft::ROFRecord mROFRef; // ROF reference

  // mPixels[].first is the index of the next pixel of the same precluster in the mPixels
  // mPixels[].second is the index of the referred pixel in the ChipPixelData (element of mChips)
  std::vector<std::pair<int, UInt_t>> mPixels;
  std::vector<int> mPreClusterHeads; // index of precluster head in the mPixels
  std::vector<int> mPreClusterIndices;
  UShort_t mCol = 0xffff; ///< Column being processed
  int mClustersCount = 0; ///< number of clusters in the output container

  bool mNoLeftColumn = true;                           ///< flag that there is no column on the left to check
  const o2::itsmft::GeometryTGeo* mGeometry = nullptr; //! ITS OR MFT upgrade geometry

  TTree* mClusTree = nullptr;                                      //! externally provided tree to write clusters output (if needed)
  std::array<Label, Cluster::maxLabels> mLabelsBuff;               //! temporary buffer for building cluster labels
  std::array<PixelData, Cluster::kMaxPatternBits * 2> mPixArrBuff; //! temporary buffer for pattern calc.

  LookUp mPattIdConverter; //! Convert the cluster topology to the corresponding entry in the dictionary.

  std::vector<o2::itsmft::ClusterPattern>* mPatterns = nullptr; // Not owned

#ifdef _PERFORM_TIMING_
  TStopwatch mTimer;
#endif
};

} // namespace itsmft
} // namespace o2
#endif /* ALICEO2_ITS_CLUSTERER_H */
