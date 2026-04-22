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

/// \file Clusterer.h
/// \brief Definition of the ITS cluster finder
#ifndef ALICEO2_ITS_CLUSTERER_H
#define ALICEO2_ITS_CLUSTERER_H

#define _PERFORM_TIMING_

// uncomment this to not allow diagonal clusters, e.g. like |* |
//                                                          | *|
#define _ALLOW_DIAGONAL_ALPIDE_CLUSTERS_

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>
#include <cstring>
#include <memory>
#include <gsl/span>
#include "CommonDataFormat/InteractionRecord.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "Framework/Logger.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTReconstruction/LookUp.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonConstants/LHCConstants.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

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
class ConstMCTruthContainerView;
template <typename T>
class MCTruthContainer;
} // namespace dataformats

namespace itsmft
{

using CompClusCont = std::vector<CompClusterExt>;
using PatternCont = std::vector<unsigned char>;
using ROFRecCont = std::vector<ROFRecord>;

template <typename LookUpT, int MaxRows>
class ClustererT
{
  using PixelReader = o2::itsmft::PixelReader;
  using PixelData = o2::itsmft::PixelData;
  using ChipPixelData = o2::itsmft::ChipPixelData;
  using CompCluster = o2::itsmft::CompCluster;
  using CompClusterExt = o2::itsmft::CompClusterExt;
  using Label = o2::MCCompLabel;
  using MCTruth = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  using ConstMCTruth = o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>;

 public:
  static constexpr int MaxLabels = 10;
  static constexpr int MaxHugeClusWarn = 5; // max number of warnings for HugeCluster

  struct BBox {
    uint16_t chipID = 0xffff;
    uint16_t rowMin = 0xffff;
    uint16_t colMin = 0xffff;
    uint16_t rowMax = 0;
    uint16_t colMax = 0;
    BBox(uint16_t c) : chipID(c) {}
    bool isInside(uint16_t row, uint16_t col) const { return row >= rowMin && row <= rowMax && col >= colMin && col <= colMax; }
    auto rowSpan() const { return rowMax - rowMin + 1; }
    auto colSpan() const { return colMax - colMin + 1; }
    bool isAcceptableSize() const { return colMax - colMin < o2::itsmft::ClusterPattern::MaxColSpan && rowMax - rowMin < o2::itsmft::ClusterPattern::MaxRowSpan; }
    void clear()
    {
      rowMin = colMin = 0xffff;
      rowMax = colMax = 0;
    }
    void adjust(uint16_t row, uint16_t col)
    {
      rowMin = std::min(row, rowMin);
      rowMax = std::max(row, rowMax);
      colMin = std::min(col, colMin);
      colMax = std::max(col, colMax);
    }
  };

  //=========================================================
  /// methods and transient data used within a thread
  struct ThreadStat {
    uint16_t firstChip = 0;
    uint16_t nChips = 0;
    uint32_t firstClus = 0;
    uint32_t firstPatt = 0;
    uint32_t nClus = 0;
    uint32_t nPatt = 0;
  };

  struct ClustererThread {
    struct PreCluster {
      int head = 0; // index of precluster head in the pixels
      int index = 0;
    };
    int id = -1;
    ClustererT* parent = nullptr; // parent clusterer
    // buffers for entries in preClusterIndices in 2 columns, to avoid boundary checks, we reserve
    // extra elements in the beginning and the end
    int column1[MaxRows + 2]{};
    int column2[MaxRows + 2]{};
    int* curr = nullptr; // pointer on the 1st row of currently processed columnsX
    int* prev = nullptr; // pointer on the 1st row of previously processed columnsX
    // pixels[].first is the index of the next pixel of the same precluster in the pixels
    // pixels[].second is the index of the referred pixel in the ChipPixelData (element of mChips)
    std::vector<std::pair<int, uint32_t>> pixels;
    uint16_t currCol = 0xffff;               ///< Column being processed
    bool noLeftCol = true;                   ///< flag that there is no column on the left to check
    std::array<Label, MaxLabels> labelsBuff; //! temporary buffer for building cluster labels
    std::vector<PixelData> pixArrBuff;       //! temporary buffer for pattern calc.
    std::vector<PreCluster> preClusters;     //! preclusters info
    //
    /// temporary storage for the thread output
    CompClusCont compClusters;
    PatternCont patterns;
    MCTruth labels;
    std::vector<ThreadStat> stats; // statistics for each thread results, used at merging
    ///
    ///< reset column buffer, for the performance reasons we use memset
    void resetColumn(int* buff) { std::memset(buff, -1, sizeof(int) * MaxRows); }

    ///< swap current and previous column buffers
    void swapColumnBuffers() { std::swap(prev, curr); }

    ///< add cluster at row (entry ip in the ChipPixeData) to the precluster with given index
    void expandPreCluster(uint32_t ip, uint16_t row, int preClusIndex)
    {
      auto& firstIndex = preClusters[preClusters[preClusIndex].index].head;
      pixels.emplace_back(firstIndex, ip);
      firstIndex = pixels.size() - 1;
      curr[row] = preClusIndex;
    }

    ///< add new precluster at given row of current column for the fired pixel with index ip in the ChipPixelData
    void addNewPrecluster(uint32_t ip, uint16_t row)
    {
      int lastIndex = preClusters.size();
      preClusters.emplace_back(pixels.size(), lastIndex);
      // new head does not point yet (-1) on other pixels, store just the entry of the pixel in the ChipPixelData
      pixels.emplace_back(-1, ip);
      curr[row] = lastIndex; // store index of the new precluster in the current column buffer
    }

    void fetchMCLabels(int digID, const ConstMCTruth* labelsDig, int& nfilled);
    void initChip(const ChipPixelData* curChipData, uint32_t first);
    void updateChip(const ChipPixelData* curChipData, uint32_t ip);
    void finishChip(ChipPixelData* curChipData, CompClusCont* compClus, PatternCont* patterns,
                    const ConstMCTruth* labelsDig, MCTruth* labelsClus);
    void finishChipSingleHitFast(uint32_t hit, ChipPixelData* curChipData, CompClusCont* compClusPtr,
                                 PatternCont* patternsPtr, const ConstMCTruth* labelsDigPtr, MCTruth* labelsClusPTr);
    void process(uint16_t chip, uint16_t nChips, CompClusCont* compClusPtr, PatternCont* patternsPtr,
                 const ConstMCTruth* labelsDigPtr, MCTruth* labelsClPtr, const ROFRecord& rofPtr);

    ClustererThread(ClustererT* par = nullptr, int _id = -1) : parent(par), id(_id), curr(column2 + 1), prev(column1 + 1)
    {
      std::fill(std::begin(column1), std::end(column1), -1);
      std::fill(std::begin(column2), std::end(column2), -1);
    }
  };
  //=========================================================

  ClustererT();
  ClustererT(ClustererT&&) = delete;
  ClustererT& operator=(ClustererT&&) = delete;
  ~ClustererT() = default;

  ClustererT(const ClustererT&) = delete;
  ClustererT& operator=(const ClustererT&) = delete;

  void process(int nThreads, PixelReader& r, CompClusCont* compClus, PatternCont* patterns, ROFRecCont* vecROFRec, MCTruth* labelsCl = nullptr);

  template <typename VCLUS, typename VPAT>
  static void streamCluster(const std::vector<PixelData>& pixbuf, const std::array<Label, MaxLabels>* lblBuff, const BBox& bbox, const LookUpT& pattIdConverter,
                            VCLUS* compClusPtr, VPAT* patternsPtr, MCTruth* labelsClusPtr, int nlab, bool isHuge = false);

  bool isContinuousReadOut() const { return mContinuousReadout; }
  void setContinuousReadOut(bool v) { mContinuousReadout = v; }

  bool isDropHugeClusters() const { return mDropHugeClusters; }
  void setDropHugeClusters(bool v) { mDropHugeClusters = v; }

  int getMaxBCSeparationToMask() const { return mMaxBCSeparationToMask; }
  void setMaxBCSeparationToMask(int n) { mMaxBCSeparationToMask = n; }

  int getMaxRowColDiffToMask() const { return mMaxRowColDiffToMask; }
  void setMaxRowColDiffToMask(int v) { mMaxRowColDiffToMask = v; }

  int getMaxROFDepthToSquash(int layer = -1) const { return (layer < 0) ? mSquashingDepth : mSquashingLayerDepth[layer]; }
  void setMaxROFDepthToSquash(int v) { mSquashingDepth = v; }
  void addMaxROFDepthToSquash(int v) { mSquashingLayerDepth.push_back(v); }

  int getMaxBCSeparationToSquash(int layer = -1) const { return (layer < 0) ? mMaxBCSeparationToSquash : mMaxBCSeparationToSquashLayer[layer]; }
  void setMaxBCSeparationToSquash(int n) { mMaxBCSeparationToSquash = n; }
  void addMaxBCSeparationToSquash(int n) { mMaxBCSeparationToSquashLayer.push_back(n); }

  void print(bool showTiming = true) const;
  void clear();
  void reset();

  void setNChips(int n)
  {
    mChips.resize(n);
    mChipsOld.resize(n);
  }

  ///< load the dictionary of cluster topologies
  void loadDictionary(const std::string& fileName) { mPattIdConverter.loadDictionary(fileName); }
  template <typename TD>
  void setDictionary(const TD* dict)
  {
    mPattIdConverter.setDictionary(dict);
  }

  TStopwatch& getTimer() { return mTimer; }           // cannot be const
  TStopwatch& getTimerMerge() { return mTimerMerge; } // cannot be const

 private:
  void flushClusters(CompClusCont* compClus, MCTruth* labels);

  // clusterization options
  bool mContinuousReadout = true; ///< flag continuous readout
  bool mDropHugeClusters = false; ///< don't include clusters that would be split in more than one

  ///< mask continuously fired pixels in frames separated by less than this amount of BCs (fired from hit in prev. ROF)
  int mMaxBCSeparationToMask = (6000.f / o2::constants::lhc::LHCBunchSpacingNS) + 10;
  int mMaxRowColDiffToMask = 0; ///< provide their difference in col/row is <= than this
  int mNHugeClus = 0;           ///< number of encountered huge clusters

  ///< Squashing options
  int mSquashingDepth = 0; ///< squashing is applied to next N rofs
  int mMaxBCSeparationToSquash = (6000.f / o2::constants::lhc::LHCBunchSpacingNS) + 10;
  std::vector<int> mSquashingLayerDepth;
  std::vector<int> mMaxBCSeparationToSquashLayer;

  std::vector<std::unique_ptr<ClustererThread>> mThreads; // buffers for threads
  std::vector<ChipPixelData> mChips;                      // currently processed ROF's chips data
  std::vector<ChipPixelData> mChipsOld;                   // previously processed ROF's chips data (for masking)
  std::vector<ChipPixelData*> mFiredChipsPtr;             // pointers on the fired chips data in the decoder cache

  LookUpT mPattIdConverter; //! Convert the cluster topology to the corresponding entry in the dictionary.

  TStopwatch mTimer;
  TStopwatch mTimerMerge;
};

template <typename LookUpT, int MaxRows>
template <typename VCLUS, typename VPAT>
void ClustererT<LookUpT, MaxRows>::streamCluster(const std::vector<PixelData>& pixbuf, const std::array<Label, MaxLabels>* lblBuff, const ClustererT::BBox& bbox, const LookUpT& pattIdConverter,
                                                 VCLUS* compClusPtr, VPAT* patternsPtr, MCTruth* labelsClusPtr, int nlab, bool isHuge)
{
  if (labelsClusPtr && lblBuff) { // MC labels were requested
    auto cnt = compClusPtr->size();
    for (int i = nlab; i--;) {
      labelsClusPtr->addElement(cnt, (*lblBuff)[i]);
    }
  }
  auto colSpanW = bbox.colSpan();
  auto rowSpanW = bbox.rowSpan();
  // add to compact clusters, which must be always filled
  std::array<unsigned char, ClusterPattern::MaxPatternBytes> patt{};
  for (const auto& pix : pixbuf) {
    uint32_t ir = pix.getRowDirect() - bbox.rowMin, ic = pix.getCol() - bbox.colMin;
    int nbits = ir * colSpanW + ic;
    patt[nbits >> 3] |= (0x1 << (7 - (nbits % 8)));
  }
  uint16_t pattID = (isHuge || pattIdConverter.size(bbox.chipID) == 0) ? CompCluster::InvalidPatternID : pattIdConverter.findGroupID(rowSpanW, colSpanW, bbox.chipID, patt.data());
  uint16_t row = bbox.rowMin, col = bbox.colMin;
  if (pattID == CompCluster::InvalidPatternID || pattIdConverter.isGroup(pattID, bbox.chipID)) {
    if (pattID != CompCluster::InvalidPatternID) {
      // For grouped topologies, the reference pixel is the COG pixel
      float xCOG = 0., zCOG = 0.;
      ClusterPattern::getCOG(rowSpanW, colSpanW, patt.data(), xCOG, zCOG);
      row += std::round(xCOG);
      col += std::round(zCOG);
    }
    if (patternsPtr) {
      patternsPtr->emplace_back((unsigned char)rowSpanW);
      patternsPtr->emplace_back((unsigned char)colSpanW);
      int nBytes = rowSpanW * colSpanW / 8;
      if (((rowSpanW * colSpanW) % 8) != 0) {
        nBytes++;
      }
      patternsPtr->insert(patternsPtr->end(), std::begin(patt), std::begin(patt) + nBytes);
    }
  }
  compClusPtr->emplace_back(row, col, pattID, bbox.chipID);
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
void ClustererT<LookUpT, MaxRows>::process(int nThreads, PixelReader& reader, CompClusCont* compClus,
                                           PatternCont* patterns, ROFRecCont* vecROFRec, MCTruth* labelsCl)
{
#ifdef _PERFORM_TIMING_
  mTimer.Start(kFALSE);
#endif
  nThreads = std::max(nThreads, 1);
  auto autoDecode = reader.getDecodeNextAuto();
  o2::InteractionRecord lastIR{};
  do {
    if (autoDecode) {
      reader.setDecodeNextAuto(false); // internally do not autodecode
      if (!reader.decodeNextTrigger()) {
        break; // on the fly decoding was requested, but there were no data left
      }
    }
    if (reader.getInteractionRecord().isDummy()) {
      continue; // No IR info was found
    }
    if (!lastIR.isDummy() && lastIR >= reader.getInteractionRecord()) {
      const int MaxErrLog = 2;
      static int errLocCount = 0;
      if (errLocCount++ < MaxErrLog) {
        LOGP(warn, "Impossible ROF IR {}, does not exceed previous {}, discarding in clusterization", reader.getInteractionRecord().asString(), lastIR.asString());
      }
      continue;
    }
    lastIR = reader.getInteractionRecord();
    // pre-fetch all non-empty chips of current ROF
    ChipPixelData* curChipData = nullptr;
    mFiredChipsPtr.clear();
    size_t nPix = 0;
    while ((curChipData = reader.getNextChipData(mChips))) {
      mFiredChipsPtr.push_back(curChipData);
      nPix += curChipData->getData().size();
    }

    auto& rof = vecROFRec->emplace_back(reader.getInteractionRecord(), vecROFRec->size(), compClus->size(), 0); // create new ROF

    uint16_t nFired = mFiredChipsPtr.size();
    if (!nFired) {
      if (autoDecode) {
        continue;
      }
      break; // just 1 ROF was asked to be processed
    }
    nThreads = std::min<int>(nFired, nThreads);
#ifndef WITH_OPENMP
    nThreads = 1;
#endif
    uint16_t chipStep = nThreads > 1 ? (nThreads == 2 ? 20 : 10) : nFired;
    int dynGrp = std::min(4, std::max(1, nThreads / 2));
    if (nThreads > mThreads.size()) {
      int oldSz = mThreads.size();
      mThreads.resize(nThreads);
      for (int i = oldSz; i < nThreads; i++) {
        mThreads[i] = std::make_unique<ClustererThread>(this, i);
      }
    }
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic, dynGrp) num_threads(nThreads)
    //>> start of MT region
    for (uint16_t ic = 0; ic < nFired; ic += chipStep) {
      auto ith = omp_get_thread_num();
      if (nThreads > 1) {
        mThreads[ith]->process(ic, std::min(chipStep, uint16_t(nFired - ic)),
                               &mThreads[ith]->compClusters,
                               patterns ? &mThreads[ith]->patterns : nullptr,
                               labelsCl ? reader.getDigitsMCTruth() : nullptr,
                               labelsCl ? &mThreads[ith]->labels : nullptr, rof);
      } else { // put directly to the destination
        mThreads[0]->process(0, nFired, compClus, patterns, labelsCl ? reader.getDigitsMCTruth() : nullptr, labelsCl, rof);
      }
    }
    //<< end of MT region
#else
    mThreads[0]->process(0, nFired, compClus, patterns, labelsCl ? reader.getDigitsMCTruth() : nullptr, labelsCl, rof);
#endif
    // copy data of all threads but the 1st one to final destination
    if (nThreads > 1) {
#ifdef _PERFORM_TIMING_
      mTimerMerge.Start(false);
#endif
      size_t nClTot = 0, nPattTot = 0;
      int chid = 0;
      std::vector<int> thrStatIdx(nThreads);
      for (int ith = 0; ith < nThreads; ith++) {
        std::sort(mThreads[ith]->stats.begin(), mThreads[ith]->stats.end(), [](const ThreadStat& a, const ThreadStat& b) { return a.firstChip < b.firstChip; });
        thrStatIdx[ith] = 0;
        nClTot += mThreads[ith]->compClusters.size();
        nPattTot += mThreads[ith]->patterns.size();
      }
      compClus->reserve(nClTot);
      if (patterns) {
        patterns->reserve(nPattTot);
      }
      while (chid < nFired) {
        for (int ith = 0; ith < nThreads; ith++) {
          if (thrStatIdx[ith] >= mThreads[ith]->stats.size()) {
            continue;
          }
          const auto& stat = mThreads[ith]->stats[thrStatIdx[ith]];
          if (stat.firstChip == chid) {
            thrStatIdx[ith]++;
            chid += stat.nChips; // next chip to look
            if (stat.nClus > 0) {
              const auto clbeg = mThreads[ith]->compClusters.begin() + stat.firstClus;
              compClus->insert(compClus->end(), clbeg, clbeg + stat.nClus);
              if (patterns) {
                const auto ptbeg = mThreads[ith]->patterns.begin() + stat.firstPatt;
                patterns->insert(patterns->end(), ptbeg, ptbeg + stat.nPatt);
              }
              if (labelsCl) {
                labelsCl->mergeAtBack(mThreads[ith]->labels, stat.firstClus, stat.nClus);
              }
            }
          }
        }
      }
      for (int ith = 0; ith < nThreads; ith++) {
        mThreads[ith]->patterns.clear();
        mThreads[ith]->compClusters.clear();
        mThreads[ith]->labels.clear();
        mThreads[ith]->stats.clear();
      }
#ifdef _PERFORM_TIMING_
      mTimerMerge.Stop();
#endif
    } else {
      mThreads[0]->stats.clear();
    }
    rof.setNEntries(compClus->size() - rof.getFirstEntry()); // update
  } while (autoDecode);
  reader.setDecodeNextAuto(autoDecode); // restore setting
#ifdef _PERFORM_TIMING_
  mTimer.Stop();
#endif
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
void ClustererT<LookUpT, MaxRows>::ClustererThread::process(uint16_t chip, uint16_t nChips, CompClusCont* compClusPtr, PatternCont* patternsPtr,
                                                            const ConstMCTruth* labelsDigPtr, MCTruth* labelsClPtr, const ROFRecord& rofPtr)
{
  if (stats.empty() || stats.back().firstChip + stats.back().nChips != chip) { // there is a jump, register new block
    stats.emplace_back(ThreadStat{.firstChip = chip, .nChips = 0, .firstClus = uint32_t(compClusPtr->size()), .firstPatt = patternsPtr ? uint32_t(patternsPtr->size()) : 0, .nClus = 0, .nPatt = 0});
  }
  for (int ic = 0; ic < nChips; ic++) {
    auto* curChipData = parent->mFiredChipsPtr[chip + ic];
    auto chipID = curChipData->getChipID();
    if (parent->mMaxBCSeparationToMask > 0) { // mask pixels fired from the previous ROF
      const auto& chipInPrevROF = parent->mChipsOld[chipID];
      if (std::abs(rofPtr.getBCData().differenceInBC(chipInPrevROF.getInteractionRecord())) < parent->mMaxBCSeparationToMask) {
        parent->mMaxRowColDiffToMask ? curChipData->maskFiredInSample(parent->mChipsOld[chipID], parent->mMaxRowColDiffToMask) : curChipData->maskFiredInSample(parent->mChipsOld[chipID]);
      }
    }
    auto validPixID = curChipData->getFirstUnmasked();
    auto npix = curChipData->getData().size();
    if (validPixID < npix) { // chip data may have all of its pixels masked!
      auto valp = validPixID++;
      if (validPixID == npix) { // special case of a single pixel fired on the chip
        finishChipSingleHitFast(valp, curChipData, compClusPtr, patternsPtr, labelsDigPtr, labelsClPtr);
      } else {
        initChip(curChipData, valp);
        for (; validPixID < npix; validPixID++) {
          if (!curChipData->getData()[validPixID].isMasked()) {
            updateChip(curChipData, validPixID);
          }
        }
        finishChip(curChipData, compClusPtr, patternsPtr, labelsDigPtr, labelsClPtr);
      }
    }
    if (parent->mMaxBCSeparationToMask > 0) { // current chip data will be used in the next ROF to mask overflow pixels
      parent->mChipsOld[chipID].swap(*curChipData);
    }
  }
  auto& currStat = stats.back();
  currStat.nChips += nChips;
  currStat.nClus = compClusPtr->size() - currStat.firstClus;
  currStat.nPatt = patternsPtr ? (patternsPtr->size() - currStat.firstPatt) : 0;
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
void ClustererT<LookUpT, MaxRows>::ClustererThread::finishChip(ChipPixelData* curChipData, CompClusCont* compClusPtr,
                                                               PatternCont* patternsPtr, const ConstMCTruth* labelsDigPtr, MCTruth* labelsClusPtr)
{
  const auto& pixData = curChipData->getData();
  int nPreclusters = preClusters.size();
  // account for the eventual reindexing of preClusters: Id2 might have been reindexed to Id1, which later was reindexed to Id0
  for (int i = 1; i < nPreclusters; i++) {
    if (preClusters[i].index != i) { // reindexing is always done towards smallest index
      preClusters[i].index = preClusters[preClusters[i].index].index;
    }
  }
  for (int i1 = 0; i1 < nPreclusters; ++i1) {
    auto& preCluster = preClusters[i1];
    auto ci = preCluster.index;
    if (ci < 0) {
      continue;
    }
    BBox bbox(curChipData->getChipID());
    int nlab = 0;
    int next = preCluster.head;
    pixArrBuff.clear();
    while (next >= 0) {
      const auto& pixEntry = pixels[next];
      const auto pix = pixData[pixEntry.second];
      pixArrBuff.push_back(pix); // needed for cluster topology
      bbox.adjust(pix.getRowDirect(), pix.getCol());
      if (labelsClusPtr) {
        if (parent->mSquashingDepth) { // the MCtruth for this pixel is stored in chip data: due to squashing we lose contiguity
          fetchMCLabels(curChipData->getOrderedPixId(pixEntry.second), labelsDigPtr, nlab);
        } else { // the MCtruth for this pixel is at curChipData->startID+pixEntry.second
          fetchMCLabels(pixEntry.second + curChipData->getStartID(), labelsDigPtr, nlab);
        }
      }
      next = pixEntry.first;
    }
    preCluster.index = -1;
    for (int i2 = i1 + 1; i2 < nPreclusters; ++i2) {
      auto& preCluster2 = preClusters[i2];
      if (preCluster2.index != ci) {
        continue;
      }
      next = preCluster2.head;
      while (next >= 0) {
        const auto& pixEntry = pixels[next];
        const auto pix = pixData[pixEntry.second]; // PixelData
        pixArrBuff.push_back(pix);                 // needed for cluster topology
        bbox.adjust(pix.getRowDirect(), pix.getCol());
        if (labelsClusPtr) {
          if (parent->mSquashingDepth) { // the MCtruth for this pixel is stored in chip data: due to squashing we lose contiguity
            fetchMCLabels(curChipData->getOrderedPixId(pixEntry.second), labelsDigPtr, nlab);
          } else { // the MCtruth for this pixel is at curChipData->startID+pixEntry.second
            fetchMCLabels(pixEntry.second + curChipData->getStartID(), labelsDigPtr, nlab);
          }
        }
        next = pixEntry.first;
      }
      preCluster2.index = -1;
    }
    if (bbox.isAcceptableSize()) {
      parent->streamCluster(pixArrBuff, &labelsBuff, bbox, parent->mPattIdConverter, compClusPtr, patternsPtr, labelsClusPtr, nlab);
    } else {
      auto warnLeft = MaxHugeClusWarn - parent->mNHugeClus;
      if (!parent->mDropHugeClusters) {
        if (warnLeft > 0) {
          LOGP(warn, "Splitting a huge cluster: chipID {}, rows {}:{} cols {}:{}{}", bbox.chipID, bbox.rowMin, bbox.rowMax, bbox.colMin, bbox.colMax,
               warnLeft == 1 ? " (Further warnings will be muted)" : "");
#ifdef WITH_OPENMP
#pragma omp critical
#endif
          {
            parent->mNHugeClus++;
          }
        }
        BBox bboxT(bbox); // truncated box
        std::vector<PixelData> pixbuf;
        do {
          bboxT.rowMin = bbox.rowMin;
          bboxT.colMax = std::min(bbox.colMax, uint16_t(bboxT.colMin + o2::itsmft::ClusterPattern::MaxColSpan - 1));
          do { // Select a subset of pixels fitting the reduced bounding box
            bboxT.rowMax = std::min(bbox.rowMax, uint16_t(bboxT.rowMin + o2::itsmft::ClusterPattern::MaxRowSpan - 1));
            for (const auto& pix : pixArrBuff) {
              if (bboxT.isInside(pix.getRowDirect(), pix.getCol())) {
                pixbuf.push_back(pix);
              }
            }
            if (!pixbuf.empty()) { // Stream a piece of cluster only if the reduced bounding box is not empty
              parent->streamCluster(pixbuf, &labelsBuff, bboxT, parent->mPattIdConverter, compClusPtr, patternsPtr, labelsClusPtr, nlab, true);
              pixbuf.clear();
            }
            bboxT.rowMin = bboxT.rowMax + 1;
          } while (bboxT.rowMin < bbox.rowMax);
          bboxT.colMin = bboxT.colMax + 1;
        } while (bboxT.colMin < bbox.colMax);
      }
    }
  }
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
void ClustererT<LookUpT, MaxRows>::ClustererThread::finishChipSingleHitFast(uint32_t hit, ChipPixelData* curChipData, CompClusCont* compClusPtr,
                                                                            PatternCont* patternsPtr, const ConstMCTruth* labelsDigPtr, MCTruth* labelsClusPtr)
{
  auto pix = curChipData->getData()[hit];
  uint16_t row = pix.getRowDirect(), col = pix.getCol();

  if (labelsClusPtr) { // MC labels were requested
    int nlab = 0;
    fetchMCLabels(curChipData->getStartID() + hit, labelsDigPtr, nlab);
    auto cnt = compClusPtr->size();
    for (int i = nlab; i--;) {
      labelsClusPtr->addElement(cnt, labelsBuff[i]);
    }
  }

  // add to compact clusters, which must be always filled
  unsigned char patt[ClusterPattern::MaxPatternBytes]{0x1 << (7 - (0 % 8))}; // unrolled 1 hit version of full loop in finishChip
  const auto chipID = curChipData->getChipID();
  uint16_t pattID = (parent->mPattIdConverter.size(chipID) == 0) ? CompCluster::InvalidPatternID : parent->mPattIdConverter.findGroupID(1, 1, chipID, patt);
  if ((pattID == CompCluster::InvalidPatternID || parent->mPattIdConverter.isGroup(pattID, chipID)) && patternsPtr) {
    patternsPtr->emplace_back(1); // rowspan
    patternsPtr->emplace_back(1); // colspan
    patternsPtr->insert(patternsPtr->end(), std::begin(patt), std::begin(patt) + 1);
  }
  compClusPtr->emplace_back(row, col, pattID, chipID);
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
ClustererT<LookUpT, MaxRows>::ClustererT()
{
#ifdef _PERFORM_TIMING_
  mTimer.Stop();
  mTimer.Reset();
  mTimerMerge.Stop();
  mTimerMerge.Reset();
#endif
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
void ClustererT<LookUpT, MaxRows>::ClustererThread::initChip(const ChipPixelData* curChipData, uint32_t first)
{
  // init chip with the 1st unmasked pixel (entry "from" in the mChipData)
  prev = column1 + 1;
  curr = column2 + 1;
  resetColumn(curr);
  pixels.clear();
  preClusters.clear();
  auto pix = curChipData->getData()[first];
  currCol = pix.getCol();
  curr[pix.getRowDirect()] = 0; // can use getRowDirect since the pixel is not masked
  // start the first pre-cluster
  preClusters.emplace_back();
  pixels.emplace_back(-1, first); // id of current pixel
  noLeftCol = true;
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
void ClustererT<LookUpT, MaxRows>::ClustererThread::updateChip(const ChipPixelData* curChipData, uint32_t ip)
{
  const auto pix = curChipData->getData()[ip];
  uint16_t row = pix.getRowDirect(); // can use getRowDirect since the pixel is not masked
  if (currCol != pix.getCol()) {     // switch the buffers
    swapColumnBuffers();
    resetColumn(curr);
    noLeftCol = false;
    if (pix.getCol() > currCol + 1) {
      // no connection with previous column, this pixel cannot belong to any of the
      // existing preclusters, create a new precluster and flag to check only the row above for next pixels of this column
      currCol = pix.getCol();
      addNewPrecluster(ip, row);
      noLeftCol = true;
      return;
    }
    currCol = pix.getCol();
  }

  if (noLeftCol) { // check only the row above
    if (curr[row - 1] >= 0) {
      expandPreCluster(ip, row, curr[row - 1]); // attach to the precluster of the previous row
    } else {
      addNewPrecluster(ip, row); // start new precluster
    }
  } else {
    // row above should be always checked
    int nnb = 0, lowestIndex = curr[row - 1], lowestNb = 0, *nbrCol[4], nbrRow[4];
    if (lowestIndex >= 0) {
      nbrCol[nnb] = curr;
      nbrRow[nnb++] = row - 1;
    } else {
      lowestIndex = 0x7ffff;
      lowestNb = -1;
    }
#ifdef _ALLOW_DIAGONAL_ALPIDE_CLUSTERS_
    for (int i : {-1, 0, 1}) {
      auto v = prev[row + i];
      if (v >= 0) {
        nbrCol[nnb] = prev;
        nbrRow[nnb] = row + i;
        if (v < lowestIndex) {
          lowestIndex = v;
          lowestNb = nnb;
        }
        nnb++;
      }
    }
#else
    if (prev[row] >= 0) {
      nbrCol[nnb] = prev;
      nbrRow[nnb] = row;
      if (prev[row] < lowestIndex) {
        lowestIndex = prev[row];
        lowestNb = nnb;
      }
      nnb++;
    }
#endif
    if (!nnb) {                  // no neighbours, create new precluster
      addNewPrecluster(ip, row); // start new precluster
    } else {
      expandPreCluster(ip, row, lowestIndex); // attach to the adjascent precluster with smallest index
      if (nnb > 1) {
        for (int inb = 0; inb < nnb; inb++) { // reassign precluster index to smallest one, replicating updated values to columns caches
          auto& prevIndex = (nbrCol[inb])[nbrRow[inb]];
          prevIndex = preClusters[prevIndex].index = lowestIndex;
        }
      }
    }
  }
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
void ClustererT<LookUpT, MaxRows>::ClustererThread::fetchMCLabels(int digID, const ConstMCTruth* labelsDig, int& nfilled)
{
  // transfer MC labels to cluster
  if (nfilled >= MaxLabels) {
    return;
  }
  const auto& lbls = labelsDig->getLabels(digID);
  for (int i = lbls.size(); i--;) {
    int ic = nfilled;
    for (; ic--;) { // check if the label is already present
      if (labelsBuff[ic] == lbls[i]) {
        return; // label is found, do nothing
      }
    }
    labelsBuff[nfilled++] = lbls[i];
    if (nfilled >= MaxLabels) {
      break;
    }
  }
  //
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
void ClustererT<LookUpT, MaxRows>::clear()
{
  // reset
#ifdef _PERFORM_TIMING_
  mTimer.Stop();
  mTimer.Reset();
  mTimerMerge.Stop();
  mTimerMerge.Reset();
#endif
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
void ClustererT<LookUpT, MaxRows>::print(bool showsTiming) const
{
  // print settings
  if (mSquashingLayerDepth.empty()) {
    LOGP(info, "Clusterizer squashes overflow pixels separated by {} BC and <= {} in row/col seeking down to {} neighbour ROFs", mMaxBCSeparationToSquash, mMaxRowColDiffToMask, mSquashingDepth);
  } else {
    LOGP(info, "Clusterizer squashes overflow pixels <= {} in row/col", mMaxRowColDiffToMask);
    for (size_t i{0}; i < mSquashingLayerDepth.size(); ++i) {
      LOGP(info, "\tClusterizer on layer {} separated by {} BC seeking down to {} neighbour ROFs", i, mMaxBCSeparationToSquashLayer[i], mSquashingLayerDepth[i]);
    }
  }
  LOGP(info, "Clusterizer masks overflow pixels separated by < {} BC and <= {} in row/col", mMaxBCSeparationToMask, mMaxRowColDiffToMask);
  LOGP(info, "Clusterizer does {} drop huge clusters", mDropHugeClusters ? "" : "not");

  if (showsTiming) {
#ifdef _PERFORM_TIMING_
    auto& tmr = const_cast<TStopwatch&>(mTimer); // ugly but this is what root does internally
    auto& tmrm = const_cast<TStopwatch&>(mTimerMerge);
    LOG(info) << "Inclusive clusterization timing (w/o disk IO): Cpu: " << tmr.CpuTime()
              << " Real: " << tmr.RealTime() << " s in " << tmr.Counter() << " slots";
    LOG(info) << "Threads output merging timing                : Cpu: " << tmrm.CpuTime()
              << " Real: " << tmrm.RealTime() << " s in " << tmrm.Counter() << " slots";

#endif
  }
}

//__________________________________________________
template <typename LookUpT, int MaxRows>
void ClustererT<LookUpT, MaxRows>::reset()
{
  // reset for new run
  clear();
  mNHugeClus = 0;
}

using Clusterer = ClustererT<LookUp, SegmentationAlpide::NRows>;
extern template class ClustererT<LookUp, SegmentationAlpide::NRows>;

} // namespace itsmft
} // namespace o2
#endif /* ALICEO2_ITS_CLUSTERER_H */
