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

#include <utility>
#include <vector>
#include <cstring>
#include <memory>
#include <gsl/span>
#include "ITSMFTBase/SegmentationAlpide.h"
#include "DataFormatsITS3/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "ITS3Reconstruction/LookUp.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonConstants/LHCConstants.h"
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
class ConstMCTruthContainerView;
template <typename T>
class MCTruthContainer;
} // namespace dataformats

namespace its3
{

using CompClusCont = std::vector<its3::CompClusterExt>;
using PatternCont = std::vector<unsigned char>;
using ROFRecCont = std::vector<itsmft::ROFRecord>;

// template <class CompClusCont, class PatternCont, class ROFRecCont> // container types (PMR or std::vectors)

class Clusterer
{
  using PixelReader = o2::itsmft::PixelReader;
  using PixelData = o2::itsmft::PixelData;
  using ChipPixelData = o2::itsmft::ChipPixelData;
  using CompCluster = o2::its3::CompCluster;
  using CompClusterExt = o2::its3::CompClusterExt;
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
      if (row < rowMin) {
        rowMin = row;
      }
      if (row > rowMax) {
        rowMax = row;
      }
      if (col < colMin) {
        colMin = col;
      }
      if (col > colMax) {
        colMax = col;
      }
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
    int id = -1;
    int nLayersITS3 = 3;         // number of ITS3 layers
    Clusterer* parent = nullptr; // parent clusterer
    // buffers for entries in preClusterIndices in 2 columns, to avoid boundary checks, we reserve
    // extra elements in the beginning and the end
    int* column1 = nullptr;
    int* column2 = nullptr;
    int* curr = nullptr; // pointer on the 1st row of currently processed columnsX
    int* prev = nullptr; // pointer on the 1st row of previously processed columnsX
    int size = itsmft::SegmentationAlpide::NRows + 2;
    // pixels[].first is the index of the next pixel of the same precluster in the pixels
    // pixels[].second is the index of the referred pixel in the ChipPixelData (element of mChips)
    std::vector<std::pair<int, uint32_t>> pixels;
    std::vector<int> preClusterHeads; // index of precluster head in the pixels
    std::vector<int> preClusterIndices;
    uint16_t currCol = 0xffff;               ///< Column being processed
    bool noLeftCol = true;                   ///< flag that there is no column on the left to check
    std::array<Label, MaxLabels> labelsBuff; //! temporary buffer for building cluster labels
    std::vector<PixelData> pixArrBuff;       //! temporary buffer for pattern calc.
    //
    /// temporary storage for the thread output
    CompClusCont compClusters;
    PatternCont patterns;
    MCTruth labels;
    std::vector<ThreadStat> stats; // statistics for each thread results, used at merging
    ///
    ///< reset column buffer, for the performance reasons we use memset
    void resetColumn(int* buff) { std::memset(buff, -1, sizeof(int) * (size - 2)); }

    ///< swap current and previous column buffers
    void swapColumnBuffers() { std::swap(prev, curr); }

    ///< add cluster at row (entry ip in the ChipPixeData) to the precluster with given index
    void expandPreCluster(uint32_t ip, uint16_t row, int preClusIndex)
    {
      auto& firstIndex = preClusterHeads[preClusterIndices[preClusIndex]];
      pixels.emplace_back(firstIndex, ip);
      firstIndex = pixels.size() - 1;
      curr[row] = preClusIndex;
    }

    ///< add new precluster at given row of current column for the fired pixel with index ip in the ChipPixelData
    void addNewPrecluster(uint32_t ip, uint16_t row)
    {
      preClusterHeads.push_back(pixels.size());
      // new head does not point yet (-1) on other pixels, store just the entry of the pixel in the ChipPixelData
      pixels.emplace_back(-1, ip);
      int lastIndex = preClusterIndices.size();
      preClusterIndices.push_back(lastIndex);
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
                 const ConstMCTruth* labelsDigPtr, MCTruth* labelsClPtr, const itsmft::ROFRecord& rofPtr);

    ~ClustererThread()
    {
      if (column1) {
        delete[] column1;
      }
      if (column2) {
        delete[] column2;
      }
    }
    ClustererThread(Clusterer* par = nullptr, int _id = -1) : parent(par), id(_id), curr(column2 + 1), prev(column1 + 1)
    {
      // std::fill(std::begin(column1), std::end(column1), -1);
      // std::fill(std::begin(column2), std::end(column2), -1);
    }
  };
  //=========================================================

  Clusterer();
  ~Clusterer() = default;

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  void process(int nThreads, PixelReader& r, CompClusCont* compClus, PatternCont* patterns, ROFRecCont* vecROFRec, MCTruth* labelsCl = nullptr);

  template <typename VCLUS, typename VPAT>
  static void streamCluster(const std::vector<PixelData>& pixbuf, const std::array<Label, MaxLabels>* lblBuff, const BBox& bbox, const its3::LookUp& pattIdConverter,
                            VCLUS* compClusPtr, VPAT* patternsPtr, MCTruth* labelsClusPtr, int nlab, bool isHuge = false);

  bool isContinuousReadOut() const { return mContinuousReadout; }
  void setContinuousReadOut(bool v) { mContinuousReadout = v; }

  int getMaxBCSeparationToMask() const { return mMaxBCSeparationToMask; }
  void setMaxBCSeparationToMask(int n) { mMaxBCSeparationToMask = n; }

  int getMaxRowColDiffToMask() const { return mMaxRowColDiffToMask; }
  void setMaxRowColDiffToMask(int v) { mMaxRowColDiffToMask = v; }

  int getMaxROFDepthToSquash() const { return mSquashingDepth; }
  void setMaxROFDepthToSquash(int v) { mSquashingDepth = v; }

  int getMaxBCSeparationToSquash() const { return mMaxBCSeparationToSquash; }
  void setMaxBCSeparationToSquash(int n) { mMaxBCSeparationToSquash = n; }

  void print() const;
  void clear();

  void setNChips(int n)
  {
    mChips.resize(n);
    mChipsOld.resize(n);
  }

  void setNumLayersITS3(int n) { mNlayersITS3 = n; }

  ///< load the dictionary of cluster topologies
  void loadDictionary(const std::string& fileName) { mPattIdConverter.loadDictionary(fileName); }
  void setDictionary(const its3::TopologyDictionary* dict) { mPattIdConverter.setDictionary(dict); }

  TStopwatch& getTimer() { return mTimer; }           // cannot be const
  TStopwatch& getTimerMerge() { return mTimerMerge; } // cannot be const

 private:
  void flushClusters(CompClusCont* compClus, MCTruth* labels);

  // geometry options
  int mNlayersITS3 = 3; ///< number of ITS3 layers

  // clusterization options
  bool mContinuousReadout = true; ///< flag continuous readout

  ///< mask continuosly fired pixels in frames separated by less than this amount of BCs (fired from hit in prev. ROF)
  int mMaxBCSeparationToMask = 6000. / o2::constants::lhc::LHCBunchSpacingNS + 10;
  int mMaxRowColDiffToMask = 0; ///< provide their difference in col/row is <= than this
  int mNHugeClus = 0;           ///< number of encountered huge clusters

  ///< Squashing options
  int mSquashingDepth = 0; ///< squashing is applied to next N rofs
  int mMaxBCSeparationToSquash = 6000. / o2::constants::lhc::LHCBunchSpacingNS + 10;

  std::vector<std::unique_ptr<ClustererThread>> mThreads; // buffers for threads
  std::vector<ChipPixelData> mChips;                      // currently processed ROF's chips data
  std::vector<ChipPixelData> mChipsOld;                   // previously processed ROF's chips data (for masking)
  std::vector<ChipPixelData*> mFiredChipsPtr;             // pointers on the fired chips data in the decoder cache

  its3::LookUp mPattIdConverter; //! Convert the cluster topology to the corresponding entry in the dictionary.

  TStopwatch mTimer;
  TStopwatch mTimerMerge;
};

template <typename VCLUS, typename VPAT>
void Clusterer::streamCluster(const std::vector<PixelData>& pixbuf, const std::array<Label, MaxLabels>* lblBuff, const Clusterer::BBox& bbox, const its3::LookUp& pattIdConverter,
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
  std::array<unsigned char, itsmft::ClusterPattern::MaxPatternBytes> patt{};
  for (const auto& pix : pixbuf) {
    uint32_t ir = pix.getRowDirect() - bbox.rowMin, ic = pix.getCol() - bbox.colMin;
    int nbits = ir * colSpanW + ic;
    patt[nbits >> 3] |= (0x1 << (7 - (nbits % 8)));
  }
  uint16_t pattID = (isHuge || pattIdConverter.size() == 0) ? CompCluster::InvalidPatternID : pattIdConverter.findGroupID(rowSpanW, colSpanW, patt.data());
  uint16_t row = bbox.rowMin, col = bbox.colMin;
  if (pattID == CompCluster::InvalidPatternID || pattIdConverter.isGroup(pattID)) {
    if (pattID != CompCluster::InvalidPatternID) {
      // For groupped topologies, the reference pixel is the COG pixel
      float xCOG = 0., zCOG = 0.;
      itsmft::ClusterPattern::getCOG(rowSpanW, colSpanW, patt.data(), xCOG, zCOG);
      row += round(xCOG);
      col += round(zCOG);
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

} // namespace its3
} // namespace o2
#endif /* ALICEO2_ITS_CLUSTERER_H */
