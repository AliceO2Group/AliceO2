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
#include <memory>
#include <gsl/span>
#include "ITSMFTBase/SegmentationAlpide.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTReconstruction/LookUp.h"
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
class MCTruthContainer;
}

namespace itsmft
{

class GeometryTGeo;

using FullClusCont = std::vector<Cluster>;
using CompClusCont = std::vector<CompClusterExt>;
using PatternCont = std::vector<unsigned char>;
using ROFRecCont = std::vector<ROFRecord>;

//template <class FullClusCont, class CompClusCont, class PatternCont, class ROFRecCont> // container types (PMR or std::vectors)

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

 public:
  //=========================================================
  /// methods and transient data used within a thread
  struct ClustererThread {
    static constexpr float SigmaX2 = SegmentationAlpide::PitchRow * SegmentationAlpide::PitchRow / 12.;
    static constexpr float SigmaY2 = SegmentationAlpide::PitchCol * SegmentationAlpide::PitchCol / 12.;

    Clusterer* parent = nullptr; // parent clusterer
    // buffers for entries in preClusterIndices in 2 columns, to avoid boundary checks, we reserve
    // extra elements in the beginning and the end
    int column1[SegmentationAlpide::NRows + 2];
    int column2[SegmentationAlpide::NRows + 2];
    int* curr = nullptr; // pointer on the 1st row of currently processed columnsX
    int* prev = nullptr; // pointer on the 1st row of previously processed columnsX
    // pixels[].first is the index of the next pixel of the same precluster in the pixels
    // pixels[].second is the index of the referred pixel in the ChipPixelData (element of mChips)
    std::vector<std::pair<int, uint32_t>> pixels;
    std::vector<int> preClusterHeads; // index of precluster head in the pixels
    std::vector<int> preClusterIndices;
    uint16_t currCol = 0xffff;                                      ///< Column being processed
    bool noLeftCol = true;                                          ///< flag that there is no column on the left to check
    std::array<Label, Cluster::maxLabels> labelsBuff;               //! temporary buffer for building cluster labels
    std::array<PixelData, Cluster::kMaxPatternBits * 2> pixArrBuff; //! temporary buffer for pattern calc.
    //
    /// temporary storage for the thread output
    FullClusCont fullClusters;
    CompClusCont compClusters;
    PatternCont patterns;
    MCTruth labels;
    ///
    ///< reset column buffer, for the performance reasons we use memset
    void resetColumn(int* buff) { std::memset(buff, -1, sizeof(int) * SegmentationAlpide::NRows); }

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

    void fetchMCLabels(int digID, const MCTruth* labelsDig, int& nfilled);
    void initChip(const ChipPixelData* curChipData, uint32_t first);
    void updateChip(const ChipPixelData* curChipData, uint32_t ip);
    void finishChip(ChipPixelData* curChipData, FullClusCont* fullClus, CompClusCont* compClus, PatternCont* patterns,
                    const MCTruth* labelsDig, MCTruth* labelsClus);
    void finishChipSingleHitFast(uint32_t hit, ChipPixelData* curChipData, FullClusCont* fullClusPtr, CompClusCont* compClusPtr,
                                 PatternCont* patternsPtr, const MCTruth* labelsDigPtr, MCTruth* labelsClusPTr);
    void process(gsl::span<ChipPixelData*> chipPtrs, FullClusCont* fullClusPtr, CompClusCont* compClusPtr, PatternCont* patternsPtr,
                 const MCTruth* labelsDigPtr, MCTruth* labelsClPtr, const ROFRecord* rofPtr);

    ClustererThread(Clusterer* par = nullptr) : parent(par), curr(column2 + 1), prev(column1 + 1)
    {
      std::fill(std::begin(column1), std::end(column1), -1);
      std::fill(std::begin(column2), std::end(column2), -1);
    }
  };
  //=========================================================

  Clusterer();
  ~Clusterer() = default;

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  void process(int nThreads, PixelReader& r, FullClusCont* fullClus, CompClusCont* compClus, PatternCont* patterns, ROFRecCont* vecROFRec, MCTruth* labelsCl = nullptr);

  void setGeometry(const o2::itsmft::GeometryTGeo* gm) { mGeometry = gm; }

  bool isContinuousReadOut() const { return mContinuousReadout; }
  void setContinuousReadOut(bool v) { mContinuousReadout = v; }

  int getMaxBCSeparationToMask() const { return mMaxBCSeparationToMask; }
  void setMaxBCSeparationToMask(int n) { mMaxBCSeparationToMask = n; }

  int getMaxRowColDiffToMask() const { return mMaxRowColDiffToMask; }
  void setMaxRowColDiffToMask(int v) { mMaxRowColDiffToMask = v; }

  void setWantFullClusters(bool v) { mWantFullClusters = v; }
  void setWantCompactClusters(bool v) { mWantCompactClusters = v; }

  bool getWantFullClusters() const { return mWantFullClusters; }
  bool getWantCompactClusters() const { return mWantCompactClusters; }

  uint32_t getCurrROF() const { return mROFRef.getROFrame(); }

  void print() const;
  void clear();

  void setOutputTree(TTree* tr) { mClusTree = tr; }

  void setNChips(int n)
  {
    mChips.resize(n);
    mChipsOld.resize(n);
  }

  ///< load the dictionary of cluster topologies
  void loadDictionary(const std::string& fileName) { mPattIdConverter.loadDictionary(fileName); }

  TStopwatch& getTimer() { return mTimer; } // cannot be const
  TStopwatch& getTimerMerge() { return mTimerMerge; } // cannot be const

 private:

  ///< recalculate min max row and column of the cluster accounting for the position of pix
  static void adjustBoundingBox(uint16_t row, uint16_t col, uint16_t& rMin, uint16_t& rMax, uint16_t& cMin, uint16_t& cMax)
  {
    if (row < rMin) {
      rMin = row;
    }
    if (row > rMax) {
      rMax = row;
    }
    if (col < cMin) {
      cMin = col;
    }
    if (col > cMax) {
      cMax = col;
    }
  }

  void flushClusters(FullClusCont* fullClus, CompClusCont* compClus, MCTruth* labels);

  // clusterization options
  bool mContinuousReadout = true;    ///< flag continuous readout
  bool mWantFullClusters = true;     ///< request production of full clusters with pattern and coordinates
  bool mWantCompactClusters = false; ///< request production of compact clusters with patternID and corner address

  ///< mask continuosly fired pixels in frames separated by less than this amount of BCs (fired from hit in prev. ROF)
  int mMaxBCSeparationToMask = 6000. / o2::constants::lhc::LHCBunchSpacingNS + 10;
  int mMaxRowColDiffToMask = 0; ///< provide their difference in col/row is <= than this

  std::vector<std::unique_ptr<ClustererThread>> mThreads; // buffers for threads
  std::vector<ChipPixelData> mChips;                      // currently processed ROF's chips data
  std::vector<ChipPixelData> mChipsOld;                   // previously processed ROF's chips data (for masking)
  std::vector<ChipPixelData*> mFiredChipsPtr;             // pointers on the fired chips data in the decoder cache

  const o2::itsmft::GeometryTGeo* mGeometry = nullptr; //! ITS OR MFT upgrade geometry

  LookUp mPattIdConverter; //! Convert the cluster topology to the corresponding entry in the dictionary.

  // this makes sense only for single-threaded execution with autosaving of the tree: legacy, TOREM
  o2::itsmft::ROFRecord mROFRef; // ROF reference
  TTree* mClusTree = nullptr;    //! externally provided tree to write clusters output (if needed)

  TStopwatch mTimer;
  TStopwatch mTimerMerge;
};


} // namespace itsmft
} // namespace o2
#endif /* ALICEO2_ITS_CLUSTERER_H */
