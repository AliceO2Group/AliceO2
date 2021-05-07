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
class ConstMCTruthContainerView;
template <typename T>
class MCTruthContainer;
} // namespace dataformats

namespace its3
{

using CompClusCont = std::vector<itsmft::CompClusterExt>;
using PatternCont = std::vector<unsigned char>;
using ROFRecCont = std::vector<itsmft::ROFRecord>;

//template <class CompClusCont, class PatternCont, class ROFRecCont> // container types (PMR or std::vectors)

class Clusterer
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
  //=========================================================
  /// methods and transient data used within a thread
  struct ThreadStat {
    uint16_t firstChip = 0;
    uint16_t nChips = 0;
    uint32_t firstClus = 0;
    uint32_t firstPatt = 0;
    uint32_t nClus = 0;
    uint32_t nPatt = 0;
    ThreadStat() = default;
  };

  struct ClustererThread {

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
    uint16_t currCol = 0xffff;                                      ///< Column being processed
    bool noLeftCol = true;                                          ///< flag that there is no column on the left to check
    std::array<Label, MaxLabels> labelsBuff;                        //! temporary buffer for building cluster labels
    std::vector<PixelData> pixArrBuff;                              //! temporary buffer for pattern calc.
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

    void streamCluster(const std::vector<PixelData>& pixbuf, uint16_t rowMin, uint16_t rowSpanW, uint16_t colMin, uint16_t colSpanW,
                       uint16_t chipID,
                       CompClusCont* compClusPtr, PatternCont* patternsPtr,
                       MCTruth* labelsClusPtr, int nlab, bool isHuge = false);

    void fetchMCLabels(int digID, const ConstMCTruth* labelsDig, int& nfilled);
    void initChip(const ChipPixelData* curChipData, uint32_t first, int chipID);
    void updateChip(const ChipPixelData* curChipData, uint32_t ip);
    void finishChip(ChipPixelData* curChipData, CompClusCont* compClus, PatternCont* patterns,
                    const ConstMCTruth* labelsDig, MCTruth* labelsClus);
    void finishChipSingleHitFast(uint32_t hit, ChipPixelData* curChipData, CompClusCont* compClusPtr,
                                 PatternCont* patternsPtr, const ConstMCTruth* labelsDigPtr, MCTruth* labelsClusPTr);
    void process(uint16_t chip, uint16_t nChips, CompClusCont* compClusPtr, PatternCont* patternsPtr,
                 const ConstMCTruth* labelsDigPtr, MCTruth* labelsClPtr, const itsmft::ROFRecord& rofPtr);

    ClustererThread(Clusterer* par = nullptr) : parent(par) {}
    ~ClustererThread()
    {
      if (column1) {
        delete[] column1;
      }
      if (column2) {
        delete[] column2;
      }
    }
  };
  //=========================================================

  Clusterer();
  ~Clusterer() = default;

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  void process(int nThreads, PixelReader& r, CompClusCont* compClus, PatternCont* patterns, ROFRecCont* vecROFRec, MCTruth* labelsCl = nullptr);

  bool isContinuousReadOut() const { return mContinuousReadout; }
  void setContinuousReadOut(bool v) { mContinuousReadout = v; }

  int getMaxBCSeparationToMask() const { return mMaxBCSeparationToMask; }
  void setMaxBCSeparationToMask(int n) { mMaxBCSeparationToMask = n; }

  int getMaxRowColDiffToMask() const { return mMaxRowColDiffToMask; }
  void setMaxRowColDiffToMask(int v) { mMaxRowColDiffToMask = v; }

  void print() const;
  void clear();

  void setNChips(int n)
  {
    mChips.resize(n);
    mChipsOld.resize(n);
  }

  ///< load the dictionary of cluster topologies
  void loadDictionary(const std::string& fileName) { mPattIdConverter.loadDictionary(fileName); }

  TStopwatch& getTimer() { return mTimer; }           // cannot be const
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

  void flushClusters(CompClusCont* compClus, MCTruth* labels);

  // clusterization options
  bool mContinuousReadout = true; ///< flag continuous readout

  ///< mask continuosly fired pixels in frames separated by less than this amount of BCs (fired from hit in prev. ROF)
  int mMaxBCSeparationToMask = 6000. / o2::constants::lhc::LHCBunchSpacingNS + 10;
  int mMaxRowColDiffToMask = 0; ///< provide their difference in col/row is <= than this

  std::vector<std::unique_ptr<ClustererThread>> mThreads; // buffers for threads
  std::vector<ChipPixelData> mChips;                      // currently processed ROF's chips data
  std::vector<ChipPixelData> mChipsOld;                   // previously processed ROF's chips data (for masking)
  std::vector<ChipPixelData*> mFiredChipsPtr;             // pointers on the fired chips data in the decoder cache

  itsmft::LookUp mPattIdConverter; //! Convert the cluster topology to the corresponding entry in the dictionary.

  TStopwatch mTimer;
  TStopwatch mTimerMerge;
};

} // namespace its3
} // namespace o2
#endif /* ALICEO2_ITS_CLUSTERER_H */