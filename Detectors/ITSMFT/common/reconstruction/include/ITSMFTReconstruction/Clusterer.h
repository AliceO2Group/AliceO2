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
  ~Clusterer();

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  template <class FullClusCont, class CompClusCont, class PatternCont, class ROFRecCont>
  void process(PixelReader& r,
               FullClusCont* fullClus = nullptr,
               CompClusCont* compClus = nullptr,
               PatternCont* patterns = nullptr,
               ROFRecCont* vecROFRec = nullptr,
               MCTruth* labelsCl = nullptr);

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

  const TStopwatch& getTimer() const { return mTimer; }

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

  template <class FullClusCont, class CompClusCont, class PatternCont>
  void finishChip(FullClusCont* fullClus, CompClusCont* compClus, PatternCont* patterns,
                  const MCTruth* labelsDig = nullptr, MCTruth* labelsClus = nullptr);

  void fetchMCLabels(int digID, const MCTruth* labelsDig, int& nfilled);

  ///< flush cluster data accumulated so far into the tree
  template <class FullClusCont, class CompClusCont>
  void flushClusters(FullClusCont* fullClus, CompClusCont* compClus, MCTruth* labels)
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

  bool mNoLeftColumn = true;                           ///< flag that there is no column on the left to check
  const o2::itsmft::GeometryTGeo* mGeometry = nullptr; //! ITS OR MFT upgrade geometry

  TTree* mClusTree = nullptr;                                      //! externally provided tree to write clusters output (if needed)
  std::array<Label, Cluster::maxLabels> mLabelsBuff;               //! temporary buffer for building cluster labels
  std::array<PixelData, Cluster::kMaxPatternBits * 2> mPixArrBuff; //! temporary buffer for pattern calc.

  LookUp mPattIdConverter; //! Convert the cluster topology to the corresponding entry in the dictionary.

  TStopwatch mTimer;
};

//__________________________________________________
template <class FullClusCont, class CompClusCont, class PatternCont, class ROFRecCont>
void Clusterer::process(PixelReader& reader, FullClusCont* fullClus, CompClusCont* compClus,
                        PatternCont* patterns, ROFRecCont* vecROFRec, MCTruth* labelsCl)
{

#ifdef _PERFORM_TIMING_
  mTimer.Start(kFALSE);
#endif

  o2::itsmft::ROFRecord* rof = nullptr;
  auto clustersCount = compClus->size(); // RSTODO: in principle, the compClus is never supposed to be 0

  while ((mChipData = reader.getNextChipData(mChips))) {
    if (!rof || !(mChipData->getInteractionRecord() == rof->getBCData())) { // new ROF starts
      if (rof) {                                                            // finalize previous slot
        auto cntUpd = compClus->size();
        rof->setNEntries(cntUpd - clustersCount); // update
        clustersCount = cntUpd;
        if (mClusTree) {  // if necessary, flush existing data
          mROFRef = *rof; // just for the legacy way of writing to the tree
          flushClusters(fullClus, compClus, labelsCl);
        }
      }
      rof = &vecROFRec->emplace_back(mChipData->getInteractionRecord(), mChipData->getROFrame(), clustersCount, 0); // create new ROF
    }
    auto chipID = mChipData->getChipID();

    if (mMaxBCSeparationToMask > 0) { // mask pixels fired from the previous ROF
      if (mChipsOld.size() < mChips.size()) {
        mChipsOld.resize(mChips.size()); // expand buffer of previous ROF data
      }
      const auto& chipInPrevROF = mChipsOld[chipID];
      if (std::abs(rof->getBCData().differenceInBC(chipInPrevROF.getInteractionRecord())) < mMaxBCSeparationToMask) {
        mChipData->maskFiredInSample(mChipsOld[chipID]);
      }
    }
    auto validPixID = mChipData->getFirstUnmasked();

    if (validPixID < mChipData->getData().size()) { // chip data may have all of its pixels masked!
      initChip(validPixID++);
      for (; validPixID < mChipData->getData().size(); validPixID++) {
        if (!mChipData->getData()[validPixID].isMasked()) {
          updateChip(validPixID);
        }
      }
      finishChip(fullClus, compClus, patterns, reader.getDigitsMCTruth(), labelsCl);
    }
    if (mMaxBCSeparationToMask > 0) { // current chip data will be used in the next ROF to mask overflow pixels
      mChipsOld[chipID].swap(*mChipData);
    }
  }
  // finalize last ROF
  if (rof) {
    auto cntUpd = compClus->size();
    rof->setNEntries(cntUpd - clustersCount); // update
    if (mClusTree) {                          // if necessary, flush existing data
      mROFRef = *rof;
      flushClusters(fullClus, compClus, labelsCl);
    }
  }
#ifdef _PERFORM_TIMING_
  mTimer.Stop();
#endif
}

/*
//__________________________________________________
template<class FullClusCont, class  CompClusCont, class PatternCont, class ROFRecCont>
void Clusterer::process(PixelReader& reader, FullClusCont* fullClus, CompClusCont* compClus,
                        PatternCont* patterns, ROFRecCont* vecROFRec, MCTruth* labelsCl)
{

#ifdef _PERFORM_TIMING_
  mTimer.Start(kFALSE);
#endif
  mClustersCount = compClus ? compClus->size() : (fullClus ? fullClus->size() : 0);

  while ((mChipData = reader.getNextChipData(mChips))) { // read next chip data to corresponding
    // vector in the mChips and return the pointer on it 
    if (!(mChipData->getInteractionRecord() == mROFRef.getBCData())) { // new ROF starts
      mROFRef.setNEntries(mClustersCount - mROFRef.getEntry().getFirstEntry()); // number of entries in previous ROF
      if (!mROFRef.getBCData().isDummy()) {
        if (mClusTree) { // if necessary, flush existing data
          mROFRef.setFirstEntry(mClusTree->GetEntries());
          flushClusters(fullClus, compClus, labelsCl);
        }
        if (vecROFRec) {
          vecROFRec->emplace_back(mROFRef);
        }
      }
      mROFRef.getEntry().setFirstEntry(mClustersCount);
      mROFRef.getBCData() = mChipData->getInteractionRecord();
      mROFRef.setROFrame(mChipData->getROFrame()); // TODO: outphase this
    }

    auto chipID = mChipData->getChipID();
    
    if (mMaxBCSeparationToMask > 0) { // mask pixels fired from the previous ROF
      if (mChipsOld.size() < mChips.size()) {
        mChipsOld.resize(mChips.size()); // expand buffer of previous ROF data
      }
      const auto& chipInPrevROF = mChipsOld[chipID];
      if (std::abs(mROFRef.getBCData().differenceInBC(chipInPrevROF.getInteractionRecord())) < mMaxBCSeparationToMask) {
        mChipData->maskFiredInSample(mChipsOld[chipID]);
      }
    }
    auto validPixID = mChipData->getFirstUnmasked();
    
    if (validPixID < mChipData->getData().size()) { // chip data may have all of its pixels masked!
      initChip(validPixID++);
      for (; validPixID < mChipData->getData().size(); validPixID++) {
        if (!mChipData->getData()[validPixID].isMasked()) {
          updateChip(validPixID);
        }
      }
      finishChip(fullClus, compClus, patterns, reader.getDigitsMCTruth(), labelsCl);
    }
    if (mMaxBCSeparationToMask > 0) { // current chip data will be used in the next ROF to mask overflow pixels
      mChipsOld[chipID].swap(*mChipData);
    }
  }
  mROFRef.setNEntries(mClustersCount - mROFRef.getEntry().getFirstEntry()); // number of entries in this ROF

  // flush last ROF
  if (!mROFRef.getBCData().isDummy()) {
    if (mClusTree) { // if necessary, flush existing data
      mROFRef.setFirstEntry(mClusTree->GetEntries());
      flushClusters(fullClus, compClus, labelsCl);
    }
    if (vecROFRec) {
      vecROFRec->emplace_back(mROFRef); // the ROFrecords vector is stored outside, in a single entry of the tree
    }
  }
#ifdef _PERFORM_TIMING_
  mTimer.Stop();
#endif
}

 */

//__________________________________________________
template <class FullClusCont, class CompClusCont, class PatternCont>
void Clusterer::finishChip(FullClusCont* fullClus, CompClusCont* compClus, PatternCont* patterns,
                           const MCTruth* labelsDig, MCTruth* labelsClus)
{
  constexpr Float_t SigmaX2 = SegmentationAlpide::PitchRow * SegmentationAlpide::PitchRow / 12.; // FIXME
  constexpr Float_t SigmaY2 = SegmentationAlpide::PitchCol * SegmentationAlpide::PitchCol / 12.; // FIXME
  auto clustersCount = compClus->size();
  const auto& pixData = mChipData->getData();
  int nadd = 0;
  for (int i1 = 0; i1 < mPreClusterHeads.size(); ++i1) {
    const auto ci = mPreClusterIndices[i1];
    if (ci < 0) {
      continue;
    }
    UShort_t rowMax = 0, rowMin = 65535;
    UShort_t colMax = 0, colMin = 65535;
    int nlab = 0, npix = 0;
    int next = mPreClusterHeads[i1];
    while (next >= 0) {
      const auto& pixEntry = mPixels[next];
      const auto pix = pixData[pixEntry.second];
      if (npix < mPixArrBuff.size()) {
        mPixArrBuff[npix++] = pix; // needed for cluster topology
        adjustBoundingBox(pix, rowMin, rowMax, colMin, colMax);
        if (labelsClus) { // the MCtruth for this pixel is at mChipData->startID+pixEntry.second
          fetchMCLabels(pixEntry.second + mChipData->getStartID(), labelsDig, nlab);
        }
        next = pixEntry.first;
      }
    }
    mPreClusterIndices[i1] = -1;
    for (int i2 = i1 + 1; i2 < mPreClusterHeads.size(); ++i2) {
      if (mPreClusterIndices[i2] != ci) {
        continue;
      }
      next = mPreClusterHeads[i2];
      while (next >= 0) {
        const auto& pixEntry = mPixels[next];
        const auto pix = pixData[pixEntry.second]; // PixelData
        if (npix < mPixArrBuff.size()) {
          mPixArrBuff[npix++] = pix; // needed for cluster topology
          adjustBoundingBox(pix, rowMin, rowMax, colMin, colMax);
          if (labelsClus) { // the MCtruth for this pixel is at mChipData->startID+pixEntry.second
            fetchMCLabels(pixEntry.second + mChipData->getStartID(), labelsDig, nlab);
          }
          next = pixEntry.first;
        }
      }
      mPreClusterIndices[i2] = -1;
    }
    UShort_t rowSpan = rowMax - rowMin + 1, colSpan = colMax - colMin + 1;
    Cluster clus;
    clus.setSensorID(mChipData->getChipID());
    clus.setNxNzN(rowSpan, colSpan, npix);
    UShort_t colSpanW = colSpan, rowSpanW = rowSpan;
    if (colSpan * rowSpan > Cluster::kMaxPatternBits) { // need to store partial info
      // will curtail largest dimension
      if (colSpan > rowSpan) {
        if ((colSpanW = Cluster::kMaxPatternBits / rowSpan) == 0) {
          colSpanW = 1;
          rowSpanW = Cluster::kMaxPatternBits;
        }
      } else {
        if ((rowSpanW = Cluster::kMaxPatternBits / colSpan) == 0) {
          rowSpanW = 1;
          colSpanW = Cluster::kMaxPatternBits;
        }
      }
    }
#ifdef _ClusterTopology_
    clus.setPatternRowSpan(rowSpanW, rowSpanW < rowSpan);
    clus.setPatternColSpan(colSpanW, colSpanW < colSpan);
    clus.setPatternRowMin(rowMin);
    clus.setPatternColMin(colMin);
    for (int i = 0; i < npix; i++) {
      const auto pix = mPixArrBuff[i];
      unsigned short ir = pix.getRowDirect() - rowMin, ic = pix.getCol() - colMin;
      if (ir < rowSpanW && ic < colSpanW) {
        clus.setPixel(ir, ic);
      }
    }
#endif              //_ClusterTopology_
    if (fullClus) { // do we need conventional clusters with full topology and coordinates?
      fullClus->push_back(clus);
      Cluster& c = fullClus->back();
      Float_t x = 0., z = 0.;
      for (int i = npix; i--;) {
        x += mPixArrBuff[i].getRowDirect();
        z += mPixArrBuff[i].getCol();
      }
      Point3D<float> xyzLoc;
      SegmentationAlpide::detectorToLocalUnchecked(x / npix, z / npix, xyzLoc);
      auto xyzTra = mGeometry->getMatrixT2L(mChipData->getChipID()) ^ (xyzLoc); // inverse transform from Local to Tracking frame
      c.setPos(xyzTra);
      c.setErrors(SigmaX2, SigmaY2, 0.f);
    }

    if (labelsClus) { // MC labels were requested
      auto cnt = compClus->size();
      for (int i = nlab; i--;) {
        labelsClus->addElement(cnt, mLabelsBuff[i]);
      }
    }

    // add to compact clusters, which must be always filled
    unsigned char patt[Cluster::kMaxPatternBytes] = {0}; // RSTODO FIX pattern filling
    for (int i = 0; i < npix; i++) {
      const auto pix = mPixArrBuff[i];
      unsigned short ir = pix.getRowDirect() - rowMin, ic = pix.getCol() - colMin;
      if (ir < rowSpanW && ic < colSpanW) {
        int nbits = ir * colSpanW + ic;
        patt[nbits >> 3] |= (0x1 << (7 - (nbits % 8)));
      }
    }
    UShort_t pattID = (mPattIdConverter.size() == 0) ? CompCluster::InvalidPatternID : mPattIdConverter.findGroupID(rowSpanW, colSpanW, patt);
    if (pattID == CompCluster::InvalidPatternID || mPattIdConverter.isGroup(pattID)) {
      float xCOG = 0., zCOG = 0.;
      ClusterPattern::getCOG(rowSpanW, colSpanW, patt, xCOG, zCOG);
      rowMin += round(xCOG);
      colMin += round(zCOG);
      if (patterns) {
        patterns->emplace_back((unsigned char)rowSpanW);
        patterns->emplace_back((unsigned char)colSpanW);
        int nBytes = rowSpanW * colSpanW / 8;
        if (((rowSpanW * colSpanW) % 8) != 0)
          nBytes++;
        patterns->insert(patterns->end(), std::begin(patt), std::begin(patt) + nBytes);
      }
    }
    compClus->emplace_back(rowMin, colMin, pattID, mChipData->getChipID());
  }
}

} // namespace itsmft
} // namespace o2
#endif /* ALICEO2_ITS_CLUSTERER_H */
