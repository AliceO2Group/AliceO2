// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.cxx
/// \brief Implementation of the ITS cluster finder
#include <algorithm>
#include <TTree.h>
#include "Framework/Logger.h"
#include "ITSMFTBase/GeometryTGeo.h"
#include "ITS3Reconstruction/Clusterer.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonDataFormat/InteractionRecord.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::itsmft;

namespace o2::its3 
{
//__________________________________________________
void Clusterer::process(int nThreads, PixelReader& reader, CompClusCont* compClus,
                        PatternCont* patterns, ROFRecCont* vecROFRec, MCTruth* labelsCl)
{
#ifdef _PERFORM_TIMING_
  mTimer.Start(kFALSE);
#endif
  if (nThreads < 1) {
    nThreads = 1;
  }
  auto autoDecode = reader.getDecodeNextAuto();
  do {
    if (autoDecode) {
      reader.setDecodeNextAuto(false); // internally do not autodecode
      if (!reader.decodeNextTrigger()) {
        break; // on the fly decoding was requested, but there were no data left
      }
    }
    // pre-fetch all non-empty chips of current ROF
    ChipPixelData* curChipData = nullptr;
    mFiredChipsPtr.clear();
    size_t nPix = 0;
    while ((curChipData = reader.getNextChipData(mChips))) {
      mFiredChipsPtr.push_back(curChipData);
      nPix += curChipData->getData().size();
    }

    auto& rof = vecROFRec->emplace_back(reader.getInteractionRecord(), 0, compClus->size(), 0); // create new ROF

    uint16_t nFired = mFiredChipsPtr.size();
    if (!nFired) {
      if (autoDecode) {
        continue;
      }
      break; // just 1 ROF was asked to be processed
    }
    if (nFired < nThreads) {
      nThreads = nFired;
    }
#ifdef WITH_OPENMP
    omp_set_num_threads(nThreads);
#else
    nThreads = 1;
#endif
    uint16_t chipStep = nThreads > 1 ? (nThreads == 2 ? 20 : (nThreads < 5 ? 5 : 1)) : nFired;
    int dynGrp = std::min(4, std::max(1, nThreads / 2));
    if (nThreads > mThreads.size()) {
      int oldSz = mThreads.size();
      mThreads.resize(nThreads);
      for (int i = oldSz; i < nThreads; i++) {
        mThreads[i] = std::make_unique<ClustererThread>(this);
      }
    }
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic, dynGrp)
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
      int chid = 0, thrStatIdx[nThreads];
      for (int ith = 0; ith < nThreads; ith++) {
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
            const auto clbeg = mThreads[ith]->compClusters.begin() + stat.firstClus;
            auto szold = compClus->size();
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
void Clusterer::ClustererThread::process(uint16_t chip, uint16_t nChips, CompClusCont* compClusPtr, PatternCont* patternsPtr,
                                         const ConstMCTruth* labelsDigPtr, MCTruth* labelsClPtr, const ROFRecord& rofPtr)
{
  if (stats.empty() || stats.back().firstChip + stats.back().nChips < chip) { // there is a jump, register new block
    stats.emplace_back(ThreadStat{chip, 0, uint32_t(compClusPtr->size()), patternsPtr ? uint32_t(patternsPtr->size()) : 0, 0, 0});
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
        initChip(curChipData, valp, chipID);
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
void Clusterer::ClustererThread::finishChip(ChipPixelData* curChipData, CompClusCont* compClusPtr,
                                            PatternCont* patternsPtr, const ConstMCTruth* labelsDigPtr, MCTruth* labelsClusPtr)
{
  auto clustersCount = compClusPtr->size();
  const auto& pixData = curChipData->getData();
  for (int i1 = 0; i1 < preClusterHeads.size(); ++i1) {
    auto ci = preClusterIndices[i1];
    if (ci < 0) {
      continue;
    }
    uint16_t rowMax = 0, rowMin = 65535;
    uint16_t colMax = 0, colMin = 65535;
    int nlab = 0;
    int next = preClusterHeads[i1];
    pixArrBuff.clear();
    while (next >= 0) {
      const auto& pixEntry = pixels[next];
      const auto pix = pixData[pixEntry.second];
      pixArrBuff.push_back(pix); // needed for cluster topology
      adjustBoundingBox(pix.getRowDirect(), pix.getCol(), rowMin, rowMax, colMin, colMax);
      if (labelsClusPtr) { // the MCtruth for this pixel is at curChipData->startID+pixEntry.second
        fetchMCLabels(pixEntry.second + curChipData->getStartID(), labelsDigPtr, nlab);
      }
      next = pixEntry.first;
    }
    preClusterIndices[i1] = -1;
    for (int i2 = i1 + 1; i2 < preClusterHeads.size(); ++i2) {
      if (preClusterIndices[i2] != ci) {
        continue;
      }
      next = preClusterHeads[i2];
      while (next >= 0) {
        const auto& pixEntry = pixels[next];
        const auto pix = pixData[pixEntry.second]; // PixelData
        pixArrBuff.push_back(pix);                 // needed for cluster topology
        adjustBoundingBox(pix.getRowDirect(), pix.getCol(), rowMin, rowMax, colMin, colMax);
        if (labelsClusPtr) { // the MCtruth for this pixel is at curChipData->startID+pixEntry.second
          fetchMCLabels(pixEntry.second + curChipData->getStartID(), labelsDigPtr, nlab);
        }
        next = pixEntry.first;
      }
      preClusterIndices[i2] = -1;
    }

    auto chipID = curChipData->getChipID();
    uint16_t colSpan = (colMax - colMin + 1);
    uint16_t rowSpan = (rowMax - rowMin + 1);
    if (colSpan <= o2::itsmft::ClusterPattern::MaxColSpan &&
        rowSpan <= o2::itsmft::ClusterPattern::MaxRowSpan) {
      streamCluster(pixArrBuff, rowMin, rowSpan, colMin, colSpan, chipID,
                    compClusPtr, patternsPtr, labelsClusPtr, nlab);
    } else {
      LOG(WARNING) << "Splitting a huge cluster !  ChipID: " << chipID;

      colSpan %= o2::itsmft::ClusterPattern::MaxColSpan;
      if (colSpan == 0) {
        colSpan = o2::itsmft::ClusterPattern::MaxColSpan;
      }

      rowSpan %= o2::itsmft::ClusterPattern::MaxRowSpan;
      if (rowSpan == 0) {
        rowSpan = o2::itsmft::ClusterPattern::MaxRowSpan;
      }

      do {
        uint16_t r = rowMin, rsp = rowSpan;

        do {
          // Select a subset of pixels fitting the reduced bounding box
          std::vector<PixelData> pixbuf;
          auto colMax = colMin + colSpan, rowMax = r + rsp;
          for (const auto& pix : pixArrBuff) {
            if (pix.getRowDirect() >= r && pix.getRowDirect() < rowMax &&
                pix.getCol() >= colMin && pix.getCol() < colMax) {
              pixbuf.push_back(pix);
            }
          }
          // Stream a piece of cluster only if the reduced bounding box is not empty
          if (!pixbuf.empty()) {
            streamCluster(pixbuf, r, rsp, colMin, colSpan, chipID,
                          compClusPtr, patternsPtr, labelsClusPtr, nlab, true);
          }
          r += rsp;
          rsp = o2::itsmft::ClusterPattern::MaxRowSpan;
        } while (r < rowMax);

        colMin += colSpan;
        colSpan = o2::itsmft::ClusterPattern::MaxColSpan;
      } while (colMin < colMax);
    }
  }
}

void Clusterer::ClustererThread::streamCluster(const std::vector<PixelData>& pixbuf, uint16_t rowMin, uint16_t rowSpanW, uint16_t colMin, uint16_t colSpanW, uint16_t chipID, CompClusCont* compClusPtr, PatternCont* patternsPtr, MCTruth* labelsClusPtr, int nlab, bool isHuge)
{
  if (labelsClusPtr) { // MC labels were requested
    auto cnt = compClusPtr->size();
    for (int i = nlab; i--;) {
      labelsClusPtr->addElement(cnt, labelsBuff[i]);
    }
  }

  // add to compact clusters, which must be always filled
  unsigned char patt[ClusterPattern::MaxPatternBytes] = {0}; // RSTODO FIX pattern filling
  for (const auto& pix : pixbuf) {
    unsigned short ir = pix.getRowDirect() - rowMin, ic = pix.getCol() - colMin;
    int nbits = ir * colSpanW + ic;
    patt[nbits >> 3] |= (0x1 << (7 - (nbits % 8)));
  }
  uint16_t pattID = (isHuge || parent->mPattIdConverter.size() == 0) ? CompCluster::InvalidPatternID : parent->mPattIdConverter.findGroupID(rowSpanW, colSpanW, patt);
  if (pattID == CompCluster::InvalidPatternID || parent->mPattIdConverter.isGroup(pattID)) {
    if (pattID != CompCluster::InvalidPatternID) {
      //For groupped topologies, the reference pixel is the COG pixel
      float xCOG = 0., zCOG = 0.;
      ClusterPattern::getCOG(rowSpanW, colSpanW, patt, xCOG, zCOG);
      rowMin += round(xCOG);
      colMin += round(zCOG);
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
  compClusPtr->emplace_back(rowMin, colMin, pattID, chipID);
}

//__________________________________________________
void Clusterer::ClustererThread::finishChipSingleHitFast(uint32_t hit, ChipPixelData* curChipData, CompClusCont* compClusPtr,
                                                         PatternCont* patternsPtr, const ConstMCTruth* labelsDigPtr, MCTruth* labelsClusPtr)
{
  auto clustersCount = compClusPtr->size();
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
  uint16_t pattID = (parent->mPattIdConverter.size() == 0) ? CompCluster::InvalidPatternID : parent->mPattIdConverter.findGroupID(1, 1, patt);
  if ((pattID == CompCluster::InvalidPatternID || parent->mPattIdConverter.isGroup(pattID)) && patternsPtr) {
    patternsPtr->emplace_back(1); // rowspan
    patternsPtr->emplace_back(1); // colspan
    patternsPtr->insert(patternsPtr->end(), std::begin(patt), std::begin(patt) + 1);
  }
  compClusPtr->emplace_back(row, col, pattID, curChipData->getChipID());
}

//__________________________________________________
Clusterer::Clusterer() : mPattIdConverter()
{
#ifdef _PERFORM_TIMING_
  mTimer.Stop();
  mTimer.Reset();
  mTimerMerge.Stop();
  mTimerMerge.Reset();
#endif
}

//__________________________________________________
void Clusterer::ClustererThread::initChip(const ChipPixelData* curChipData, uint32_t first, int chip)
{
  // init chip with the 1st unmasked pixel (entry "from" in the mChipData)
  size = itsmft::SegmentationAlpide::NRows + 2;
  if (chip < SegmentationSuperAlpide::NLayers) {
    SegmentationSuperAlpide seg(chip);
    size = seg.NRows + 2;
  }
  if (column1) {
    delete[] column1;
  }
  if (column2) {
    delete[] column2;
  }
  column1 = new int[size];
  column2 = new int[size];
  // init chip with the 1st unmasked pixel (entry "from" in the mChipData)
  prev = column1 + 1;
  curr = column2 + 1;
  resetColumn(curr);

  pixels.clear();
  preClusterHeads.clear();
  preClusterIndices.clear();
  auto pix = curChipData->getData()[first];
  currCol = pix.getCol();
  curr[pix.getRowDirect()] = 0; // can use getRowDirect since the pixel is not masked
  // start the first pre-cluster
  preClusterHeads.push_back(0);
  preClusterIndices.push_back(0);
  pixels.emplace_back(-1, first); // id of current pixel
  noLeftCol = true;               // flag that there is no column on the left to check yet
}

//__________________________________________________
void Clusterer::ClustererThread::updateChip(const ChipPixelData* curChipData, uint32_t ip)
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

  Bool_t orphan = true;

  if (noLeftCol) { // check only the row above
    if (curr[row - 1] >= 0) {
      expandPreCluster(ip, row, curr[row - 1]); // attach to the precluster of the previous row
      return;
    }
  } else {
    int neighbours[]{curr[row - 1], prev[row], prev[row + 1], prev[row - 1]};
    for (auto pci : neighbours) {
      if (pci < 0) {
        continue;
      }
      if (orphan) {
        expandPreCluster(ip, row, pci); // attach to the adjascent precluster
        orphan = false;
        continue;
      }
      // reassign precluster index to smallest one
      if (preClusterIndices[pci] < preClusterIndices[curr[row]]) {
        preClusterIndices[curr[row]] = preClusterIndices[pci];
      } else {
        preClusterIndices[pci] = preClusterIndices[curr[row]];
      }
    }
  }
  if (orphan) {
    addNewPrecluster(ip, row); // start new precluster
  }
}

//__________________________________________________
void Clusterer::ClustererThread::fetchMCLabels(int digID, const ConstMCTruth* labelsDig, int& nfilled)
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
void Clusterer::clear()
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
void Clusterer::print() const
{
  // print settings
  LOG(INFO) << "Clusterizer masks overflow pixels separated by < " << mMaxBCSeparationToMask << " BC and <= "
            << mMaxRowColDiffToMask << " in row/col";
#ifdef _PERFORM_TIMING_
  auto& tmr = const_cast<TStopwatch&>(mTimer); // ugly but this is what root does internally
  auto& tmrm = const_cast<TStopwatch&>(mTimerMerge);
  LOG(INFO) << "Inclusive clusterization timing (w/o disk IO): Cpu: " << tmr.CpuTime()
            << " Real: " << tmr.RealTime() << " s in " << tmr.Counter() << " slots";
  LOG(INFO) << "Threads output merging timing                : Cpu: " << tmrm.CpuTime()
            << " Real: " << tmrm.RealTime() << " s in " << tmrm.Counter() << " slots";

#endif
}

}