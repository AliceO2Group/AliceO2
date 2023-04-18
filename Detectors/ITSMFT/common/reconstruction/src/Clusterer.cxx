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

/// \file Clusterer.cxx
/// \brief Implementation of the ITS cluster finder
#include <algorithm>
#include <TTree.h>
#include "Framework/Logger.h"
#include "ITSMFTBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonDataFormat/InteractionRecord.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif
using namespace o2::itsmft;

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
  int rofcount{0};
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
    if (nFired < nThreads) {
      nThreads = nFired;
    }
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
      int chid = 0, thrStatIdx[nThreads];
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
  if (stats.empty() || stats.back().firstChip + stats.back().nChips != chip) { // there is a jump, register new block
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
    auto nclus0 = compClusPtr->size();
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
void Clusterer::ClustererThread::finishChip(ChipPixelData* curChipData, CompClusCont* compClusPtr,
                                            PatternCont* patternsPtr, const ConstMCTruth* labelsDigPtr, MCTruth* labelsClusPtr)
{
  const auto& pixData = curChipData->getData();
  for (int i1 = 0; i1 < preClusterHeads.size(); ++i1) {
    auto ci = preClusterIndices[i1];
    if (ci < 0) {
      continue;
    }
    BBox bbox(curChipData->getChipID());
    int nlab = 0;
    int next = preClusterHeads[i1];
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
      preClusterIndices[i2] = -1;
    }
    if (bbox.isAcceptableSize()) {
      parent->streamCluster(pixArrBuff, &labelsBuff, bbox, parent->mPattIdConverter, compClusPtr, patternsPtr, labelsClusPtr, nlab);
    } else {
      auto warnLeft = MaxHugeClusWarn - parent->mNHugeClus;
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

//__________________________________________________
void Clusterer::ClustererThread::finishChipSingleHitFast(uint32_t hit, ChipPixelData* curChipData, CompClusCont* compClusPtr,
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
void Clusterer::ClustererThread::initChip(const ChipPixelData* curChipData, uint32_t first)
{
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
#ifdef _ALLOW_DIAGONAL_ALPIDE_CLUSTERS_
    int neighbours[]{curr[row - 1], prev[row], prev[row + 1], prev[row - 1]};
#else
    int neighbours[]{curr[row - 1], prev[row]};
#endif
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
  LOGP(info, "Clusterizer squashes overflow pixels separated by {} BC and <= {} in row/col seeking down to {} neighbour ROFs", mMaxBCSeparationToSquash, mMaxRowColDiffToMask, mSquashingDepth);
  LOG(info) << "Clusterizer masks overflow pixels separated by < " << mMaxBCSeparationToMask << " BC and <= "
            << mMaxRowColDiffToMask << " in row/col";

#ifdef _PERFORM_TIMING_
  auto& tmr = const_cast<TStopwatch&>(mTimer); // ugly but this is what root does internally
  auto& tmrm = const_cast<TStopwatch&>(mTimerMerge);
  LOG(info) << "Inclusive clusterization timing (w/o disk IO): Cpu: " << tmr.CpuTime()
            << " Real: " << tmr.RealTime() << " s in " << tmr.Counter() << " slots";
  LOG(info) << "Threads output merging timing                : Cpu: " << tmrm.CpuTime()
            << " Real: " << tmrm.RealTime() << " s in " << tmrm.Counter() << " slots";

#endif
}

//__________________________________________________
void Clusterer::reset()
{
  // reset for new run
  clear();
  mNHugeClus = 0;
}
