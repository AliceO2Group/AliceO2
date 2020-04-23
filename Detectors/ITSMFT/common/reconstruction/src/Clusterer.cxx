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
#include "ITSMFTReconstruction/Clusterer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonDataFormat/InteractionRecord.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::itsmft;

//__________________________________________________
void Clusterer::process(int nThreads, PixelReader& reader, FullClusCont* fullClus, CompClusCont* compClus,
                        PatternCont* patterns, ROFRecCont* vecROFRec, MCTruth* labelsCl)
{
#ifdef _PERFORM_TIMING_
  mTimer.Start(kFALSE);
#endif
  constexpr int LoopPerThread = 4; // in MT mode run so many times more loops than the threads allowed
  auto autoDecode = reader.getDecodeNextAuto();
  o2::itsmft::ROFRecord* rof = nullptr;
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
    int nFired = mFiredChipsPtr.size();
    if (!nFired) {
      break;
    }
    auto clustersCount = compClus->size();                                                                                        // RSTODO: in principle, the compClus is never supposed to be 0
    rof = &vecROFRec->emplace_back(mFiredChipsPtr[0]->getInteractionRecord(), mFiredChipsPtr[0]->getROFrame(), clustersCount, 0); // create new ROF
    if (nFired < nThreads) {
      nThreads = nFired;
    }
    int nLoops = nThreads;
#ifdef WITH_OPENMP
    if (nThreads > 0) {
      omp_set_num_threads(nThreads);
    } else {
      nThreads = omp_get_num_threads(); // RSTODO I guess the system may decide to provide less threads than asked?
    }
    nLoops = nThreads == 1 ? 1 : std::min(nFired, LoopPerThread * nThreads);
    std::vector<int> loopLim;
    loopLim.reserve(nLoops + nLoops);
    loopLim.push_back(0);
    // decide actual workshare between the threads, trying to process the same number of pixels in each
    size_t nAvPixPerLoop = nPix / nLoops, smt = 0;
    for (int i = 0; i < nFired; i++) {
      smt += mFiredChipsPtr[i]->getData().size();
      if (smt >= nAvPixPerLoop) { // define threads boundary
        loopLim.push_back(i);
        smt = 0;
      }
    }
    if (loopLim.back() != nFired) {
      loopLim.push_back(nFired);
    }
    nLoops = loopLim.size() - 1;
    if (nThreads > nLoops) { // this should not happen
      omp_set_num_threads(nLoops);
    }
#else
    nThreads = nLoops = 1;
    std::vector<int> loopLim{0, nFired};
#endif
    if (nLoops > mThreads.size()) {
      int oldSz = mThreads.size();
      mThreads.resize(nLoops);
      for (int i = oldSz; i < nLoops; i++) {
        mThreads[i] = std::make_unique<ClustererThread>(this);
      }
    }
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
    //>> start of MT region
    for (int ith = 0; ith < nLoops; ith++) { // each loop is done by a separate thread
      auto chips = gsl::span(&mFiredChipsPtr[loopLim[ith]], loopLim[ith + 1] - loopLim[ith]);
      if (!ith) { // the 1st thread can write directly to the final destination
        mThreads[ith]->process(chips, fullClus, compClus, patterns, labelsCl ? reader.getDigitsMCTruth() : nullptr, labelsCl, rof);
      } else { // extra threads will store in their own containers
        mThreads[ith]->process(chips,
                               fullClus ? &mThreads[ith]->fullClusters : nullptr,
                               &mThreads[ith]->compClusters,
                               patterns ? &mThreads[ith]->patterns : nullptr,
                               labelsCl ? reader.getDigitsMCTruth() : nullptr,
                               labelsCl ? &mThreads[ith]->labels : nullptr, rof);
      }
    }
    //<< end of MT region

    // copy data of all threads but the 1st one to final destination
    if (nLoops > 1) {
#ifdef _PERFORM_TIMING_
      mTimerMerge.Start(false);
#endif
      size_t nClTot = compClus->size(), nPattTot = patterns ? patterns->size() : 0;
      for (int ith = 1; ith < nLoops; ith++) {
        nClTot += mThreads[ith]->compClusters.size();
        nPattTot += mThreads[ith]->patterns.size();
      }
      compClus->reserve(nClTot);
      if (fullClus) {
        fullClus->reserve(nClTot);
      }
      if (patterns) {
        patterns->reserve(nPattTot);
      }
      for (int ith = 1; ith < nLoops; ith++) {
        compClus->insert(compClus->end(), mThreads[ith]->compClusters.begin(), mThreads[ith]->compClusters.end());
        mThreads[ith]->compClusters.clear();

        if (fullClus) {
          fullClus->insert(fullClus->end(), mThreads[ith]->fullClusters.begin(), mThreads[ith]->fullClusters.end());
          mThreads[ith]->fullClusters.clear();
        }
        if (patterns) {
          patterns->insert(patterns->end(), mThreads[ith]->patterns.begin(), mThreads[ith]->patterns.end());
          mThreads[ith]->patterns.clear();
        }
        if (labelsCl) {
          labelsCl->mergeAtBack(mThreads[ith]->labels);
          mThreads[ith]->labels.clear();
        }
      }
#ifdef _PERFORM_TIMING_
      mTimerMerge.Stop();
#endif
    }
    // finalize last ROF
    if (rof) {
      auto cntUpd = compClus->size();
      rof->setNEntries(cntUpd - clustersCount); // update
      if (mClusTree) {                          // if necessary, flush existing data, legacy code, to remove
        mROFRef = *rof;
        flushClusters(fullClus, compClus, labelsCl);
      }
    }
  } while (autoDecode);

  reader.setDecodeNextAuto(autoDecode); // restore setting
#ifdef _PERFORM_TIMING_
  mTimer.Stop();
#endif
}

//__________________________________________________
void Clusterer::ClustererThread::process(gsl::span<ChipPixelData*> chipPtrs, FullClusCont* fullClusPtr, CompClusCont* compClusPtr, PatternCont* patternsPtr,
                                         const MCTruth* labelsDigPtr, MCTruth* labelsClPtr, const ROFRecord* rofPtr)
{
  int nch = chipPtrs.size();
  for (auto curChipData : chipPtrs) {
    auto chipID = curChipData->getChipID();
    if (parent->mMaxBCSeparationToMask > 0) { // mask pixels fired from the previous ROF
      const auto& chipInPrevROF = parent->mChipsOld[chipID];
      if (std::abs(rofPtr->getBCData().differenceInBC(chipInPrevROF.getInteractionRecord())) < parent->mMaxBCSeparationToMask) {
        curChipData->maskFiredInSample(parent->mChipsOld[chipID]);
      }
    }
    auto validPixID = curChipData->getFirstUnmasked();
    auto npix = curChipData->getData().size();
    if (validPixID < npix) { // chip data may have all of its pixels masked!
      auto valp = validPixID++;
      if (validPixID == npix) { // special case of a single pixel fired on the chip
        finishChipSingleHitFast(valp, curChipData, fullClusPtr, compClusPtr, patternsPtr, labelsDigPtr, labelsClPtr);
      } else {
        initChip(curChipData, valp);
        for (; validPixID < npix; validPixID++) {
          if (!curChipData->getData()[validPixID].isMasked()) {
            updateChip(curChipData, validPixID);
          }
        }
        finishChip(curChipData, fullClusPtr, compClusPtr, patternsPtr, labelsDigPtr, labelsClPtr);
      }
    }
    if (parent->mMaxBCSeparationToMask > 0) { // current chip data will be used in the next ROF to mask overflow pixels
      parent->mChipsOld[chipID].swap(*curChipData);
    }
  }
}

//__________________________________________________
void Clusterer::ClustererThread::finishChip(ChipPixelData* curChipData, FullClusCont* fullClusPtr, CompClusCont* compClusPtr,
                                            PatternCont* patternsPtr, const MCTruth* labelsDigPtr, MCTruth* labelsClusPTr)
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
    int nlab = 0, npix = 0;
    int next = preClusterHeads[i1];
    while (next >= 0) {
      const auto& pixEntry = pixels[next];
      const auto pix = pixData[pixEntry.second];
      if (npix < pixArrBuff.size()) {
        pixArrBuff[npix++] = pix; // needed for cluster topology
        adjustBoundingBox(pix.getRowDirect(), pix.getCol(), rowMin, rowMax, colMin, colMax);
        if (labelsClusPTr) { // the MCtruth for this pixel is at curChipData->startID+pixEntry.second
          fetchMCLabels(pixEntry.second + curChipData->getStartID(), labelsDigPtr, nlab);
        }
        next = pixEntry.first;
      }
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
        if (npix < pixArrBuff.size()) {
          pixArrBuff[npix++] = pix; // needed for cluster topology
          adjustBoundingBox(pix.getRowDirect(), pix.getCol(), rowMin, rowMax, colMin, colMax);
          if (labelsClusPTr) { // the MCtruth for this pixel is at curChipData->startID+pixEntry.second
            fetchMCLabels(pixEntry.second + curChipData->getStartID(), labelsDigPtr, nlab);
          }
          next = pixEntry.first;
        }
      }
      preClusterIndices[i2] = -1;
    }
    uint16_t rowSpan = rowMax - rowMin + 1, colSpan = colMax - colMin + 1;
    Cluster clus;
    clus.setSensorID(curChipData->getChipID());
    clus.setNxNzN(rowSpan, colSpan, npix);
    uint16_t colSpanW = colSpan, rowSpanW = rowSpan;
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
      const auto pix = pixArrBuff[i];
      unsigned short ir = pix.getRowDirect() - rowMin, ic = pix.getCol() - colMin;
      if (ir < rowSpanW && ic < colSpanW) {
        clus.setPixel(ir, ic);
      }
    }
#endif                 //_ClusterTopology_
    if (fullClusPtr) { // do we need conventional clusters with full topology and coordinates?
      fullClusPtr->push_back(clus);
      Cluster& c = fullClusPtr->back();
      float x = 0., z = 0.;
      for (int i = npix; i--;) {
        x += pixArrBuff[i].getRowDirect();
        z += pixArrBuff[i].getCol();
      }
      Point3D<float> xyzLoc;
      SegmentationAlpide::detectorToLocalUnchecked(x / npix, z / npix, xyzLoc);
      auto xyzTra = parent->mGeometry->getMatrixT2L(curChipData->getChipID()) ^ (xyzLoc); // inverse transform from Local to Tracking frame
      c.setPos(xyzTra);
      c.setErrors(SigmaX2, SigmaY2, 0.f);
    }

    if (labelsClusPTr) { // MC labels were requested
      auto cnt = compClusPtr->size();
      for (int i = nlab; i--;) {
        labelsClusPTr->addElement(cnt, labelsBuff[i]);
      }
    }

    // add to compact clusters, which must be always filled
    unsigned char patt[Cluster::kMaxPatternBytes] = {0}; // RSTODO FIX pattern filling
    for (int i = 0; i < npix; i++) {
      const auto pix = pixArrBuff[i];
      unsigned short ir = pix.getRowDirect() - rowMin, ic = pix.getCol() - colMin;
      if (ir < rowSpanW && ic < colSpanW) {
        int nbits = ir * colSpanW + ic;
        patt[nbits >> 3] |= (0x1 << (7 - (nbits % 8)));
      }
    }
    uint16_t pattID = (parent->mPattIdConverter.size() == 0) ? CompCluster::InvalidPatternID : parent->mPattIdConverter.findGroupID(rowSpanW, colSpanW, patt);
    if (pattID == CompCluster::InvalidPatternID || parent->mPattIdConverter.isGroup(pattID)) {
      float xCOG = 0., zCOG = 0.;
      ClusterPattern::getCOG(rowSpanW, colSpanW, patt, xCOG, zCOG);
      rowMin += round(xCOG);
      colMin += round(zCOG);
      if (patternsPtr) {
        patternsPtr->emplace_back((unsigned char)rowSpanW);
        patternsPtr->emplace_back((unsigned char)colSpanW);
        int nBytes = rowSpanW * colSpanW / 8;
        if (((rowSpanW * colSpanW) % 8) != 0)
          nBytes++;
        patternsPtr->insert(patternsPtr->end(), std::begin(patt), std::begin(patt) + nBytes);
      }
    }
    compClusPtr->emplace_back(rowMin, colMin, pattID, curChipData->getChipID());
  }
}

//__________________________________________________
void Clusterer::ClustererThread::finishChipSingleHitFast(uint32_t hit, ChipPixelData* curChipData, FullClusCont* fullClusPtr,
                                                         CompClusCont* compClusPtr, PatternCont* patternsPtr, const MCTruth* labelsDigPtr, MCTruth* labelsClusPTr)
{
  auto clustersCount = compClusPtr->size();
  auto pix = curChipData->getData()[hit];
  uint16_t row = pix.getRowDirect(), col = pix.getCol();

  if (fullClusPtr) { // do we need conventional clusters with full topology and coordinates?
    Cluster clus;
    clus.setSensorID(curChipData->getChipID());
    clus.setNxNzN(1, 1, 1);
#ifdef _ClusterTopology_
    clus.setPatternRowSpan(1, false);
    clus.setPatternColSpan(1, false);
    clus.setPatternRowMin(row);
    clus.setPatternColMin(col);
    clus.setPixel(0, 0);
#endif //_ClusterTopology_
    fullClusPtr->push_back(clus);
    Cluster& c = fullClusPtr->back();
    Point3D<float> xyzLoc;
    SegmentationAlpide::detectorToLocalUnchecked(row, col, xyzLoc);                     // implicit conversion of row,col to floats!
    auto xyzTra = parent->mGeometry->getMatrixT2L(curChipData->getChipID()) ^ (xyzLoc); // inverse transform from Local to Tracking frame
    c.setPos(xyzTra);
    c.setErrors(SigmaX2, SigmaY2, 0.f);
  }

  if (labelsClusPTr) { // MC labels were requested
    int nlab = 0;
    fetchMCLabels(curChipData->getStartID() + hit, labelsDigPtr, nlab);
    auto cnt = compClusPtr->size();
    for (int i = nlab; i--;) {
      labelsClusPTr->addElement(cnt, labelsBuff[i]);
    }
  }

  // add to compact clusters, which must be always filled
  unsigned char patt[Cluster::kMaxPatternBytes]{0x1 << (7 - (0 % 8))}; // unrolled 1 hit version of full loop in finishChip
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
  mROFRef.clear();
#ifdef _ClusterTopology_
  LOG(INFO) << "*********************************************************************";
  LOG(INFO) << "ATTENTION: YOU ARE RUNNING IN SPECIAL MODE OF STORING CLUSTER PATTERN";
  LOG(INFO) << "*********************************************************************";
#endif //_ClusterTopology_

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
void Clusterer::ClustererThread::fetchMCLabels(int digID, const MCTruth* labelsDig, int& nfilled)
{
  // transfer MC labels to cluster
  if (nfilled >= Cluster::maxLabels) {
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
    if (nfilled >= Cluster::maxLabels) {
      break;
    }
  }
  //
}

//__________________________________________________
void Clusterer::clear()
{
  // reset
  mClusTree = nullptr;
  mROFRef.clear();
#ifdef _PERFORM_TIMING_
  mTimer.Stop();
  mTimer.Reset();
  mTimerMerge.Stop();
  mTimerMerge.Reset();
#endif
}

///< flush cluster data accumulated so far into the tree, this method should be NEVER used in MT-mode
//__________________________________________________
void Clusterer::flushClusters(FullClusCont* fullClus, CompClusCont* compClus, MCTruth* labels)
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

//__________________________________________________
void Clusterer::print() const
{
  // print settings
  LOG(INFO) << "Clusterizer masks overflow pixels in strobes separated by < " << mMaxBCSeparationToMask << " BC";
#ifdef _PERFORM_TIMING_
  auto& tmr = const_cast<TStopwatch&>(mTimer); // ugly but this is what root does internally
  auto& tmrm = const_cast<TStopwatch&>(mTimerMerge);
  LOG(INFO) << "Inclusive clusterization timing (w/o disk IO): Cpu: " << tmr.CpuTime()
            << " Real: " << tmr.RealTime() << " s in " << tmr.Counter() << " slots";
  LOG(INFO) << "Threads output merging timing                : Cpu: " << tmrm.CpuTime()
            << " Real: " << tmrm.RealTime() << " s in " << tmrm.Counter() << " slots";

#endif
}
