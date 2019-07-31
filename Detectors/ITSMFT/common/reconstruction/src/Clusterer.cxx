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
#include "FairLogger.h" // for LOG

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::itsmft;
using Segmentation = o2::itsmft::SegmentationAlpide;

//__________________________________________________
Clusterer::Clusterer() : mPattIdConverter(), mCurr(mColumn2 + 1), mPrev(mColumn1 + 1)
{
  std::fill(std::begin(mColumn1), std::end(mColumn1), -1);
  std::fill(std::begin(mColumn2), std::end(mColumn2), -1);
  mROFRef.clear();
#ifdef _ClusterTopology_
  LOG(INFO) << "*********************************************************************" << FairLogger::endl;
  LOG(INFO) << "ATTENTION: YOU ARE RUNNING IN SPECIAL MODE OF STORING CLUSTER PATTERN" << FairLogger::endl;
  LOG(INFO) << "*********************************************************************" << FairLogger::endl;
#endif //_ClusterTopology_

#ifdef _PERFORM_TIMING_
  mTimer.Stop();
#endif
}

//__________________________________________________
void Clusterer::process(PixelReader& reader, std::vector<Cluster>* fullClus,
                        std::vector<CompClusterExt>* compClus, MCTruth* labelsCl,
                        std::vector<o2::itsmft::ROFRecord>* vecROFRec)
{

#ifdef _PERFORM_TIMING_
  mTimer.Start(kFALSE);
#endif

  mClustersCount = compClus ? compClus->size() : (fullClus ? fullClus->size() : 0);

  auto& currROFIR = mROFRef.getBCData();
  auto& currROFEntry = mROFRef.getROFEntry();

  while ((mChipData = reader.getNextChipData(mChips))) { // read next chip data to corresponding
    // vector in the mChips and return the pointer on it

    if (!(mChipData->getInteractionRecord() == currROFIR)) { // new ROF starts

      mROFRef.setNROFEntries(mClustersCount - currROFEntry.getIndex()); // number of entries in this ROF

      if (!currROFIR.isDummy()) {
        if (mClusTree) { // if necessary, flush existing data
          LOG(INFO) << "ITS: clusterizing new ROFrame, Orbit :" << mChipData->getInteractionRecord().orbit
                    << " BC: " << mChipData->getInteractionRecord().bc;
          mROFRef.getROFEntry().setEvent(mClusTree->GetEntries());
          flushClusters(fullClus, compClus, labelsCl);
        }
        if (vecROFRec) {
          vecROFRec->emplace_back(mROFRef);
        }
      }
      currROFEntry.setIndex(mClustersCount);
      currROFIR = mChipData->getInteractionRecord();
      mROFRef.setROFrame(mChipData->getROFrame()); // TODO: outphase this
    }

    auto chipID = mChipData->getChipID();
    // LOG(DEBUG) << "ITSClusterer got Chip " << chipID << " ROFrame " << mChipData->getROFrame()
    //            << " Nhits " << mChipData->getData().size() << FairLogger::endl;

    if (mMaxBCSeparationToMask > 0) { // mask pixels fired from the previous ROF
      if (mChipsOld.size() < mChips.size()) {
        mChipsOld.resize(mChips.size()); // expand buffer of previous ROF data
      }
      const auto& chipInPrevROF = mChipsOld[chipID];
      if (std::abs(currROFIR.differenceInBC(chipInPrevROF.getInteractionRecord())) < mMaxBCSeparationToMask) {
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
      finishChip(fullClus, compClus, reader.getDigitsMCTruth(), labelsCl);
    }
    if (mMaxBCSeparationToMask > 0) { // current chip data will be used in the next ROF to mask overflow pixels
      mChipsOld[chipID].swap(*mChipData);
    }
  }
  mROFRef.setNROFEntries(mClustersCount - currROFEntry.getIndex()); // number of entries in this ROF

  // flush last ROF
  if (!currROFIR.isDummy()) {
    if (mClusTree) { // if necessary, flush existing data
      mROFRef.getROFEntry().setEvent(mClusTree->GetEntries());
      flushClusters(fullClus, compClus, labelsCl);
    }
    if (vecROFRec) {
      vecROFRec->emplace_back(mROFRef); // the ROFrecords vector is stored outside, in a single entry of the tree
    }
  }
#ifdef _PERFORM_TIMING_
  mTimer.Stop();
  printf("Clusterization timing (w/o disk IO): ");
  mTimer.Print();
#endif
}

//__________________________________________________
void Clusterer::initChip(UInt_t first)
{
  // init chip with the 1st unmasked pixel (entry "from" in the mChipData)
  mPrev = mColumn1 + 1;
  mCurr = mColumn2 + 1;
  resetColumn(mCurr);

  mPixels.clear();
  mPreClusterHeads.clear();
  mPreClusterIndices.clear();
  auto pix = mChipData->getData()[first];
  mCol = pix.getCol();

  //addNewPrecluster(first, pix.getRowDirect()); // save on .size() calls ?
  mCurr[pix.getRowDirect()] = 0; // can use getRowDirect since the pixel is not masked
  // start the first pre-cluster
  mPreClusterHeads.push_back(0);
  mPreClusterIndices.push_back(0);
  mPixels.emplace_back(-1, first); // id of current pixel
  mNoLeftColumn = true;            // flag that there is no column on the left to check yet
}

//__________________________________________________
void Clusterer::updateChip(UInt_t ip)
{
  const auto pix = mChipData->getData()[ip];
  UShort_t row = pix.getRowDirect(); // can use getRowDirect since the pixel is not masked
  if (mCol != pix.getCol()) {        // switch the buffers
    swapColumnBuffers();
    resetColumn(mCurr);
    mNoLeftColumn = false;
    if (pix.getCol() > mCol + 1) {
      // no connection with previous column, this pixel cannot belong to any of the
      // existing preclusters, create a new precluster and flag to check only the row above for next pixels of this column
      mCol = pix.getCol();
      addNewPrecluster(ip, row);
      mNoLeftColumn = true;
      return;
    }
    mCol = pix.getCol();
  }

  Bool_t orphan = true;

  if (mNoLeftColumn) { // check only the row above
    if (mCurr[row - 1] >= 0) {
      expandPreCluster(ip, row, mCurr[row - 1]); // attach to the precluster of the previous row
      return;
    }
  } else {
    int neighbours[]{ mCurr[row - 1], mPrev[row], mPrev[row + 1], mPrev[row - 1] };
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
      if (mPreClusterIndices[pci] < mPreClusterIndices[mCurr[row]]) {
        mPreClusterIndices[mCurr[row]] = mPreClusterIndices[pci];
      } else {
        mPreClusterIndices[pci] = mPreClusterIndices[mCurr[row]];
      }
    }
  }
  if (orphan) {
    addNewPrecluster(ip, row); // start new precluster
  }
}

//__________________________________________________
void Clusterer::finishChip(std::vector<Cluster>* fullClus, std::vector<CompClusterExt>* compClus,
                           const MCTruth* labelsDig, MCTruth* labelsClus)
{
  constexpr Float_t SigmaX2 = Segmentation::PitchRow * Segmentation::PitchRow / 12.; // FIXME
  constexpr Float_t SigmaY2 = Segmentation::PitchCol * Segmentation::PitchCol / 12.; // FIXME

  const auto& pixData = mChipData->getData();

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
      } else {
        LOG(ERROR) << "Cluster size " << npix + 1 << " exceeds the buffer size" << FairLogger::endl;
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
        } else {
          LOG(ERROR) << "Cluster size " << npix + 1 << " exceeds the buffer size" << FairLogger::endl;
        }
      }
      mPreClusterIndices[i2] = -1;
    }
    UShort_t rowSpan = rowMax - rowMin + 1, colSpan = colMax - colMin + 1;
    Cluster clus;
    clus.setROFrame(mChipData->getROFrame());
    clus.setSensorID(mChipData->getChipID());
    clus.setNxNzN(rowSpan, colSpan, npix);
#ifdef _ClusterTopology_
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
      Segmentation::detectorToLocalUnchecked(x / npix, z / npix, xyzLoc);
      auto xyzTra = mGeometry->getMatrixT2L(mChipData->getChipID()) ^ (xyzLoc); // inverse transform from Local to Tracking frame
      c.setPos(xyzTra);
      c.setErrors(SigmaX2, SigmaY2, 0.f);
    }

    if (compClus) { // store compact clusters
      unsigned char patt[Cluster::kMaxPatternBytes];
      clus.getPattern(&patt[0], Cluster::kMaxPatternBytes);
      UShort_t pattID = mPattIdConverter.findGroupID(clus.getPatternRowSpan(), clus.getPatternColSpan(), patt);
      if (mPattIdConverter.IsGroup(pattID)) {
        int rowShift = 0, colShift = 0;
        ClusterTopology::getCOGshift(clus.getPatternRowSpan(), clus.getPatternColSpan(), patt, rowShift, colShift);
        rowMin += rowShift;
        colMin += colShift;
      }
      compClus->emplace_back(rowMin, colMin, pattID, mChipData->getChipID(), mChipData->getROFrame());
    }

    if (labelsClus) { // MC labels were requested
      for (int i = nlab; i--;) {
        labelsClus->addElement(mClustersCount, mLabelsBuff[i]);
      }
    }

    mClustersCount++;
  }
}

//__________________________________________________
void Clusterer::fetchMCLabels(int digID, const MCTruth* labelsDig, int& nfilled)
{
  // transfer MC labels to cluster
  if (nfilled >= Cluster::maxLabels) {
    return;
  }
  const auto& lbls = labelsDig->getLabels(digID);
  for (int i = lbls.size(); i--;) {
    int ic = nfilled;
    for (; ic--;) { // check if the label is already present
      if (mLabelsBuff[ic] == lbls[i]) {
        return; // label is found, do nothing
      }
    }
    mLabelsBuff[nfilled++] = lbls[i];
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
  mChipData = nullptr;
  mClusTree = nullptr;
  mROFRef.clear();
  mTimer.Stop();
  mTimer.Reset();
}

//__________________________________________________
void Clusterer::print() const
{
  // print settings
  printf("Mask overflow pixels in strobes separated by < %d BCs\n", mMaxBCSeparationToMask);
}
