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
#include "FairLogger.h" // for LOG

#include "ITSMFTBase/SegmentationAlpide.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::ITSMFT;
using Segmentation = o2::ITSMFT::SegmentationAlpide;

//__________________________________________________
Clusterer::Clusterer() : mCurr(mColumn2 + 1), mPrev(mColumn1 + 1)
{
  std::fill(std::begin(mColumn1), std::end(mColumn1), -1);
  std::fill(std::begin(mColumn2), std::end(mColumn2), -1);

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
void Clusterer::process(PixelReader& reader, std::vector<Cluster>& clusters, MCTruth* labelsCl)
{

#ifdef _PERFORM_TIMING_
  mTimer.Start(kFALSE);
#endif

  UInt_t prevROF = o2::ITSMFT::PixelData::DummyROF;
  while ((mChipData = reader.getNextChipData(mChips))) { // read next chip data to corresponding
    // vector in the mChips and return the pointer on it

    mCurrROF = mChipData->getROFrame();

    if (prevROF != mCurrROF) {
      if (mClusTree && prevROF != o2::ITSMFT::PixelData::DummyROF) { // if necessary, flush existing data
        flushClusters(clusters, labelsCl);
      }
      prevROF = mCurrROF;
    }

    if (mChipData->getChipID() < mCurrChipID) {
      LOG(INFO) << "ITS: clusterizing new ROFrame " << mCurrROF << FairLogger::endl;
    }
    mCurrChipID = mChipData->getChipID();
    // LOG(DEBUG) << "ITSClusterer got Chip " << mCurrChipID << " ROFrame " << mChipData->getROFrame()
    //            << " Nhits " << mChipData->getData().size() << FairLogger::endl;

    if (mMaskOverflowPixels) { // mask pixels fired from the previous ROF
      if (mChipsOld.size() < mChips.size()) {
        mChipsOld.resize(mChips.size()); // expand buffer of previous ROF data
      }
      const auto& chipInPrevROF = mChipsOld[mCurrChipID];
      if (chipInPrevROF.getROFrame() + 1 == mCurrROF) {
        mChipData->maskFiredInSample(mChipsOld[mCurrChipID]);
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
      finishChip(clusters, reader.getDigitsMCTruth(), labelsCl);
    }
    if (mMaskOverflowPixels) { // current chip data will be used in the next ROF to mask overflow pixels
      mChipsOld[mCurrChipID].swap(*mChipData);
    }
  }

  // if asked, flush last ROF
  if (mClusTree && prevROF != o2::ITSMFT::PixelData::DummyROF) { // if necessary, flush existing data
    flushClusters(clusters, labelsCl);
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
  std::fill(std::begin(mColumn1), std::end(mColumn1), -1);
  std::fill(std::begin(mColumn2), std::end(mColumn2), -1);

  mPixels.clear();
  mPreClusterHeads.clear();
  mPreClusterIndices.clear();
  auto pix = mChipData->getData()[first];
  mCol = pix.getCol();
  mCurr[pix.getRowDirect()] = 0; // can use getRowDirect since the pixel is not masked
  // start the first pre-cluster
  mPreClusterHeads.push_back(0);
  mPreClusterIndices.push_back(0);
  mPixels.emplace_back(-1, first); // id of current pixel
}

//__________________________________________________
void Clusterer::updateChip(UInt_t ip)
{
  const auto pix = mChipData->getData()[ip];
  if (mCol != pix.getCol()) { // switch the buffers
    Int_t* tmp = mCurr;
    mCurr = mPrev;
    mPrev = tmp;
    if (pix.getCol() > mCol + 1)
      std::fill(mPrev, mPrev + kMaxRow, -1);
    std::fill(mCurr, mCurr + kMaxRow, -1);
    mCol = pix.getCol();
  }

  Bool_t attached = false;
  UShort_t row = pix.getRowDirect(); // can use getRowDirect since the pixel is not masked
  Int_t neighbours[]{ mCurr[row - 1], mPrev[row], mPrev[row + 1], mPrev[row - 1] };
  for (auto pci : neighbours) {
    if (pci < 0)
      continue;
    auto& ci = mPreClusterIndices[pci];
    if (attached) {
      auto& newci = mPreClusterIndices[mCurr[row]];
      if (ci < newci)
        newci = ci;
      else
        ci = newci;
    } else {
      auto& firstIndex = mPreClusterHeads[ci];
      mPixels.emplace_back(firstIndex, ip);
      firstIndex = mPixels.size() - 1;
      mCurr[row] = pci;
      attached = true;
    }
  }

  if (attached)
    return;

  // start new precluster
  mPreClusterHeads.push_back(mPixels.size());
  mPixels.emplace_back(-1, ip);
  Int_t lastIndex = mPreClusterIndices.size();
  mPreClusterIndices.push_back(lastIndex);
  mCurr[row] = lastIndex;
}

//__________________________________________________
void Clusterer::finishChip(std::vector<Cluster>& clusters, const MCTruth* labelsDig, MCTruth* labelsClus)
{
  constexpr Float_t SigmaX2 = Segmentation::PitchRow * Segmentation::PitchRow / 12.; // FIXME
  constexpr Float_t SigmaY2 = Segmentation::PitchCol * Segmentation::PitchCol / 12.; // FIXME

  const auto& pixData = mChipData->getData();
  Int_t noc = clusters.size();
  for (Int_t i1 = 0; i1 < mPreClusterHeads.size(); ++i1) {
    const auto ci = mPreClusterIndices[i1];
    if (ci < 0) {
      continue;
    }
    UShort_t rowMax = 0, rowMin = 65535;
    UShort_t colMax = 0, colMin = 65535;
    Float_t x = 0., z = 0.;
    int nlab = 0, npix = 0;
    Int_t next = mPreClusterHeads[i1];
    while (next >= 0) {
      const auto& dig = mPixels[next];
      const auto pix = pixData[dig.second];
      x += pix.getRowDirect();
      z += pix.getCol();
      if (pix.getRowDirect() < rowMin) {
        rowMin = pix.getRowDirect();
      }
      if (pix.getRowDirect() > rowMax) {
        rowMax = pix.getRowDirect();
      }
      if (pix.getCol() < colMin) {
        colMin = pix.getCol();
      }
      if (pix.getCol() > colMax) {
        colMax = pix.getCol();
      }
      if (npix < mPixArrBuff.size()) {
        mPixArrBuff[npix] = pix; // needed for cluster topology
      }
      if (labelsClus) { // the MCtruth for this pixel is at mChipData->startID+dig.second
        fetchMCLabels(dig.second + mChipData->getStartID(), labelsDig, nlab);
      }
      npix++;
      next = dig.first;
    }
    mPreClusterIndices[i1] = -1;
    for (Int_t i2 = i1 + 1; i2 < mPreClusterHeads.size(); ++i2) {
      if (mPreClusterIndices[i2] != ci) {
        continue;
      }
      next = mPreClusterHeads[i2];
      while (next >= 0) {
        const auto& dig = mPixels[next];
        const auto pix = pixData[dig.second]; // PixelData
        x += pix.getRowDirect();
        z += pix.getCol();
        if (pix.getRowDirect() < rowMin) {
          rowMin = pix.getRowDirect();
        }
        if (pix.getRowDirect() > rowMax) {
          rowMax = pix.getRowDirect();
        }
        if (pix.getCol() < colMin) {
          colMin = pix.getCol();
        }
        if (pix.getCol() > colMax) {
          colMax = pix.getCol();
        }
        if (npix < mPixArrBuff.size()) {
          mPixArrBuff[npix] = pix; // needed for cluster topology
        }
        if (labelsClus) { // the MCtruth for this pixel is at mChipData->startID+dig.second
          fetchMCLabels(dig.second + mChipData->getStartID(), labelsDig, nlab);
        }
        npix++;
        next = dig.first;
      }
      mPreClusterIndices[i2] = -1;
    }

    Point3D<float> xyzLoc(Segmentation::getFirstRowCoordinate() + x * Segmentation::PitchRow / npix, 0.f,
                          Segmentation::getFirstColCoordinate() + z * Segmentation::PitchCol / npix);
    auto xyzTra =
      mGeometry->getMatrixT2L(mChipData->getChipID()) ^ (xyzLoc); // inverse transform from Local to Tracking frame

    clusters.emplace_back();
    Cluster& c = clusters[noc];
    c.setROFrame(mChipData->getROFrame());
    c.setSensorID(mChipData->getChipID());
    c.setPos(xyzTra);
    c.setErrors(SigmaX2, SigmaY2, 0.f);
    c.setNxNzN(rowMax - rowMin + 1, colMax - colMin + 1, npix);
    if (labelsClus) { // MC labels were requested
      for (int i = nlab; i--;) {
        labelsClus->addElement(noc, mLabelsBuff[i]);
      }
    }
    noc++;

#ifdef _ClusterTopology_
    unsigned short colSpan = (colMax + 1 - colMin), rowSpan = (rowMax + 1 - rowMin), colSpanW = colSpan,
                   rowSpanW = rowSpan;
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
    c.setPatternRowSpan(rowSpanW, rowSpanW < rowSpan);
    c.setPatternColSpan(colSpanW, colSpanW < colSpan);
    c.setPatternRowMin(rowMin);
    c.setPatternColMin(colMin);
    if (npix > mPixArrBuff.size())
      npix = mPixArrBuff.size();
    for (int i = 0; i < npix; i++) {
      const auto pix = mPixArrBuff[i];
      unsigned short ir = pix.getRowDirect() - rowMin, ic = pix.getCol() - colMin;
      if (ir < rowSpanW && ic < colSpanW) {
        c.setPixel(ir, ic);
      }
    }
#endif //_ClusterTopology_
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
        break;
      }
    }
    if (ic < 0) { // label not found
      mLabelsBuff[nfilled++] = lbls[i];
      if (nfilled >= Cluster::maxLabels) {
        break;
      }
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
  mTimer.Stop();
  mTimer.Reset();
}

//__________________________________________________
void Clusterer::print() const
{
  // print settings
  printf("Masking of overflow pixels: %s\n", mMaskOverflowPixels ? "ON" : "OFF");
}
