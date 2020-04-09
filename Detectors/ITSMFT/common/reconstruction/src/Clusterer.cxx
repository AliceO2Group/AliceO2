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
#include "Framework/Logger.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::itsmft;

Clusterer::~Clusterer()
{
  print();
}

//__________________________________________________
Clusterer::Clusterer() : mPattIdConverter(), mCurr(mColumn2 + 1), mPrev(mColumn1 + 1)
{
  std::fill(std::begin(mColumn1), std::end(mColumn1), -1);
  std::fill(std::begin(mColumn2), std::end(mColumn2), -1);
  mROFRef.clear();
#ifdef _ClusterTopology_
  LOG(INFO) << "*********************************************************************";
  LOG(INFO) << "ATTENTION: YOU ARE RUNNING IN SPECIAL MODE OF STORING CLUSTER PATTERN";
  LOG(INFO) << "*********************************************************************";
#endif //_ClusterTopology_

#ifdef _PERFORM_TIMING_
  mTimer.Stop();
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
    int neighbours[]{mCurr[row - 1], mPrev[row], mPrev[row + 1], mPrev[row - 1]};
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
#ifdef _PERFORM_TIMING_
  printf("Clusterization timing (w/o disk IO): ");
  mTimer.Print();
#endif
}
