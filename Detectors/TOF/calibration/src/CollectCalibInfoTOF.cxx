// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <TTree.h>
#include <cassert>

#include "FairLogger.h"
#include "TOFBase/Geo.h"

#include <TFile.h>
#include "TOFCalibration/CollectCalibInfoTOF.h"

using namespace o2::globaltracking;

ClassImp(CollectCalibInfoTOF);

//______________________________________________
void CollectCalibInfoTOF::run()
{
  ///< collecting the TOF calib info and writing them into a tree;
  ///< We will write to the tree as soon as there is one channel that
  ///< has at least MAXNUMBEROFHITS entries already accumulated; when this
  ///< happens, we will store the accumulated CalibInfo for all channels.
  ///< finally, we will write whatever is left at the end of the processing.

  if (!mInitDone) {
    LOG(FATAL) << "init() was not done yet";
  }

  mTimerTot.Start();

  while (loadTOFCalibInfo()) { // fill here all histos you need
    for (int ihit = 0; ihit < mTOFCalibInfo->size(); ihit++) {
      addHit((*mTOFCalibInfo)[ihit]);
    }
  }
  fillTree(); // filling with whatever we have in memory

  // fit and extract calibration parameters once histos are filled
  // ...

  mTimerTot.Stop();
  printf("Timing:\n");
  printf("Total:        ");
  mTimerTot.Print();
}

//______________________________________________
void CollectCalibInfoTOF::init()
{
  ///< initizalizations

  if (mInitDone) {
    LOG(ERROR) << "Initialization was already done";
    return;
  }

  attachInputTrees();

  // create output branch with track-tof matching
  if (mOutputTree) {
    mOutputTree->Branch(mOutputBranchName.data(), &mTOFCalibInfoOut);
    LOG(INFO) << "Accumulated calib info TOF will be stored in " << mOutputBranchName << " branch of tree "
              << mOutputTree->GetName();
  } else {
    LOG(ERROR) << "Output tree is not attached, accumulated CalibInfoTOFshort will not be stored";
  }
  mInitDone = true;

  {
    mTimerTot.Stop();
    mTimerTot.Reset();
  }

  print();
}

//______________________________________________
void CollectCalibInfoTOF::print() const
{
  ///< print the settings

  LOG(INFO) << "****** component for calibration of TOF channels ******";
  if (!mInitDone) {
    LOG(INFO) << "init is not done yet - nothing to print";
    return;
  }

  LOG(INFO) << "**********************************************************************";
}

//______________________________________________
void CollectCalibInfoTOF::attachInputTrees()
{
  ///< attaching the input tree

  if (!mTreeTOFCalibInfo) {
    LOG(FATAL) << "Input tree with TOF calib infos is not set";
  }

  if (!mTreeTOFCalibInfo->GetBranch(mTOFCalibInfoBranchName.data())) {
    LOG(FATAL) << "Did not find TOF calib info branch " << mTOFCalibInfoBranchName << " in the input tree";
  }
  mTreeTOFCalibInfo->SetBranchAddress(mTOFCalibInfoBranchName.data(), &mTOFCalibInfo);
  LOG(INFO) << "Attached tracksTOF calib info " << mTOFCalibInfoBranchName << " branch with " << mTreeTOFCalibInfo->GetEntries()
            << " entries";

  mCurrTOFInfoTreeEntry = -1;
}

//______________________________________________
bool CollectCalibInfoTOF::loadTOFCalibInfo()
{
  ///< load next chunk of TOF infos
  printf("Loading TOF calib infos: number of entries in tree = %lld\n", mTreeTOFCalibInfo->GetEntries());

  while (++mCurrTOFInfoTreeEntry < mTreeTOFCalibInfo->GetEntries()) {
    mTreeTOFCalibInfo->GetEntry(mCurrTOFInfoTreeEntry);
    LOG(INFO) << "Loading TOF calib info entry " << mCurrTOFInfoTreeEntry << " -> " << mTOFCalibInfo->size() << " infos";

    if (!mTOFCalibInfo->size()) {
      continue;
    }
    return true;
  }
  --mCurrTOFInfoTreeEntry;

  return false;
}
//______________________________________________
void CollectCalibInfoTOF::addHit(o2::dataformats::CalibInfoTOF& calibInfo)
{

  ///< This is the method that fills the array of calibInfoTOF and also
  ///< decides whether to fill the output tree or not

  mTOFCollectedCalibInfo[calibInfo.getTOFChIndex()].emplace_back(calibInfo.getTimestamp(), calibInfo.getDeltaTimePi(), calibInfo.getTot(), calibInfo.getFlags());
  if (mTOFCollectedCalibInfo[calibInfo.getTOFChIndex()].size() == MAXNUMBEROFHITS) { // the current channel has arrived to the limit of hits that we can store between two fills --> filling the tree
    fillTree();
  }
  if (calibInfo.getTimestamp() < mMinTimestamp.GetVal() || mMinTimestamp.GetVal() == -1)
    mMinTimestamp.SetVal(calibInfo.getTimestamp());
  else if (calibInfo.getTimestamp() > mMaxTimestamp.GetVal())
    mMaxTimestamp.SetVal(calibInfo.getTimestamp());
}
//______________________________________________
void CollectCalibInfoTOF::fillTree()
{

  ///< Here we will the tree from the accumulator

  for (int ich = 0; ich < Geo::NCHANNELS; ich++) {
    mTOFCalibInfoOut = &mTOFCollectedCalibInfo[ich];
    mOutputTree->Fill();
    mTOFCollectedCalibInfo[ich].clear();
  }
}
