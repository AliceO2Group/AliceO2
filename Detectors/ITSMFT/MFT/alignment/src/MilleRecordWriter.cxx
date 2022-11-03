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

/// @file MilleRecordWriter.cxx

#include <TFile.h>
#include <TTree.h>

#include "Framework/Logger.h"

#include "MFTAlignment/MilleRecordWriter.h"

using namespace o2::mft;

ClassImp(o2::mft::MilleRecordWriter);

//__________________________________________________________________________
MilleRecordWriter::MilleRecordWriter()
  : mDataTree(nullptr),
    mDataFile(nullptr),
    mIsSuccessfulInit(false),
    mIsConstraintsRec(false),
    mNEntriesAutoSave(10000),
    mDataFileName("mft_mille_records.root"),
    mDataTreeName("milleRecords"),
    mDataBranchName("data"),
    mRecord(nullptr),
    mCurrentDataID(-1)
{
  mRecord = new MillePedeRecord();
}

//__________________________________________________________________________
MilleRecordWriter::~MilleRecordWriter()
{
  if (mDataFile) {
    mDataFile->Close();
    LOG(info) << "MilleRecordWriter - closed file "
              << mDataFileName.Data();
  }
  if (mRecord) {
    delete mRecord;
  }
}

//__________________________________________________________________________
void MilleRecordWriter::setCyclicAutoSave(const long nEntries)
{
  if (nEntries <= 0) {
    return;
  }
  mNEntriesAutoSave = nEntries;
}

//__________________________________________________________________________
void MilleRecordWriter::changeDataBranchName(const bool isConstraintsRec)
{
  mIsConstraintsRec = isConstraintsRec;
  if (!mIsConstraintsRec) {
    mDataBranchName = TString("data");
  } else {
    mDataBranchName = TString("constraints");
  }
}

//__________________________________________________________________________
void MilleRecordWriter::init()
{
  mIsSuccessfulInit = false;

  if (mDataFile == nullptr) {
    mDataFile = new TFile(mDataFileName.Data(), "recreate", "", 505);
  }

  if ((!mDataFile) || (mDataFile->IsZombie())) {
    LOGF(fatal,
         "MilleRecordWriter::init() - failed to initialise records file %s!",
         mDataFileName.Data());
    return;
  }
  if (mDataTree == nullptr) {
    mDataFile->cd();
    mDataTree = new TTree(mDataTreeName.Data(), "records for MillePede2");
    mDataTree->SetAutoSave(mNEntriesAutoSave); // flush the TTree to disk every N entries
    const int bufsize = 32000;
    const int splitLevel = 99; // "all the way"
    mDataTree->Branch(mDataBranchName.Data(), "MillePedeRecord", mRecord, bufsize, splitLevel);
  }
  if (!mDataTree) {
    LOG(fatal) << "MilleRecordWriter::init() - failed to initialise TTree !";
    return;
  }

  if (!mIsConstraintsRec) {
    LOGF(info,
         "MilleRecordWriter::init() - file %s used for derivatives records",
         mDataFileName.Data());
  } else {
    LOGF(info,
         "MilleRecordWriter::init() - file %s used for constraints records",
         mDataFileName.Data());
  }
  mIsSuccessfulInit = true;
}

//__________________________________________________________________________
void MilleRecordWriter::fillRecordTree(const bool doPrint)
{
  if (!isInitOk()) {
    LOG(warning) << "MilleRecordWriter::fillRecordTree() - aborted, init was not ok !";
    return;
  }
  mDataTree->Fill();
  mCurrentDataID++;
  if (doPrint) {
    LOGF(info, "MilleRecordWriter::fillRecordTree() - added entry %i", mCurrentDataID);
    mRecord->Print();
  }
  mRecord->Reset();
}

//__________________________________________________________________________
void MilleRecordWriter::terminate()
{
  if (mDataFile && mDataFile->IsWritable() && mDataTree) {
    mDataFile->cd();
    mDataTree->Write();
    LOG(info) << "MilleRecordWriter::terminate() - wrote tree "
              << mDataTreeName.Data();
  }
}

//_____________________________________________________________________________
void MilleRecordWriter::setRecordWeight(double wgh)
{
  mRecord->SetWeight(wgh);
}

//_____________________________________________________________________________
void MilleRecordWriter::setRecordRun(int run)
{
  mRecord->SetRunID(run);
}
