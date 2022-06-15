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

/// @file MilleRecordReader.cxx

#include "Framework/Logger.h"

#include "MFTAlignment/MilleRecordReader.h"

using namespace o2::mft;

ClassImp(o2::mft::MilleRecordReader);

//__________________________________________________________________________
MilleRecordReader::MilleRecordReader()
  : mDataTree(nullptr),
    mIsSuccessfulInit(false),
    mIsConstraintsRec(false),
    mIsReadEntryOk(false),
    mDataTreeName("milleRecords"),
    mDataBranchName("data"),
    mRecord(nullptr),
    mCurrentDataID(-1),
    mNEntries(0)
{
  mRecord = new MillePedeRecord();
}

//__________________________________________________________________________
MilleRecordReader::~MilleRecordReader()
{
  if (mDataTree)
    mDataTree->Reset();
  if (mRecord)
    delete mRecord;
}

//__________________________________________________________________________
void MilleRecordReader::changeDataBranchName(const bool isConstraintsRec)
{
  mIsConstraintsRec = isConstraintsRec;
  if (!mIsConstraintsRec) {
    mDataBranchName = TString("data");
  } else {
    mDataBranchName = TString("constraints");
  }
}

//__________________________________________________________________________
void MilleRecordReader::connectToChain(TChain* ch)
{
  mIsSuccessfulInit = false;

  if (mDataTree) {
    LOG(warning) << "MilleRecordReader::connectToChain() - input chain already initialized";
    mIsSuccessfulInit = true;
    return;
  }
  if (!ch) {
    LOG(fatal) << "MilleRecordReader::connectToChain() - input chain is a null pointer";
    return;
  }
  Long64_t nEntries = ch->GetEntries();
  if (nEntries < 1) {
    LOG(fatal) << "MilleRecordReader::connectToChain() - input chain is empty";
    return;
  }
  mDataTree = ch;
  mNEntries = mDataTree->GetEntries();
  mDataTree->SetBranchAddress(mDataBranchName.Data(), &mRecord);
  if (!mIsConstraintsRec) {
    LOGF(info,
         "MilleRecordReader::connectToChain() - found %lld derivatives records",
         mNEntries);
  } else {
    LOGF(info,
         "MilleRecordReader::connectToChain() - found %lld constraints records",
         mNEntries);
  }
  mIsSuccessfulInit = true;
  mCurrentDataID = -1;
}

//__________________________________________________________________________
void MilleRecordReader::readNextEntry(const bool doPrint)
{
  mIsReadEntryOk = false;
  if (!isReaderOk()) {
    LOG(error) << "MilleRecordReader::readNextEntry() - aborted, connectToChain() was not ok !";
    return;
  }
  mCurrentDataID++;
  if (!mDataTree || mCurrentDataID >= mNEntries) {
    mCurrentDataID--;
    return;
  }
  mDataTree->GetEntry(mCurrentDataID);
  if (doPrint) {
    LOGF(info, "MilleRecordReader::readNextEntry() - read entry %i", mCurrentDataID);
    mRecord->Print();
  }
  mIsReadEntryOk = true;
}

//__________________________________________________________________________
void MilleRecordReader::readEntry(const Long_t id, const bool doPrint)
{
  mIsReadEntryOk = false;
  if (!isReaderOk()) {
    LOG(error) << "MilleRecordReader::readEntry() - aborted, connectToChain() was not ok !";
    return;
  }
  mCurrentDataID = id;
  if (!mDataTree || mCurrentDataID >= mNEntries) {
    return;
  }
  mDataTree->GetEntry(mCurrentDataID);
  if (doPrint) {
    LOGF(info, "MilleRecordReader::readEntry() - read entry %i", mCurrentDataID);
    mRecord->Print();
  }
  mIsReadEntryOk = true;
}
