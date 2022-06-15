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

/// \file MilleRecordReader.h
/// \author arakotoz@cern.ch
/// \brief Class dedicated to read MillePedeRecords from ROOT files

#ifndef ALICEO2_MFT_MILLERECORD_READER_H
#define ALICEO2_MFT_MILLERECORD_READER_H

#include <Rtypes.h>
#include <TString.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

#include "MFTAlignment/MillePedeRecord.h"

namespace o2
{
namespace mft
{

class MilleRecordReader
{
 public:
  /// \brief constructor
  MilleRecordReader();

  /// \brief destructor
  virtual ~MilleRecordReader();

  /// \brief choose data records filename
  void changeDataBranchName(const bool isConstraintsRec = true);

  /// \brief connect to input TChain
  void connectToChain(TChain* ch);

  /// \brief check if connect to input TChain went well
  bool isReaderOk() const { return mIsSuccessfulInit; }

  /// \brief check if the last operation readNextEntry() was ok
  bool isReadEntryOk() const { return mIsReadEntryOk; }

  /// \brief return the record
  o2::mft::MillePedeRecord* getRecord() { return mRecord; };

  /// \brief return the ID of the current record in the TTree
  long getCurrentDataID() const { return mCurrentDataID; }

  /// \brief read the next entry in the tree
  void readNextEntry(const bool doPrint = false);

  /// \brief read the entry # id in the tree
  void readEntry(const Long_t id, const bool doPrint = false);

  /// \brief return the number of entries
  Long64_t getNEntries() const { return mNEntries; }

 protected:
  TChain* mDataTree;                 ///< TChain container that stores the records
  bool mIsSuccessfulInit;            ///< boolean to monitor the success of the initialization
  bool mIsConstraintsRec;            ///< boolean to know if these are data records or constraints records
  bool mIsReadEntryOk;               ///< boolean to know if the last operation readNextEntry() was ok
  TString mDataTreeName;             ///< name of the record TTree/TChain
  TString mDataBranchName;           ///< name of the branch where records will be stored
  o2::mft::MillePedeRecord* mRecord; ///< the running record
  Long64_t mCurrentDataID;           ///< counter indicating the ID of the current record in the tree
  Long64_t mNEntries;                ///< number of entries in the read TChain

  ClassDef(MilleRecordReader, 0);
};
} // namespace mft
} // namespace o2

#endif
