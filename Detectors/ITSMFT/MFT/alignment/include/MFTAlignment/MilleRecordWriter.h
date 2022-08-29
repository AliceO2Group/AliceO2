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

/// \file MilleRecordWriter.h
/// \author arakotoz@cern.ch
/// \brief Class dedicated to write MillePedeRecords to output file for MFT

#ifndef ALICEO2_MFT_MILLERECORD_WRITER_H
#define ALICEO2_MFT_MILLERECORD_WRITER_H

#include <Rtypes.h>
#include <TString.h>

#include "MFTAlignment/MillePedeRecord.h"

class TFile;
class TTree;
namespace o2
{
namespace mft
{

class MilleRecordWriter
{
 public:
  /// \brief constructor
  MilleRecordWriter();

  /// \brief destructor
  virtual ~MilleRecordWriter();

  /// \brief Set the number of entries to be used by TTree::AutoSave()
  void setCyclicAutoSave(const long nEntries);

  /// \brief choose data records filename
  void setDataFileName(TString fname) { mDataFileName = fname; }

  /// \brief choose data records filename
  void changeDataBranchName(const bool isConstraintsRec = true);

  /// \brief init output file and tree
  void init();

  /// \brief check if init went well
  bool isInitOk() const { return mIsSuccessfulInit; }

  /// \brief return the record
  o2::mft::MillePedeRecord* getRecord() { return mRecord; };

  /// \brief return the ID of the current record in the TTree
  Long64_t getCurrentDataID() const { return mCurrentDataID; }

  /// \brief fill tree
  void fillRecordTree(const bool doPrint = false);

  /// \brief write tree and close output file
  void terminate();

  /// \brief assign run
  void setRecordRun(int run);

  /// \brief assign weight
  void setRecordWeight(double wgh);

 protected:
  TTree* mDataTree;                  ///< TTree container that stores the records
  TFile* mDataFile;                  ///< output file where the records are written
  bool mIsSuccessfulInit;            ///< boolean to monitor the success of the initialization
  bool mIsConstraintsRec;            ///< boolean to know if these are data records or constraints records
  long mNEntriesAutoSave;            ///< max entries in the buffer after which TTree::AutoSave() is automatically used
  TString mDataFileName;             ///< name of the output file that will store the record TTree
  TString mDataTreeName;             ///< name of the record TTree
  TString mDataBranchName;           ///< name of the branch where records will be stored
  o2::mft::MillePedeRecord* mRecord; ///< the running record
  Long64_t mCurrentDataID;           ///< counter increasing when adding a record to the tree

  ClassDef(MilleRecordWriter, 0);
};
} // namespace mft
} // namespace o2

#endif
