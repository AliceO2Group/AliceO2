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
#ifndef O2_FRAMEWORK_TABLETREEHELPERS_H_
#define O2_FRAMEWORK_TABLETREEHELPERS_H_

#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TableBuilder.h"

// =============================================================================
namespace o2::framework
{

// -----------------------------------------------------------------------------
// TableToTree allows to save the contents of a given arrow::Table into
// a TTree
// ColumnToBranch is used by GenericTableToTree
//
// To write the contents of a table ta to a tree tr on file f do:
//  . GenericTableToTree t2t(ta, f,treename);
//  . t2t.addBranches();
//    OR t2t.addBranch(column.get(), field.get()), ...;
//  . t2t.process();
//
// .............................................................................
class ColumnToBranchBase
{
 public:
  ColumnToBranchBase(arrow::ChunkedArray* column, arrow::Field* field, int size = 1);
  virtual ~ColumnToBranchBase() = default;
  void at(const int64_t* pos);

 protected:
  std::string mBranchName;

  virtual void resetBuffer() = 0;
  virtual void nextChunk() = 0;

  arrow::ChunkedArray* mColumn = nullptr;

  int64_t const* mCurrentPos = nullptr;
  mutable int mFirstIndex = 0;
  mutable int mCurrentChunk = 0;

  int listSize = 1;
};

class TableToTree
{
 public:
  TableToTree(std::shared_ptr<arrow::Table> table, TFile* file, const char* treename);

  TTree* process();
  void addBranch(std::shared_ptr<arrow::ChunkedArray> column, std::shared_ptr<arrow::Field> field);
  void addAllBranches();

 private:
  arrow::Table* mTable;
  int64_t mRows = 0;
  TTree* mTree = nullptr;
  std::vector<std::unique_ptr<ColumnToBranchBase>> mColumnReaders;
};

class TreeToTable
{

 private:
  std::shared_ptr<arrow::Table> mTable;
  std::vector<std::string> mColumnNames;
  std::string mTableLabel;

 public:
  // set table label to be added into schema metadata
  void setLabel(const char* label);

  // add a column to be included in the arrow::table
  void addColumn(const char* colname);

  // add all branches in @a tree as columns
  bool addAllColumns(TTree* tree);

  // do the looping with the TTreeReader
  void fill(TTree* tree);

  // create the table
  std::shared_ptr<arrow::Table> finalize();
};

// -----------------------------------------------------------------------------
} // namespace o2::framework

// =============================================================================
#endif // O2_FRAMEWORK_TABLETREEHELPERS_H_
