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
// -----------------------------------------------------------------------------
// TreeToTable allows to fill the contents of a given TTree to an arrow::Table
//  ColumnIterator is used by TreeToTable
//
// To copy the contents of a tree tr to a table ta do:
//  . TreeToTable t2t(tr);
//  . t2t.addColumn(columnname1); t2t.addColumn(columnname2); ...
//    OR
//    t2t.addAllColumns();
//  . auto ta = t2t.process();
//
// .............................................................................
struct ROOTTypeInfo {
  EDataType type;
  char suffix[3];
  int size;
};

auto arrowTypeFromROOT(EDataType type, int size);
auto basicROOTTypeFromArrow(arrow::Type::type id);

class BranchToColumn
{
 public:
  BranchToColumn(TBranch* branch, bool VLA, std::string name, EDataType type, int listSize, arrow::MemoryPool* pool);
  //  BranchToColumn(TBranch* branch, TBranch* sizeBranch, std::string name, EDataType type, arrow::MemoryPool* pool);
  ~BranchToColumn() = default;
  TBranch* branch();

  std::pair<std::shared_ptr<arrow::ChunkedArray>, std::shared_ptr<arrow::Field>> read(TBuffer* buffer);

 private:
  TBranch* mBranch = nullptr;
  bool mVLA = false;
  std::string mColumnName;
  EDataType mType;
  std::shared_ptr<arrow::DataType> mArrowType;
  arrow::ArrayBuilder* mValueBuilder = nullptr;
  std::unique_ptr<arrow::ArrayBuilder> mListBuilder = nullptr;
  int mListSize = 1;
  std::unique_ptr<arrow::ArrayBuilder> mBuilder = nullptr;
  arrow::MemoryPool* mPool = nullptr;
};

class ColumnToBranch
{
 public:
  ColumnToBranch(TTree* tree, std::shared_ptr<arrow::ChunkedArray> const& column, std::shared_ptr<arrow::Field> const& field);
  ColumnToBranch(ColumnToBranch const& other) = delete;
  ColumnToBranch(ColumnToBranch&& other) = delete;
  void at(const int64_t* pos);
  int fieldSize() const { return mFieldSize; }
  char const* branchName() const { return mBranchName.c_str(); }

 private:
  void accessChunk();
  void nextChunk();

  std::string mBranchName;
  TBranch* mBranch = nullptr;
  TBranch* mSizeBranch = nullptr;
  arrow::ChunkedArray* mColumn = nullptr;
  int64_t mFirstIndex = 0;
  int mCurrentChunk = 0;
  int mListSize = 1;
  ROOTTypeInfo mElementType;
  arrow::Type::type mFieldType;
  std::vector<uint8_t> cache;
  std::shared_ptr<arrow::Array> mCurrentArray = nullptr;
  int64_t mChunkLength;
  int mFieldSize = 0;
};

class TableToTree
{
 public:
  TableToTree(std::shared_ptr<arrow::Table> const& table, TFile* file, const char* treename);

  std::shared_ptr<TTree> process();
  void addBranch(std::shared_ptr<arrow::ChunkedArray> const& column, std::shared_ptr<arrow::Field> const& field);
  void addAllBranches();

 private:
  arrow::Table* mTable;
  int64_t mRows = 0;
  std::shared_ptr<TTree> mTree;
  std::vector<std::unique_ptr<ColumnToBranch>> mColumnReaders;
};

class TreeToTable
{
 public:
  TreeToTable(arrow::MemoryPool* pool = arrow::default_memory_pool());
  void setLabel(const char* label);
  void addAllColumns(TTree* tree, std::vector<std::string>&& names = {});
  void fill(TTree*);
  std::shared_ptr<arrow::Table> finalize();

 private:
  arrow::MemoryPool* mArrowMemoryPool;
  std::vector<std::unique_ptr<BranchToColumn>> mBranchReaders;
  std::string mTableLabel;
  std::shared_ptr<arrow::Table> mTable;

  void addReader(TBranch* branch, std::string const& name, bool VLA);
};

// -----------------------------------------------------------------------------
} // namespace o2::framework

// =============================================================================
#endif // O2_FRAMEWORK_TABLETREEHELPERS_H_
