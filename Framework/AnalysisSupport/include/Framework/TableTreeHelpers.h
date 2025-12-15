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

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/record_batch.h>
#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "Framework/TableBuilder.h"
#include <arrow/dataset/file_base.h>
#include <memory>

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
struct ROOTTypeInfo {
  EDataType type;
  char suffix[3];
  int size;
};

auto arrowTypeFromROOT(EDataType type, int size);
auto basicROOTTypeFromArrow(arrow::Type::type id);

class ColumnToBranch
{
 public:
  ColumnToBranch(TTree* tree, std::shared_ptr<arrow::ChunkedArray> const& column, std::shared_ptr<arrow::Field> const& field);
  ColumnToBranch(ColumnToBranch const& other) = delete;
  ColumnToBranch(ColumnToBranch&& other) = delete;
  void at(const int64_t* pos);
  [[nodiscard]] int fieldSize() const { return mFieldSize; }
  [[nodiscard]] int columnEntries() const { return mColumn->length(); }
  [[nodiscard]] char const* branchName() const { return mBranchName.c_str(); }

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
  int64_t mChunkLength = 0;
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

// -----------------------------------------------------------------------------
} // namespace o2::framework

// =============================================================================
#endif // O2_FRAMEWORK_TABLETREEHELPERS_H_
