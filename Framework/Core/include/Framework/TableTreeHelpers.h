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
// GenericTableToTree allows to save the contents of a given arrow::Table into
// a TTree
// BranchFiller is used by GenericTableToTree
//
// To write the contents of a table ta to a tree tr on file f do:
//  . GenericTableToTree t2t(f,treename);
//  . t2t.addBranches(ta);
//    OR t2t.addBranch(column.get(), field.get()), ...;
//  . t2t.write();
//
// .............................................................................
class ColumnToBranchBase
{
 public:
  ColumnToBranchBase(arrow::ChunkedArray* column, arrow::Field* field, int size = 1);
  virtual ~ColumnToBranchBase(){};
  void at(int64_t* pos)
  {
    mCurrentPos = pos;
    resetBuffer();
  }

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

template <typename T>
struct ROOTTypeString {
  static constexpr char const* str = "/E";
};

template <>
struct ROOTTypeString<bool> {
  static constexpr char const* str = "/O";
};

template <>
struct ROOTTypeString<uint8_t> {
  static constexpr char const* str = "/b";
};

template <>
struct ROOTTypeString<uint16_t> {
  static constexpr char const* str = "/s";
};

template <>
struct ROOTTypeString<uint32_t> {
  static constexpr char const* str = "/i";
};

template <>
struct ROOTTypeString<uint64_t> {
  static constexpr char const* str = "/l";
};

template <>
struct ROOTTypeString<int8_t> {
  static constexpr char const* str = "/B";
};

template <>
struct ROOTTypeString<int16_t> {
  static constexpr char const* str = "/S";
};

template <>
struct ROOTTypeString<int32_t> {
  static constexpr char const* str = "/I";
};

template <>
struct ROOTTypeString<int64_t> {
  static constexpr char const* str = "/L";
};

template <>
struct ROOTTypeString<float> {
  static constexpr char const* str = "/F";
};

template <>
struct ROOTTypeString<double> {
  static constexpr char const* str = "/D";
};

template <typename T>
constexpr auto ROOTTypeString_t = ROOTTypeString<T>::str;

template <typename T>
class ColumnToBranch : public ColumnToBranchBase
{
 public:
  ColumnToBranch(TTree* tree, arrow::ChunkedArray* column, arrow::Field* field, int size = 1)
    : ColumnToBranchBase(column, field, size)
  {
    if constexpr (std::is_pointer_v<T>) {
      mLeaflist = mBranchName + "[" + std::to_string(listSize) + "]" + ROOTTypeString_t<std::remove_pointer_t<T>>;
    } else {
      mLeaflist = mBranchName + ROOTTypeString_t<T>;
    }
    mBranch = tree->GetBranch(mBranchName.c_str());
    if (mBranch == nullptr) {
      mBranch = tree->Branch(mBranchName.c_str(), (char*)nullptr, mLeaflist.c_str());
    }
    if constexpr (std::is_same_v<bool, std::remove_pointer_t<T>>) {
      mCurrent = new bool[listSize];
      mLast = mCurrent + listSize;
      accessChunk(0);
    } else {
      accessChunk(0);
    }
  }

 private:
  std::string mLeaflist;

  void resetBuffer() override
  {
    if constexpr (std::is_same_v<bool, std::remove_pointer_t<T>>) {
      if (O2_BUILTIN_UNLIKELY((*mCurrentPos - mFirstIndex) * listSize >= getCurrentArray()->length())) {
        nextChunk();
      }
    } else {
      if (O2_BUILTIN_UNLIKELY((mCurrent + (*mCurrentPos - mFirstIndex) * listSize) >= mLast)) {
        nextChunk();
      }
    }
    accessChunk(*mCurrentPos);
    mBranch->SetAddress((void*)(mCurrent + (*mCurrentPos - mFirstIndex) * listSize));
  }

  auto getCurrentArray() const
  {
    if (listSize > 1) {
      return std::static_pointer_cast<o2::soa::arrow_array_for_t<std::remove_pointer_t<T>>>(std::static_pointer_cast<arrow::FixedSizeListArray>(mColumn->chunk(mCurrentChunk))->values());
    } else {
      return std::static_pointer_cast<o2::soa::arrow_array_for_t<std::remove_pointer_t<T>>>(mColumn->chunk(mCurrentChunk));
    }
  }

  void nextChunk() override
  {
    mFirstIndex += getCurrentArray()->length();
    ++mCurrentChunk;
  }

  void accessChunk(int at)
  {
    auto array = getCurrentArray();
    if constexpr (std::is_same_v<bool, std::remove_pointer_t<T>>) {
      for (auto i = 0; i < listSize; ++i) {
        mCurrent[i] = (bool)array->Value((at - mFirstIndex) * listSize + i);
      }
    } else {
      mCurrent = (std::remove_pointer_t<T>*)array->raw_values();
      mLast = mCurrent + array->length() * listSize;
    }
  }

  mutable std::remove_pointer_t<T>* mCurrent = nullptr;
  mutable std::remove_pointer_t<T>* mLast = nullptr;
  TBranch* mBranch = nullptr;
};

class TableToTree
{
 public:
  TableToTree(TFile* file, const char* treename);

  TTree* write();
  void addBranch(arrow::ChunkedArray* column, arrow::Field* field);
  void addBranches(arrow::Table* table);

 private:
  int64_t mRows = 0;
  TTree* mTree = nullptr;
  std::vector<std::unique_ptr<ColumnToBranchBase>> ColumnExtractors;
};

class GenericTreeToTable
{
 public:
 private:
  std::shared_ptr<arrow::Table> mTable;
  std::vector<std::shared_ptr<arrow::Field>> mFields;
  std::string mTableLabel;
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
