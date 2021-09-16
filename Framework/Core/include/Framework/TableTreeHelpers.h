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
//  . TableToTree t2t(f,treename);
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

namespace
{
std::shared_ptr<arrow::DataType> arrowTypeFromROOT(EDataType type, int size)
{
  auto typeGenerator = [](std::shared_ptr<arrow::DataType>&& type, int size) -> std::shared_ptr<arrow::DataType> {
    if (size == 1) {
      return std::move(type);
    }
    return arrow::fixed_size_list(type, size);
  };

  switch (type) {
    case EDataType::kBool_t:
      return typeGenerator(arrow::boolean(), size);
    case EDataType::kUChar_t:
      return typeGenerator(arrow::uint8(), size);
    case EDataType::kUShort_t:
      return typeGenerator(arrow::uint16(), size);
    case EDataType::kUInt_t:
      return typeGenerator(arrow::uint32(), size);
    case EDataType::kULong64_t:
      return typeGenerator(arrow::uint64(), size);
    case EDataType::kChar_t:
      return typeGenerator(arrow::int8(), size);
    case EDataType::kShort_t:
      return typeGenerator(arrow::int16(), size);
    case EDataType::kInt_t:
      return typeGenerator(arrow::int32(), size);
    case EDataType::kLong64_t:
      return typeGenerator(arrow::int64(), size);
    case EDataType::kFloat_t:
      return typeGenerator(arrow::float32(), size);
    case EDataType::kDouble_t:
      return typeGenerator(arrow::float64(), size);
    default:
      throw runtime_error("Unsupported branch type");
  }
}
} // namespace

class BranchToColumnBase
{
 public:
  BranchToColumnBase(TBranch* branch, const char* name, EDataType type, int listSize);
  virtual ~BranchToColumnBase(){};

  TBranch* branch()
  {
    return mBranch;
  }

  virtual std::pair<std::shared_ptr<arrow::ChunkedArray>, std::shared_ptr<arrow::Field>> read(TBuffer* buffer) = 0;

 protected:
  TBranch* mBranch = nullptr;
  std::string mColumnName;
  EDataType mType;
  arrow::ArrayBuilder* mValueBuilder = nullptr;
  std::unique_ptr<arrow::FixedSizeListBuilder> mListBuilder = nullptr;
  int mListSize = 1;
  std::unique_ptr<arrow::ArrayBuilder> builder = nullptr;
};

template <typename T>
class BranchToColumn : public BranchToColumnBase
{
  using ArrowType = typename detail::ConversionTraits<T>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

 public:
  BranchToColumn(TBranch* branch, const char* name, EDataType type, int listSize, arrow::MemoryPool* pool = arrow::default_memory_pool())
    : BranchToColumnBase(branch, name, type, listSize)
  {
    if (mListSize > 1) {
      auto status = arrow::MakeBuilder(pool, arrow::TypeTraits<ArrowType>::type_singleton(), &builder);
      if (!status.ok()) {
        throw runtime_error("Cannot create value builder");
      }
      mListBuilder = std::make_unique<arrow::FixedSizeListBuilder>(pool, std::move(builder), mListSize);
      mValueBuilder = static_cast<BuilderType*>(mListBuilder->value_builder());
    } else {
      auto status = arrow::MakeBuilder(pool, arrow::TypeTraits<ArrowType>::type_singleton(), &builder);
      if (!status.ok()) {
        throw runtime_error("Cannot create builder");
      }
      mValueBuilder = builder.get();
    }
  }

  std::pair<std::shared_ptr<arrow::ChunkedArray>, std::shared_ptr<arrow::Field>> read(TBuffer* buffer) override
  {
    auto arrowType = arrowTypeFromROOT(mType, mListSize);

    auto totalEntries = static_cast<int>(mBranch->GetEntries());
    auto status = Reserve(totalEntries);
    if (!status.ok()) {
      throw runtime_error("Failed to reserve memory for array builder");
    }
    int readEntries = 0;
    buffer->Reset();
    while (readEntries < totalEntries) {
      auto read = mBranch->GetBulkRead().GetBulkEntries(readEntries, *buffer);
      readEntries += read;
      status &= AppendValues(reinterpret_cast<unsigned char const*>(buffer->GetCurrent()), read);
    }
    if (!status.ok()) {
      throw runtime_error("Failed to append values to array");
    }
    std::shared_ptr<arrow::Array> array;
    status &= Finish(&array);
    if (!status.ok()) {
      throw runtime_error("Failed to create boolean array");
    }
    auto chunk = std::make_shared<arrow::ChunkedArray>(array);
    auto field = std::make_shared<arrow::Field>(mBranch->GetName(), arrowType);
    return std::tie(chunk, field);
  }

 private:
  arrow::Status AppendValues(unsigned char const* buffer, int numEntries)
  {
    using B = typename std::conditional<std::is_same_v<T, bool>, uint8_t, T>::type;
    if (mListSize > 1) {
      auto status = static_cast<BuilderType*>(mValueBuilder)->AppendValues(reinterpret_cast<B const*>(buffer), numEntries * mListSize);
      status &= mListBuilder->AppendValues(numEntries);
      return status;
    } else {
      auto status = static_cast<BuilderType*>(mValueBuilder)->AppendValues(reinterpret_cast<B const*>(buffer), numEntries);
      return status;
    }
  }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* array)
  {
    if (mListSize > 1) {
      return mListBuilder->Finish(array);
    } else {
      return mValueBuilder->Finish(array);
    }
  }

  arrow::Status Reserve(int numEntries)
  {
    if (mListSize > 1) {
      auto status = mListBuilder->Reserve(numEntries);
      status &= mValueBuilder->Reserve(numEntries * mListSize);
      return status;
    } else {
      return mValueBuilder->Reserve(numEntries);
    }
  }
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
  std::vector<std::unique_ptr<ColumnToBranchBase>> mColumnReaders;
};

class TreeToTable
{
 public:
  TreeToTable(arrow::MemoryPool* pool = arrow::default_memory_pool())
    : mArrowMemoryPool{pool}
  {
  }
  void setLabel(const char* label);
  void addColumns(TTree* tree, std::vector<const char*>&& names = {});
  void read();
  auto finalize()
  {
    return mTable;
  }

 private:
  arrow::MemoryPool* mArrowMemoryPool;
  std::vector<std::unique_ptr<BranchToColumnBase>> mBranchReaders;
  std::string mTableLabel;
  std::shared_ptr<arrow::Table> mTable;

  void AddReader(TBranch* branch, const char* name);
};

// -----------------------------------------------------------------------------
} // namespace o2::framework

// =============================================================================
#endif // O2_FRAMEWORK_TABLETREEHELPERS_H_
