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

#ifndef O2_FRAMEWORK_INDEXBUILDERHELPERS_H_
#define O2_FRAMEWORK_INDEXBUILDERHELPERS_H_
#include "Framework/RuntimeError.h"
#include "arrow/array.h"
#include <arrow/chunked_array.h>
#include <arrow/builder.h>
#include <arrow/memory_pool.h>
#include <string>
#include <memory>
#include <type_traits>

namespace o2::framework
{
struct ChunkedArrayIterator {
  ChunkedArrayIterator(std::shared_ptr<arrow::ChunkedArray> source);
  virtual ~ChunkedArrayIterator() = default;

  std::shared_ptr<arrow::ChunkedArray> mSource;
  size_t mPosition = 0;
  int mChunk = 0;
  size_t mOffset = 0;
  std::shared_ptr<arrow::Int32Array> mCurrentArray = nullptr;
  int const* mCurrent = nullptr;
  int const* mLast = nullptr;
  size_t mFirstIndex = 0;

  std::shared_ptr<arrow::Int32Array> getCurrentArray();
  void nextChunk();
  void prevChunk();
  int valueAt(size_t pos);
};

struct SelfIndexColumnBuilder {
  SelfIndexColumnBuilder(const char* name, arrow::MemoryPool* pool);
  virtual ~SelfIndexColumnBuilder() = default;

  template <typename C>
  inline std::shared_ptr<arrow::ChunkedArray> result() const
  {
    std::shared_ptr<arrow::Array> array;
    auto status = static_cast<arrow::Int32Builder*>(mBuilder.get())->Finish(&array);
    if (!status.ok()) {
      throw runtime_error("Cannot build an array");
    }

    return std::make_shared<arrow::ChunkedArray>(array);
  }
  std::shared_ptr<arrow::Field> field() const;
  template <typename C>
  inline bool find(int)
  {
    return true;
  }

  template <typename C>
  inline void fill(int idx)
  {
    (void)static_cast<arrow::Int32Builder*>(mBuilder.get())->Append(idx);
  }

  std::string mColumnName;
  std::shared_ptr<arrow::DataType> mArrowType;
  std::unique_ptr<arrow::ArrayBuilder> mBuilder = nullptr;
};

class IndexColumnBuilder : public SelfIndexColumnBuilder, public ChunkedArrayIterator
{
 public:
  IndexColumnBuilder(std::shared_ptr<arrow::ChunkedArray> source, const char* name, int listSize, arrow::MemoryPool* pool);
  ~IndexColumnBuilder() override = default;

  template <typename C>
  inline std::shared_ptr<arrow::ChunkedArray> result() const
  {
    if constexpr (std::is_same_v<typename C::type, std::vector<int>>) {
      return resultMulti();
    } else if constexpr (std::is_same_v<typename C::type, int[2]>) {
      return resultSlice();
    } else {
      return resultSingle();
    }
  }

  template <typename C>
  inline bool find(int idx)
  {
    if constexpr (std::is_same_v<typename C::type, std::vector<int>>) {
      return findMulti(idx);
    } else if constexpr (std::is_same_v<typename C::type, int[2]>) {
      return findSlice(idx);
    } else {
      return findSingle(idx);
    }
  }

  template <typename C>
  inline void fill(int idx)
  {
    ++mResultSize;
    if constexpr (std::is_same_v<typename C::type, std::vector<int>>) {
      fillMulti(idx);
    } else if constexpr (std::is_same_v<typename C::type, int[2]>) {
      fillSlice(idx);
    } else {
      fillSingle(idx);
    }
  }

 private:
  arrow::Status preSlice();
  arrow::Status preFind();

  bool findSingle(int idx);
  bool findSlice(int idx);
  bool findMulti(int idx);

  void fillSingle(int idx);
  void fillSlice(int idx);
  void fillMulti(int idx);

  std::shared_ptr<arrow::ChunkedArray> resultSingle() const;
  std::shared_ptr<arrow::ChunkedArray> resultSlice() const;
  std::shared_ptr<arrow::ChunkedArray> resultMulti() const;

  int mListSize = 1;
  arrow::ArrayBuilder* mValueBuilder = nullptr;
  std::unique_ptr<arrow::ArrayBuilder> mListBuilder = nullptr;

  size_t mSourceSize = 0;
  size_t mResultSize = 0;

  std::shared_ptr<arrow::NumericArray<arrow::Int32Type>> mValuesArrow = nullptr;
  std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> mCounts = nullptr;
  std::vector<int> mValues;
  std::vector<std::vector<int>> mIndices;
  int mFillOffset = 0;
  int mValuePos = 0;
};

std::shared_ptr<arrow::Table> makeArrowTable(const char* label, std::vector<std::shared_ptr<arrow::ChunkedArray>>&& columns, std::vector<std::shared_ptr<arrow::Field>>&& fields);
} // namespace o2::framework

#endif // O2_FRAMEWORK_INDEXBUILDERHELPERS_H_
