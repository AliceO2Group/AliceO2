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
#include <arrow/chunked_array.h>
#include <arrow/builder.h>
#include <arrow/memory_pool.h>
#include <string>
#include <memory>

namespace o2::soa
{
enum struct IndexKind : int {
  IdxInvalid = -1,
  IdxSelf = 0,
  IdxSingle = 1,
  IdxSlice = 2,
  IdxArray = 3
};
} // namespace o2::soa

namespace o2::framework
{
void cannotBuildAnArray();
void cannotCreateIndexBuilder();

struct ChunkedArrayIterator {
  ChunkedArrayIterator(std::shared_ptr<arrow::ChunkedArray> source);

  std::shared_ptr<arrow::ChunkedArray> mSource = nullptr;
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

struct SelfBuilder {
  std::unique_ptr<arrow::ArrayBuilder> mBuilder = nullptr;
  SelfBuilder(arrow::MemoryPool* pool);
};

struct SingleBuilder {
  ChunkedArrayIterator arrayIterator;
  std::unique_ptr<arrow::ArrayBuilder> mBuilder = nullptr;
  SingleBuilder(std::shared_ptr<arrow::ChunkedArray> source, arrow::MemoryPool* pool);
};

struct SliceBuilder {
  ChunkedArrayIterator arrayIterator;
  arrow::ArrayBuilder* mValueBuilder = nullptr;
  std::unique_ptr<arrow::ArrayBuilder> mListBuilder = nullptr;
  std::shared_ptr<arrow::NumericArray<arrow::Int32Type>> mValues = nullptr;
  std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> mCounts = nullptr;
  SliceBuilder(std::shared_ptr<arrow::ChunkedArray> source, arrow::MemoryPool* pool);

  arrow::Status preSlice();
};

struct ArrayBuilder {
  ChunkedArrayIterator arrayIterator;
  arrow::ArrayBuilder* mValueBuilder = nullptr;
  std::vector<int> mValues;
  std::vector<std::vector<int>> mIndices;
  std::unique_ptr<arrow::ArrayBuilder> mListBuilder = nullptr;
  ArrayBuilder(std::shared_ptr<arrow::ChunkedArray> source, arrow::MemoryPool* pool);

  arrow::Status preFind();
};

struct IndexColumnBuilderNG {
  std::variant<std::monostate, SelfBuilder, SingleBuilder, SliceBuilder, ArrayBuilder> builder;

  IndexColumnBuilderNG(soa::IndexKind kind, arrow::MemoryPool* pool, std::shared_ptr<arrow::ChunkedArray> source = nullptr)
  {
    switch (kind) {
      case soa::IndexKind::IdxSelf:
        builder = SelfBuilder{pool};
        break;
      case soa::IndexKind::IdxSingle:
        builder = SingleBuilder{source, pool};
        break;
      case soa::IndexKind::IdxSlice:
        builder = SliceBuilder{source, pool};
        break;
      case soa::IndexKind::IdxArray:
        builder = ArrayBuilder{source, pool};
        break;
      default:
        cannotCreateIndexBuilder();
    }
  }
};

struct SelfIndexColumnBuilder {
  SelfIndexColumnBuilder(const char* name, arrow::MemoryPool* pool);
  virtual ~SelfIndexColumnBuilder() = default;

  inline std::shared_ptr<arrow::ChunkedArray> result() const
  {
    std::shared_ptr<arrow::Array> array;
    auto status = static_cast<arrow::Int32Builder*>(mBuilder.get())->Finish(&array);
    if (!status.ok()) {
      cannotBuildAnArray();
    }

    return std::make_shared<arrow::ChunkedArray>(array);
  }

  inline bool find(int)
  {
    return true;
  }

  inline void fill(int idx)
  {
    (void)static_cast<arrow::Int32Builder*>(mBuilder.get())->Append(idx);
  }

  std::string mColumnName;
  std::unique_ptr<arrow::ArrayBuilder> mBuilder = nullptr;
};

class IndexColumnBuilder : public SelfIndexColumnBuilder, public ChunkedArrayIterator
{
 public:
  IndexColumnBuilder(std::shared_ptr<arrow::ChunkedArray> source, const char* name, int listSize, arrow::MemoryPool* pool);
  ~IndexColumnBuilder() override = default;

  inline std::shared_ptr<arrow::ChunkedArray> result() const
  {
    if (mListSize == -1) {
      return resultMulti();
    } else if (mListSize == 2) {
      return resultSlice();
    } else {
      return resultSingle();
    }
  }

  inline bool find(int idx)
  {
    if (mListSize == -1) {
      return findMulti(idx);
    } else if (mListSize == 2) {
      return findSlice(idx);
    } else {
      return findSingle(idx);
    }
  }

  inline void fill(int idx)
  {
    ++mResultSize;
    if (mListSize == -1) {
      fillMulti(idx);
    } else if (mListSize == 2) {
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
} // namespace o2::framework

#endif // O2_FRAMEWORK_INDEXBUILDERHELPERS_H_
