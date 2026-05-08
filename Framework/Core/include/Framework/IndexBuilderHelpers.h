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
void cannotBuildAnArray(const char* reason);
void cannotCreateIndexBuilder();

struct ChunkedArrayIterator {
  ChunkedArrayIterator(std::shared_ptr<arrow::ChunkedArray> source);
  void reset(std::shared_ptr<arrow::ChunkedArray>& source);

  std::shared_ptr<arrow::ChunkedArray> mSource = nullptr;
  size_t mPosition = 0;
  int mChunk = 0;
  size_t mOffset = 0;
  std::shared_ptr<arrow::Int32Array> mCurrentArray = nullptr;
  int const* mCurrent = nullptr;
  int const* mLast = nullptr;
  size_t mFirstIndex = 0;
  size_t mSourceSize = 0;

  std::shared_ptr<arrow::Int32Array> getCurrentArray();
  void nextChunk();
  void prevChunk();
  int valueAt(size_t pos);
};

struct SelfBuilder {
  std::unique_ptr<arrow::ArrayBuilder> mBuilder = nullptr;
  std::unique_ptr<framework::ChunkedArrayIterator> keyIndex = nullptr;
  SelfBuilder(arrow::MemoryPool* pool);
  void reset(std::shared_ptr<arrow::ChunkedArray>);

  inline bool find(int) const
  {
    return true;
  }
  void fill(int idx);
  std::shared_ptr<arrow::ChunkedArray> result() const;
};

struct SingleBuilder : public ChunkedArrayIterator {
  std::unique_ptr<arrow::ArrayBuilder> mBuilder = nullptr;
  SingleBuilder(std::shared_ptr<arrow::ChunkedArray> source, arrow::MemoryPool* pool);
  void reset(std::shared_ptr<arrow::ChunkedArray> source);

  bool find(int idx);
  void fill(int idx);
  std::shared_ptr<arrow::ChunkedArray> result() const;
};

struct SliceBuilder : public ChunkedArrayIterator {
  arrow::ArrayBuilder* mValueBuilder = nullptr;
  std::unique_ptr<arrow::ArrayBuilder> mListBuilder = nullptr;
  std::shared_ptr<arrow::NumericArray<arrow::Int32Type>> mValues = nullptr;
  std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> mCounts = nullptr;
  int mValuePos = 0;
  SliceBuilder(std::shared_ptr<arrow::ChunkedArray> source, arrow::MemoryPool* pool);
  void reset(std::shared_ptr<arrow::ChunkedArray> source);

  bool find(int idx);
  void fill(int idx);
  std::shared_ptr<arrow::ChunkedArray> result() const;

  arrow::Status preSlice();
};

struct ArrayBuilder : public ChunkedArrayIterator {
  arrow::ArrayBuilder* mValueBuilder = nullptr;
  std::vector<int> mValues;
  std::vector<std::vector<int>> mIndices;
  std::unique_ptr<arrow::ArrayBuilder> mListBuilder = nullptr;
  ArrayBuilder(std::shared_ptr<arrow::ChunkedArray> source, arrow::MemoryPool* pool);
  void reset(std::shared_ptr<arrow::ChunkedArray> source);

  bool find(int idx);
  void fill(int idx);
  std::shared_ptr<arrow::ChunkedArray> result() const;

  arrow::Status preFind();
};

struct IndexColumnBuilder {
  std::variant<std::monostate, SelfBuilder, SingleBuilder, SliceBuilder, ArrayBuilder> builder;
  size_t mResultSize = 0;
  int mColumnPos = -1;
  IndexColumnBuilder(soa::IndexKind kind, int pos, arrow::MemoryPool* pool, std::shared_ptr<arrow::ChunkedArray> source = nullptr);
  void reset(std::shared_ptr<arrow::ChunkedArray> source = nullptr);

  bool find(int idx);
  void fill(int idx);
  std::shared_ptr<arrow::ChunkedArray> result() const;
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_INDEXBUILDERHELPERS_H_
