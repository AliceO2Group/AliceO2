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

#include "Framework/IndexBuilderHelpers.h"
#include "Framework/CompilerBuiltins.h"
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

namespace o2::framework
{
ChunkedArrayIterator::ChunkedArrayIterator(arrow::ChunkedArray* source)
  : mSource{source}
{
  mCurrentArray = getCurrentArray();
  mCurrent = reinterpret_cast<int const*>(mCurrentArray->values()->data()) + mOffset;
  mLast = mCurrent + mCurrentArray->length();
}

SelfIndexColumnBuilder::SelfIndexColumnBuilder(const char* name, arrow::MemoryPool* pool)
  : mColumnName{name}
{
  auto status = arrow::MakeBuilder(pool, arrow::int32(), &mBuilder);
  if (!status.ok()) {
    throw runtime_error("Cannot create array builder!");
  }
}

std::shared_ptr<arrow::Field> SelfIndexColumnBuilder::field() const
{
  return std::make_shared<arrow::Field>(mColumnName, arrow::int32());
}

IndexColumnBuilder::IndexColumnBuilder(arrow::ChunkedArray* source, const char* name, int listSize, arrow::MemoryPool* pool)
  : ChunkedArrayIterator{source},
    SelfIndexColumnBuilder{name, pool},
    mListSize{listSize},
    mSourceSize{(size_t)source->length()}
{
  switch (mListSize) {
    case 1: {
      mValueBuilder = mBuilder.get();
      mArrowType = arrow::int32();
    }; break;
    case 2: {
      mListBuilder = std::make_unique<arrow::FixedSizeListBuilder>(pool, std::move(mBuilder), mListSize);
      mValueBuilder = static_cast<arrow::FixedSizeListBuilder*>(mListBuilder.get())->value_builder();
      mArrowType = arrow::fixed_size_list(arrow::int32(), 2);
    }; break;
    case -1: {
      mListBuilder = std::make_unique<arrow::ListBuilder>(pool, std::move(mBuilder));
      mValueBuilder = static_cast<arrow::ListBuilder*>(mListBuilder.get())->value_builder();
      mArrowType = arrow::list(arrow::int32());
    }; break;
    default:
      throw runtime_error_f("Invalid list size for index column: %d", mListSize);
  }
}

std::shared_ptr<arrow::ChunkedArray> IndexColumnBuilder::resultSingle() const
{
  std::shared_ptr<arrow::Array> array;
  auto status = static_cast<arrow::Int32Builder*>(mValueBuilder)->Finish(&array);
  if (!status.ok()) {
    throw runtime_error("Cannot build an array");
  }
  return std::make_shared<arrow::ChunkedArray>(array);
}

std::shared_ptr<arrow::ChunkedArray> IndexColumnBuilder::resultSlice() const
{
  std::shared_ptr<arrow::Array> array;
  std::shared_ptr<arrow::Array> varray;
  auto status = static_cast<arrow::Int32Builder*>(mValueBuilder)->Finish(&varray);
  if (!status.ok()) {
    throw runtime_error("Cannot build an array");
  }
  array = std::make_shared<arrow::FixedSizeListArray>(mArrowType, mResultSize, varray);
  return std::make_shared<arrow::ChunkedArray>(array);
}

std::shared_ptr<arrow::ChunkedArray> IndexColumnBuilder::resultMulti() const
{
  throw runtime_error("Not implemented");
}

bool IndexColumnBuilder::findSingle(int idx)
{
  size_t step = 0;
  auto count = mSourceSize - mPosition;
  while (count > 0) {
    step = count / 2;
    mPosition += step;
    if (valueAt(mPosition) <= idx) {
      count -= step + 1;
    } else {
      mPosition -= step;
      count = step;
    }
  }

  return (mPosition < mSourceSize && valueAt(mPosition) == idx);
}

bool IndexColumnBuilder::findSlice(int idx)
{
  throw runtime_error("Not implemented");
}

bool IndexColumnBuilder::findMulti(int idx)
{
  throw runtime_error("Not implemented");
}

void IndexColumnBuilder::fillSingle(int idx)
{
  // entry point
  if (mPosition < mSourceSize && valueAt(mPosition) == idx) {
    (void)static_cast<arrow::Int32Builder*>(mValueBuilder)->Append((int)mPosition);
  } else {
    (void)static_cast<arrow::Int32Builder*>(mValueBuilder)->Append(-1);
  }
}

void IndexColumnBuilder::fillSlice(int idx)
{
  throw runtime_error("Not implemented");
}

void IndexColumnBuilder::fillMulti(int idx)
{
  throw runtime_error("Not implemented");
}

std::shared_ptr<arrow::Int32Array> ChunkedArrayIterator::getCurrentArray()
{
  auto chunk = mSource->chunk(mChunk);
  mOffset = chunk->offset();
  return std::static_pointer_cast<arrow::Int32Array>(chunk);
}

void ChunkedArrayIterator::nextChunk()
{
  auto previousArray = getCurrentArray();
  mFirstIndex += previousArray->length();

  ++mChunk;
  auto array = getCurrentArray();
  mCurrent = reinterpret_cast<int const*>(array->values()->data()) + mOffset - mFirstIndex;
  mLast = mCurrent + array->length() + mFirstIndex;
}

void ChunkedArrayIterator::prevChunk()
{
  auto previousArray = getCurrentArray();
  mFirstIndex -= previousArray->length();

  --mChunk;
  auto array = getCurrentArray();
  mCurrent = reinterpret_cast<int const*>(array->values()->data()) + mOffset - mFirstIndex;
  mLast = mCurrent + array->length() + mFirstIndex;
}

int ChunkedArrayIterator::valueAt(size_t pos)
{
  while (O2_BUILTIN_UNLIKELY(mCurrent + pos >= mLast)) {
    nextChunk();
  }
  while (O2_BUILTIN_UNLIKELY(pos < mFirstIndex)) {
    prevChunk();
  }
  return *(mCurrent + pos);
}

std::shared_ptr<arrow::Table> makeArrowTable(const char* label, std::vector<std::shared_ptr<arrow::ChunkedArray>>&& columns, std::vector<std::shared_ptr<arrow::Field>>&& fields)
{
  auto schema = std::make_shared<arrow::Schema>(fields);
  schema->WithMetadata(
    std::make_shared<arrow::KeyValueMetadata>(
      std::vector{std::string{"label"}},
      std::vector{std::string{label}}));
  return arrow::Table::Make(schema, columns);
}
} // namespace o2::framework
