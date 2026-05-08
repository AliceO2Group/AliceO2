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

#include "Framework/RuntimeError.h"
#include "Framework/IndexBuilderHelpers.h"
#include "Framework/CompilerBuiltins.h"
#include "Framework/VariantHelpers.h"
#include <arrow/util/config.h>
#if (ARROW_VERSION_MAJOR > 20)
#include <arrow/compute/initialize.h>
#endif
#include <arrow/compute/kernel.h>
#include <arrow/compute/api_aggregate.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

namespace o2::framework
{
void cannotBuildAnArray(const char* reason)
{
  throw framework::runtime_error_f("Cannot finish an array: %s", reason);
}

void cannotCreateIndexBuilder()
{
  throw framework::runtime_error("Cannot create index column builder: invalid kind of index column");
}

ChunkedArrayIterator::ChunkedArrayIterator(std::shared_ptr<arrow::ChunkedArray> source)
  : mSource{source},
    mSourceSize{(size_t)source->length()}
{
  mCurrentArray = getCurrentArray();
  mCurrent = reinterpret_cast<int const*>(mCurrentArray->values()->data()) + mOffset;
  mLast = mCurrent + mCurrentArray->length();
}

void ChunkedArrayIterator::reset(std::shared_ptr<arrow::ChunkedArray>& source)
{
  mPosition = 0;
  mChunk = 0;
  mOffset = 0;
  mCurrentArray = nullptr;
  mCurrent = nullptr;
  mLast = nullptr;
  mFirstIndex = 0;
  mSourceSize = 0;

  mSource = source;
  mSourceSize = (size_t)source->length();
  mCurrentArray = getCurrentArray();
  mCurrent = reinterpret_cast<int const*>(mCurrentArray->values()->data()) + mOffset;
  mLast = mCurrent + mCurrentArray->length();
}

SelfBuilder::SelfBuilder(arrow::MemoryPool* pool)
{
  auto status = arrow::MakeBuilder(pool, arrow::int32(), &mBuilder);
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot create array builder for the self-index: %s", status.ToString().c_str());
  }
}

void SelfBuilder::reset(std::shared_ptr<arrow::ChunkedArray>)
{
  mBuilder->Reset();
  keyIndex = nullptr;
}

void SelfBuilder::fill(int idx)
{
  auto status = static_cast<arrow::Int32Builder*>(mBuilder.get())->Append(idx);
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot append to self-index array: %s", status.ToString().c_str());
  }
}

std::shared_ptr<arrow::ChunkedArray> SelfBuilder::result() const
{
  std::shared_ptr<arrow::Array> array;
  auto status = static_cast<arrow::Int32Builder*>(mBuilder.get())->Finish(&array);
  if (!status.ok()) {
    cannotBuildAnArray(status.ToString().c_str());
  }

  return std::make_shared<arrow::ChunkedArray>(array);
}

SingleBuilder::SingleBuilder(std::shared_ptr<arrow::ChunkedArray> source, arrow::MemoryPool* pool)
  : ChunkedArrayIterator{source}
{
  auto status = arrow::MakeBuilder(pool, arrow::int32(), &mBuilder);
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot create array builder for the single-valued index: %s", status.ToString().c_str());
  }
}

void SingleBuilder::reset(std::shared_ptr<arrow::ChunkedArray> source)
{
  static_cast<ChunkedArrayIterator*>(this)->reset(source);
  mBuilder->Reset();
}

bool SingleBuilder::find(int idx)
{
  auto count = mSourceSize - mPosition;
  while (count > 0) {
    size_t step = count / 2;
    mPosition += step;
    if (valueAt(mPosition) <= idx) {
      count -= step + 1;
    } else {
      mPosition -= step;
      count = step;
    }
  }

  if (mPosition < mSourceSize && valueAt(mPosition) < idx) {
    ++mPosition;
  }

  return (mPosition < mSourceSize && valueAt(mPosition) == idx);
}

void SingleBuilder::fill(int idx)
{
  arrow::Status status;
  if (mPosition < mSourceSize && valueAt(mPosition) == idx) {
    status = static_cast<arrow::Int32Builder*>(mBuilder.get())->Append((int)mPosition);
  } else {
    status = static_cast<arrow::Int32Builder*>(mBuilder.get())->Append(-1);
  }
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot append to array: %s", status.ToString().c_str());
  }
}

std::shared_ptr<arrow::ChunkedArray> SingleBuilder::result() const
{
  std::shared_ptr<arrow::Array> array;
  auto status = static_cast<arrow::Int32Builder*>(mBuilder.get())->Finish(&array);
  if (!status.ok()) {
    cannotBuildAnArray(status.ToString().c_str());
  }
  return std::make_shared<arrow::ChunkedArray>(array);
}

SliceBuilder::SliceBuilder(std::shared_ptr<arrow::ChunkedArray> source, arrow::MemoryPool* pool)
  : ChunkedArrayIterator{source}
{
  auto status = preSlice();
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot pre-slice the source for slice-index building: %s", status.ToString().c_str());
  }

  std::unique_ptr<arrow::ArrayBuilder> builder;
  status = arrow::MakeBuilder(pool, arrow::int32(), &builder);
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot create array for the slice-index builder: %s", status.ToString().c_str());
  }
  mListBuilder = std::make_unique<arrow::FixedSizeListBuilder>(pool, std::move(builder), 2);
  mValueBuilder = static_cast<arrow::FixedSizeListBuilder*>(mListBuilder.get())->value_builder();
}

void SliceBuilder::reset(std::shared_ptr<arrow::ChunkedArray> source)
{
  mValues = nullptr;
  mCounts = nullptr;
  mListBuilder->Reset();
  mValuePos = 0;
  static_cast<ChunkedArrayIterator*>(this)->reset(source);
  auto status = preSlice();
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot pre-slice the source for slice-index building: %s", status.ToString().c_str());
  }
}

bool SliceBuilder::find(int idx)
{
  auto count = mValues->length() - mValuePos;
  while (count > 0) {
    auto step = count / 2;
    mValuePos += step;
    if (mValues->Value(mValuePos) <= idx) {
      count -= step + 1;
    } else {
      mValuePos -= step;
      count = step;
    }
  }

  if (mValuePos < mValues->length() && mValues->Value(mValuePos) <= idx) {
    ++mPosition;
  }

  return (mValuePos < mValues->length() && mValues->Value(mValuePos) == idx);
}

void SliceBuilder::fill(int idx)
{
  int data[2] = {-1, -1};
  if (mValuePos < mValues->length() && mValues->Value(mValuePos) == idx) {
    for (auto i = 0; i < mValuePos; ++i) {
      data[0] += mCounts->Value(i);
    }
    data[0] += 1;
    data[1] = data[0] + mCounts->Value(mValuePos) - 1;
  }
  (void)static_cast<arrow::FixedSizeListBuilder*>(mListBuilder.get())->AppendValues(1);
  (void)static_cast<arrow::Int32Builder*>(mValueBuilder)->AppendValues(data, 2);
}

std::shared_ptr<arrow::ChunkedArray> SliceBuilder::result() const
{
  std::shared_ptr<arrow::Array> array;
  auto status = static_cast<arrow::FixedSizeListBuilder*>(mListBuilder.get())->Finish(&array);
  if (!status.ok()) {
    cannotBuildAnArray(status.ToString().c_str());
  }
  return std::make_shared<arrow::ChunkedArray>(array);
}

arrow::Status SliceBuilder::SliceBuilder::preSlice()
{
#if (ARROW_VERSION_MAJOR > 20)
  auto status = arrow::compute::Initialize();
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot initialize arrow compute: %s", status.ToString().c_str());
  }
#else
  arrow::Status status;
#endif
  arrow::Datum value_counts;
  auto options = arrow::compute::ScalarAggregateOptions::Defaults();
  ARROW_ASSIGN_OR_RAISE(value_counts, arrow::compute::CallFunction("value_counts", {mSource}, &options));
  auto pair = static_cast<arrow::StructArray>(value_counts.array());
  mValues = std::make_shared<arrow::NumericArray<arrow::Int32Type>>(pair.field(0)->data());
  mCounts = std::make_shared<arrow::NumericArray<arrow::Int64Type>>(pair.field(1)->data());
  return arrow::Status::OK();
}

ArrayBuilder::ArrayBuilder(std::shared_ptr<arrow::ChunkedArray> source, arrow::MemoryPool* pool)
  : ChunkedArrayIterator{source}
{
  auto&& status = preFind();
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot pre-find in a source for array-index building: %s", status.ToString().c_str());
  }

  std::unique_ptr<arrow::ArrayBuilder> builder;
  status = arrow::MakeBuilder(pool, arrow::int32(), &builder);
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot create array for the array-index builder: %s", status.ToString().c_str());
  }
  mListBuilder = std::make_unique<arrow::ListBuilder>(pool, std::move(builder));
  mValueBuilder = static_cast<arrow::ListBuilder*>(mListBuilder.get())->value_builder();
}

void ArrayBuilder::reset(std::shared_ptr<arrow::ChunkedArray> source)
{
  static_cast<ChunkedArrayIterator*>(this)->reset(source);
  auto status = preFind();
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot pre-find in a source for array-index building: %s", status.ToString().c_str());
  }
  mValues.clear();
  mIndices.clear();
  mListBuilder->Reset();
}

bool ArrayBuilder::find(int idx)
{
  return (std::find(mValues.begin(), mValues.end(), idx) != mValues.end());
}

void ArrayBuilder::fill(int idx)
{
  (void)static_cast<arrow::ListBuilder*>(mListBuilder.get())->Append();
  if (std::find(mValues.begin(), mValues.end(), idx) != mValues.end()) {
    (void)static_cast<arrow::Int32Builder*>(mValueBuilder)->AppendValues(mIndices[idx].data(), mIndices[idx].size());
  } else {
    (void)static_cast<arrow::Int32Builder*>(mValueBuilder)->AppendValues(nullptr, 0);
  }
}

std::shared_ptr<arrow::ChunkedArray> ArrayBuilder::result() const
{
  std::shared_ptr<arrow::Array> array;
  auto status = static_cast<arrow::ListBuilder*>(mListBuilder.get())->Finish(&array);
  if (!status.ok()) {
    cannotBuildAnArray(status.ToString().c_str());
  }
  return std::make_shared<arrow::ChunkedArray>(array);
}

arrow::Status ArrayBuilder::preFind()
{
#if (ARROW_VERSION_MAJOR > 20)
  auto status = arrow::compute::Initialize();
  if (!status.ok()) {
    throw framework::runtime_error_f("Cannot initialize arrow compute: %s", status.ToString().c_str());
  }
#else
  arrow::Status status;
#endif
  arrow::Datum max;
  auto options = arrow::compute::ScalarAggregateOptions::Defaults();
  ARROW_ASSIGN_OR_RAISE(max, arrow::compute::CallFunction("max", {mSource}, &options));
  auto maxValue = std::dynamic_pointer_cast<arrow::Int32Scalar>(max.scalar())->value;
  mIndices.resize(maxValue + 1);

  auto row = 0;
  for (auto i = 0; i < mSource->length(); ++i) {
    auto v = valueAt(i);
    if (v >= 0) {
      mValues.emplace_back(v);
      mIndices[v].push_back(row);
    }
    ++row;
  }
  std::sort(mValues.begin(), mValues.end());

  return arrow::Status::OK();
}

IndexColumnBuilder::IndexColumnBuilder(soa::IndexKind kind, int pos, arrow::MemoryPool* pool, std::shared_ptr<arrow::ChunkedArray> source)
  : mColumnPos{pos}
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

void IndexColumnBuilder::reset(std::shared_ptr<arrow::ChunkedArray> source)
{
  std::visit(
    overloaded{
      [](std::monostate) {},
      [&source](auto& b) { b.reset(source); }},
    builder);
}

bool IndexColumnBuilder::find(int idx)
{
  return std::visit(
    overloaded{
      [](std::monostate) { return false; },
      [&idx](auto& b) { return b.find(idx); },
    },
    builder);
}

void IndexColumnBuilder::fill(int idx)
{
  std::visit(
    overloaded{
      [](std::monostate) {},
      [&idx](auto& b) { b.fill(idx); }},
    builder);
}

std::shared_ptr<arrow::ChunkedArray> IndexColumnBuilder::result() const
{
  return std::visit(
    overloaded{
      [](std::monostate) -> std::shared_ptr<arrow::ChunkedArray> { return nullptr; },
      [](auto& b) { return b.result(); }},
    builder);
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
} // namespace o2::framework
