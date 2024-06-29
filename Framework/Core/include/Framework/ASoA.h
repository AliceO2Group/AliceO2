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

#ifndef O2_FRAMEWORK_ASOA_H_
#define O2_FRAMEWORK_ASOA_H_

#include "Framework/Pack.h"
#include "Framework/FunctionalHelpers.h"
#include "Headers/DataHeader.h"
#include "Framework/CompilerBuiltins.h"
#include "Framework/Traits.h"
#include "Framework/Expressions.h"
#include "Framework/ArrowTypes.h"
#include "Framework/ArrowTableSlicingCache.h"
#include "Framework/SliceCache.h"
#include "Framework/VariantHelpers.h"
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/util/config.h>
#include <gandiva/selection_vector.h>
#include <cassert>
#include <fmt/format.h>
#include <gsl/span>
#include <limits>

#define DECLARE_SOA_METADATA()       \
  template <typename T>              \
  struct MetadataTrait {             \
    using metadata = std::void_t<T>; \
  };

namespace o2::aod
{
DECLARE_SOA_METADATA();
}

namespace o2::soa
{

struct Binding {
  void const* ptr = nullptr;
  size_t hash = 0;

  template <typename T>
  void bind(T const* table)
  {
    ptr = table;
    hash = o2::framework::TypeIdHelpers::uniqueId<T>();
  }

  template <typename T>
  T const* get() const
  {
    if (hash == o2::framework::TypeIdHelpers::uniqueId<T>()) {
      return static_cast<T const*>(ptr);
    }
    return nullptr;
  }
};

void accessingInvalidIndexFor(const char* getter);
void dereferenceWithWrongType();
void missingFilterDeclaration(int hash, int ai);

template <typename... C>
auto createFieldsFromColumns(framework::pack<C...>)
{
  return std::vector<std::shared_ptr<arrow::Field>>{C::asArrowField()...};
}

using SelectionVector = std::vector<int64_t>;

template <typename, typename = void>
inline constexpr bool is_index_column_v = false;

template <typename T>
inline constexpr bool is_index_column_v<T, std::void_t<decltype(sizeof(typename T::binding_t))>> = true;

template <typename, typename = void>
inline constexpr bool is_type_with_originals_v = false;

template <typename T>
inline constexpr bool is_type_with_originals_v<T, std::void_t<decltype(sizeof(typename T::originals))>> = true;

template <typename T, typename = void>
inline constexpr bool is_type_with_parent_v = false;

template <typename T>
inline constexpr bool is_type_with_parent_v<T, std::void_t<decltype(sizeof(typename T::parent_t))>> = true;

template <typename, typename = void>
inline constexpr bool is_type_with_metadata_v = false;

template <typename T>
inline constexpr bool is_type_with_metadata_v<T, std::void_t<decltype(sizeof(typename T::metadata))>> = true;

template <typename, typename = void>
inline constexpr bool is_type_with_binding_v = false;

template <typename T>
inline constexpr bool is_type_with_binding_v<T, std::void_t<decltype(sizeof(typename T::binding_t))>> = true;

template <typename, typename = void>
inline constexpr bool is_type_spawnable_v = false;

template <typename T>
inline constexpr bool is_type_spawnable_v<T, std::void_t<decltype(sizeof(typename T::spawnable_t))>> = true;

template <typename, typename = void>
inline constexpr bool is_soa_extension_table_v = false;

template <typename T>
inline constexpr bool is_soa_extension_table_v<T, std::void_t<decltype(sizeof(typename T::expression_pack_t))>> = true;

template <typename T, typename = void>
inline constexpr bool is_index_table_v = false;

template <typename T>
inline constexpr bool is_index_table_v<T, std::void_t<decltype(sizeof(typename T::indexing_t))>> = true;

template <typename, typename = void>
inline constexpr bool is_self_index_column_v = false;

template <typename T>
inline constexpr bool is_self_index_column_v<T, std::void_t<decltype(sizeof(typename T::self_index_t))>> = true;

template <typename, typename = void>
inline constexpr bool is_with_base_table_v = false;

template <typename T>
inline constexpr bool is_with_base_table_v<T, std::void_t<decltype(sizeof(typename T::base_table_t))>> = true;

template <typename B, typename E>
struct EquivalentIndex {
  constexpr static bool value = false;
};

template <typename B, typename E>
constexpr bool is_index_equivalent_v = EquivalentIndex<B, E>::value || EquivalentIndex<E, B>::value;

template <typename T, typename TLambda>
void call_if_has_originals(TLambda&& lambda)
{
  if constexpr (is_type_with_originals_v<T>) {
    lambda(static_cast<T*>(nullptr));
  }
}

template <typename T, typename TLambda>
void call_if_has_not_originals(TLambda&& lambda)
{
  if constexpr (!is_type_with_originals_v<T>) {
    lambda(static_cast<T*>(nullptr));
  }
}

template <typename H, typename... T>
constexpr auto make_originals_from_type()
{
  using decayed = std::decay_t<H>;
  if constexpr (sizeof...(T) == 0) {
    if constexpr (is_type_with_originals_v<decayed>) {
      return typename decayed::originals{};
    } else if constexpr (is_type_with_originals_v<typename decayed::table_t>) {
      return typename decayed::table_t::originals{};
    } else if constexpr (is_type_with_parent_v<decayed>) {
      return make_originals_from_type<typename decayed::parent_t>();
    } else {
      return framework::pack<decayed>{};
    }
  } else if constexpr (is_type_with_originals_v<decayed>) {
    return framework::concatenate_pack(typename decayed::originals{}, make_originals_from_type<T...>());
  } else if constexpr (is_type_with_originals_v<typename decayed::table_t>) {
    return framework::concatenate_pack(typename decayed::table_t::originals{}, make_originals_from_type<T...>());
  } else {
    return framework::concatenate_pack(framework::pack<decayed>{}, make_originals_from_type<T...>());
  }
}

template <typename... T>
constexpr auto make_originals_from_type(framework::pack<T...>)
{
  return make_originals_from_type<T...>();
}

/// Policy class for columns which are chunked. This
/// will make the compiler take the most generic (and
/// slow approach).
struct Chunked {
  constexpr static bool chunked = true;
};

/// Policy class for columns which are known to be fully
/// inside a chunk. This will generate optimal code.
struct Flat {
  constexpr static bool chunked = false;
};

/// unwrapper
template <typename T>
struct unwrap {
  using type = T;
};

template <typename T>
struct unwrap<std::vector<T>> {
  using type = T;
};

template <>
struct unwrap<bool> {
  using type = char;
};

template <typename T>
using unwrap_t = typename unwrap<T>::type;

/// Iterator on a single column.
/// FIXME: the ChunkingPolicy for now is fixed to Flat and is a mere boolean
/// which is used to switch off slow "chunking aware" parts. This is ok for
/// now, but most likely we should move the whole chunk navigation logic there.
template <typename T, typename ChunkingPolicy = Chunked>
class ColumnIterator : ChunkingPolicy
{
  static constexpr char SCALE_FACTOR = std::is_same_v<std::decay_t<T>, bool> ? 3 : 0;

 public:
  /// Constructor of the column iterator. Notice how it takes a pointer
  /// to the ChunkedArray (for the data store) and to the index inside
  /// it. This means that a ColumnIterator is actually only available
  /// as part of a RowView.
  ColumnIterator(arrow::ChunkedArray const* column)
    : mColumn{column},
      mCurrent{nullptr},
      mCurrentPos{nullptr},
      mLast{nullptr},
      mFirstIndex{0},
      mCurrentChunk{0},
      mOffset{0}
  {
    auto array = getCurrentArray();
    mCurrent = reinterpret_cast<unwrap_t<T> const*>(array->values()->data()) + (mOffset >> SCALE_FACTOR);
    mLast = mCurrent + array->length();
  }

  ColumnIterator() = default;
  ColumnIterator(ColumnIterator<T, ChunkingPolicy> const&) = default;
  ColumnIterator<T, ChunkingPolicy>& operator=(ColumnIterator<T, ChunkingPolicy> const&) = default;

  ColumnIterator(ColumnIterator<T, ChunkingPolicy>&&) = default;
  ColumnIterator<T, ChunkingPolicy>& operator=(ColumnIterator<T, ChunkingPolicy>&&) = default;

  /// Move the iterator to the next chunk.
  void nextChunk() const
  {
    auto previousArray = getCurrentArray();
    mFirstIndex += previousArray->length();

    mCurrentChunk++;
    auto array = getCurrentArray();
    mCurrent = reinterpret_cast<unwrap_t<T> const*>(array->values()->data()) + (mOffset >> SCALE_FACTOR) - (mFirstIndex >> SCALE_FACTOR);
    mLast = mCurrent + array->length() + (mFirstIndex >> SCALE_FACTOR);
  }

  void prevChunk() const
  {
    auto previousArray = getCurrentArray();
    mFirstIndex -= previousArray->length();

    mCurrentChunk--;
    auto array = getCurrentArray();
    mCurrent = reinterpret_cast<unwrap_t<T> const*>(array->values()->data()) + (mOffset >> SCALE_FACTOR) - (mFirstIndex >> SCALE_FACTOR);
    mLast = mCurrent + array->length() + (mFirstIndex >> SCALE_FACTOR);
  }

  void moveToChunk(int chunk)
  {
    if (mCurrentChunk < chunk) {
      while (mCurrentChunk != chunk) {
        nextChunk();
      }
    } else {
      while (mCurrentChunk != chunk) {
        prevChunk();
      }
    }
  }

  /// Move the iterator to the end of the column.
  void moveToEnd()
  {
    mCurrentChunk = mColumn->num_chunks() - 1;
    auto array = getCurrentArray();
    mFirstIndex = mColumn->length() - array->length();
    mCurrent = reinterpret_cast<unwrap_t<T> const*>(array->values()->data()) + (mOffset >> SCALE_FACTOR) - (mFirstIndex >> SCALE_FACTOR);
    mLast = mCurrent + array->length() + (mFirstIndex >> SCALE_FACTOR);
  }

  decltype(auto) operator*() const
  {
    if constexpr (ChunkingPolicy::chunked) {
      if constexpr (std::is_same_v<arrow_array_for_t<T>, arrow::ListArray>) {
        auto list = std::static_pointer_cast<arrow::ListArray>(mColumn->chunk(mCurrentChunk));
        if (O2_BUILTIN_UNLIKELY(*mCurrentPos - mFirstIndex >= list->length())) {
          nextChunk();
        }
      } else {
        if (O2_BUILTIN_UNLIKELY(((mCurrent + (*mCurrentPos >> SCALE_FACTOR)) >= mLast))) {
          nextChunk();
        }
      }
    }
    if constexpr (std::is_same_v<bool, std::decay_t<T>>) {
      // FIXME: check if shifting the masked bit to the first position is better than != 0
      return (*(mCurrent - (mOffset >> SCALE_FACTOR) + ((*mCurrentPos + mOffset) >> SCALE_FACTOR)) & (1 << ((*mCurrentPos + mOffset) & 0x7))) != 0;
    } else if constexpr (std::is_same_v<arrow_array_for_t<T>, arrow::ListArray>) {
      auto list = std::static_pointer_cast<arrow::ListArray>(mColumn->chunk(mCurrentChunk));
      auto offset = list->value_offset(*mCurrentPos - mFirstIndex);
      auto length = list->value_length(*mCurrentPos - mFirstIndex);
      return gsl::span{mCurrent + mFirstIndex + offset, mCurrent + mFirstIndex + (offset + length)};
    } else {
      return *(mCurrent + (*mCurrentPos >> SCALE_FACTOR));
    }
  }

  // Move to the chunk which containts element pos
  ColumnIterator<T>& moveToPos()
  {
    // If we get outside range of the current chunk, go to the next.
    if constexpr (ChunkingPolicy::chunked) {
      while (O2_BUILTIN_UNLIKELY((mCurrent + (*mCurrentPos >> SCALE_FACTOR)) >= mLast)) {
        nextChunk();
      }
    }
    return *this;
  }

  // Move to the chunk which containts element pos
  ColumnIterator<T>& checkNextChunk()
  {
    if constexpr (ChunkingPolicy::chunked) {
      if (O2_BUILTIN_LIKELY((mCurrent + (*mCurrentPos >> SCALE_FACTOR)) <= mLast)) {
        return *this;
      }
      nextChunk();
    }
    return *this;
  }

  mutable unwrap_t<T> const* mCurrent;
  int64_t const* mCurrentPos;
  mutable unwrap_t<T> const* mLast;
  arrow::ChunkedArray const* mColumn;
  mutable int mFirstIndex;
  mutable int mCurrentChunk;
  mutable int mOffset;

 private:
  /// get pointer to mCurrentChunk chunk
  auto getCurrentArray() const
  {
    std::shared_ptr<arrow::Array> chunkToUse = mColumn->chunk(mCurrentChunk);
    mOffset = chunkToUse->offset();
    if constexpr (std::is_same_v<arrow_array_for_t<T>, arrow::FixedSizeListArray>) {
      chunkToUse = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(chunkToUse)->values();
      return std::static_pointer_cast<arrow_array_for_t<value_for_t<T>>>(chunkToUse);
    } else if constexpr (std::is_same_v<arrow_array_for_t<T>, arrow::ListArray>) {
      chunkToUse = std::dynamic_pointer_cast<arrow::ListArray>(chunkToUse)->values();
      mOffset = chunkToUse->offset();
      return std::static_pointer_cast<arrow_array_for_t<value_for_t<T>>>(chunkToUse);
    } else {
      return std::static_pointer_cast<arrow_array_for_t<T>>(chunkToUse);
    }
  }
};

template <typename T, typename INHERIT>
struct Column {
  using inherited_t = INHERIT;
  Column(ColumnIterator<T> const& it)
    : mColumnIterator{it}
  {
  }

  Column() = default;
  Column(Column const&) = default;
  Column& operator=(Column const&) = default;

  Column(Column&&) = default;
  Column& operator=(Column&&) = default;

  using persistent = std::true_type;
  using type = T;
  static constexpr const char* const& columnLabel() { return INHERIT::mLabel; }
  ColumnIterator<T> const& getIterator() const
  {
    return mColumnIterator;
  }

  static auto asArrowField()
  {
    return std::make_shared<arrow::Field>(inherited_t::mLabel, framework::expressions::concreteArrowType(framework::expressions::selectArrowType<type>()));
  }

  /// FIXME: rather than keeping this public we should have a protected
  /// non-const getter and mark this private.
  ColumnIterator<T> mColumnIterator;
};

/// The purpose of this class is to store the lambda which is associated to the
/// method call.
template <typename F, typename INHERIT>
struct DynamicColumn {
  using inherited_t = INHERIT;

  using persistent = std::false_type;
  static constexpr const char* const& columnLabel() { return INHERIT::mLabel; }
};

template <typename INHERIT>
struct IndexColumn {
  using inherited_t = INHERIT;

  using persistent = std::false_type;
  static constexpr const char* const& columnLabel() { return INHERIT::mLabel; }
};

template <typename INHERIT>
struct MarkerColumn {
  using inherited_t = INHERIT;

  using persistent = std::false_type;
  static constexpr const char* const& columnLabel() { return INHERIT::mLabel; }
};

template <size_t M = 0>
struct Marker : o2::soa::MarkerColumn<Marker<M>> {
  using type = size_t;
  using base = o2::soa::MarkerColumn<Marker<M>>;
  constexpr inline static auto value = M;

  Marker() = default;
  Marker(Marker const&) = default;
  Marker(Marker&&) = default;

  Marker& operator=(Marker const&) = default;
  Marker& operator=(Marker&&) = default;

  Marker(arrow::ChunkedArray const*) {}
  constexpr inline auto mark()
  {
    return value;
  }

  static constexpr const char* mLabel = "Marker";
};

template <int64_t START = 0, int64_t END = -1>
struct Index : o2::soa::IndexColumn<Index<START, END>> {
  using base = o2::soa::IndexColumn<Index<START, END>>;
  constexpr inline static int64_t start = START;
  constexpr inline static int64_t end = END;

  Index() = default;
  Index(Index const&) = default;
  Index(Index&&) = default;

  Index& operator=(Index const&) = default;
  Index& operator=(Index&&) = default;

  Index(arrow::ChunkedArray const*)
  {
  }

  constexpr inline int64_t rangeStart()
  {
    return START;
  }

  constexpr inline int64_t rangeEnd()
  {
    return END;
  }

  [[nodiscard]] int64_t index() const
  {
    return index<0>();
  }

  [[nodiscard]] int64_t filteredIndex() const
  {
    return index<1>();
  }

  [[nodiscard]] int64_t globalIndex() const
  {
    return index<0>() + offsets<0>();
  }

  template <int N = 0>
  [[nodiscard]] int64_t index() const
  {
    return *std::get<N>(rowIndices);
  }

  template <int N = 0>
  [[nodiscard]] int64_t offsets() const
  {
    return *std::get<N>(rowOffsets);
  }

  void setIndices(std::tuple<int64_t const*, int64_t const*> indices)
  {
    rowIndices = indices;
  }

  void setOffsets(std::tuple<uint64_t const*> offsets)
  {
    rowOffsets = offsets;
  }

  static constexpr const char* mLabel = "Index";
  using type = int64_t;

  using bindings_t = typename o2::framework::pack<>;
  std::tuple<> boundIterators;
  std::tuple<int64_t const*, int64_t const*> rowIndices;
  /// The offsets within larger tables. Currently only
  /// one level of nesting is supported.
  std::tuple<uint64_t const*> rowOffsets;
};

template <typename T>
using is_dynamic_t = framework::is_specialization<typename T::base, DynamicColumn>;

namespace persistent_type_helper
{
// This checks both for the existence of the ::persistent member in the class T as well as the value returned stored in it.
// Hack: a pointer to any field of type int inside persistent. Both true_type and false_type do not have any int field, but anyways we pass nullptr.
// The compiler picks the version with exact number of arguments when only it can, i.e., when T::persistent is defined.
template <class T>
typename T::persistent test(int T::persistent::*);

template <class>
std::false_type test(...);
} // namespace persistent_type_helper

template <typename T>
using is_persistent_t = decltype(persistent_type_helper::test<T>(nullptr));

template <typename T>
constexpr auto is_persistent_v = is_persistent_t<T>::value;

template <typename T>
constexpr auto is_dynamic_v = is_dynamic_t<T>::value;

template <typename T>
using is_external_index_t = typename std::conditional<is_index_column_v<T>, std::true_type, std::false_type>::type;

template <typename T>
using is_self_index_t = typename std::conditional<is_self_index_column_v<T>, std::true_type, std::false_type>::type;

template <typename T, template <auto...> class Ref>
struct is_index : std::false_type {
};

template <template <auto...> class Ref, auto... Args>
struct is_index<Ref<Args...>, Ref> : std::true_type {
};

template <typename T>
using is_index_t = is_index<T, Index>;

struct IndexPolicyBase {
  /// Position inside the current table
  int64_t mRowIndex = 0;
  /// Offset within a larger table
  uint64_t mOffset = 0;
};

struct RowViewSentinel {
  int64_t const index;
};

struct FilteredIndexPolicy : IndexPolicyBase {
  // We use -1 in the IndexPolicyBase to indicate that the index is
  // invalid. What will validate the index is the this->setCursor()
  // which happens below which will properly setup the first index
  // by remapping the filtered index 0 to whatever unfiltered index
  // it belongs to.
  FilteredIndexPolicy(gsl::span<int64_t const> selection, int64_t rows, uint64_t offset = 0)
    : IndexPolicyBase{-1, offset},
      mSelectedRows(selection),
      mMaxSelection(selection.size()),
      nRows{rows}
  {
    this->setCursor(0);
  }

  void resetSelection(gsl::span<int64_t const> selection)
  {
    mSelectedRows = selection;
    mMaxSelection = selection.size();
    this->setCursor(0);
  }

  FilteredIndexPolicy() = default;
  FilteredIndexPolicy(FilteredIndexPolicy&&) = default;
  FilteredIndexPolicy(FilteredIndexPolicy const&) = default;
  FilteredIndexPolicy& operator=(FilteredIndexPolicy const&) = default;
  FilteredIndexPolicy& operator=(FilteredIndexPolicy&&) = default;

  [[nodiscard]] std::tuple<int64_t const*, int64_t const*>
    getIndices() const
  {
    return std::make_tuple(&mRowIndex, &mSelectionRow);
  }

  [[nodiscard]] std::tuple<uint64_t const*>
    getOffsets() const
  {
    return std::make_tuple(&mOffset);
  }

  void limitRange(int64_t start, int64_t end)
  {
    this->setCursor(start);
    if (end >= 0) {
      mMaxSelection = std::min(end, mMaxSelection);
    }
  }

  void setCursor(int64_t i)
  {
    mSelectionRow = i;
    updateRow();
  }

  void moveByIndex(int64_t i)
  {
    mSelectionRow += i;
    updateRow();
  }

  friend bool operator==(FilteredIndexPolicy const& lh, FilteredIndexPolicy const& rh)
  {
    return lh.mSelectionRow == rh.mSelectionRow;
  }

  bool operator==(RowViewSentinel const& sentinel) const
  {
    return O2_BUILTIN_UNLIKELY(mSelectionRow == sentinel.index);
  }

  /// Move iterator to one after the end. Since this is a view
  /// we move the mSelectionRow to one past the view size and
  /// the mRowIndex to one past the last entry in the selection
  void moveToEnd()
  {
    this->mSelectionRow = this->mMaxSelection;
    this->mRowIndex = -1;
  }

  [[nodiscard]] auto getSelectionRow() const
  {
    return mSelectionRow;
  }

  [[nodiscard]] auto size() const
  {
    return mMaxSelection;
  }

  [[nodiscard]] auto raw_size() const
  {
    return nRows;
  }

 private:
  inline void updateRow()
  {
    this->mRowIndex = O2_BUILTIN_LIKELY(mSelectionRow < mMaxSelection) ? mSelectedRows[mSelectionRow] : -1;
  }
  gsl::span<int64_t const> mSelectedRows;
  int64_t mSelectionRow = 0;
  int64_t mMaxSelection = 0;
  int64_t nRows = 0;
};

struct DefaultIndexPolicy : IndexPolicyBase {
  /// Needed to be able to copy the policy
  DefaultIndexPolicy() = default;
  DefaultIndexPolicy(DefaultIndexPolicy&&) = default;
  DefaultIndexPolicy(DefaultIndexPolicy const&) = default;
  DefaultIndexPolicy& operator=(DefaultIndexPolicy const&) = default;
  DefaultIndexPolicy& operator=(DefaultIndexPolicy&&) = default;

  /// mMaxRow is one behind the last row, so effectively equal to the number of
  /// rows @a nRows. Offset indicates that the index is actually part of
  /// a larger
  DefaultIndexPolicy(int64_t nRows, uint64_t offset)
    : IndexPolicyBase{0, offset},
      mMaxRow(nRows)
  {
  }

  DefaultIndexPolicy(FilteredIndexPolicy const& other)
    : IndexPolicyBase{0, other.mOffset},
      mMaxRow(other.raw_size())
  {
  }

  void limitRange(int64_t start, int64_t end)
  {
    this->setCursor(start);
    if (end >= 0) {
      mMaxRow = std::min(end, mMaxRow);
    }
  }

  [[nodiscard]] std::tuple<int64_t const*, int64_t const*>
    getIndices() const
  {
    return std::make_tuple(&mRowIndex, &mRowIndex);
  }

  [[nodiscard]] std::tuple<uint64_t const*>
    getOffsets() const
  {
    return std::make_tuple(&mOffset);
  }

  void setCursor(int64_t i)
  {
    this->mRowIndex = i;
  }
  void moveByIndex(int64_t i)
  {
    this->mRowIndex += i;
  }

  void moveToEnd()
  {
    this->setCursor(mMaxRow);
  }

  friend bool operator==(DefaultIndexPolicy const& lh, DefaultIndexPolicy const& rh)
  {
    return lh.mRowIndex == rh.mRowIndex;
  }

  bool operator==(RowViewSentinel const& sentinel) const
  {
    return O2_BUILTIN_UNLIKELY(this->mRowIndex == sentinel.index);
  }

  [[nodiscard]] auto size() const
  {
    return mMaxRow;
  }

  int64_t mMaxRow = 0;
};

template <typename... C>
class Table;

/// Similar to a pair but not a pair, to avoid
/// exposing the second type everywhere.
template <typename C>
struct ColumnDataHolder {
  C* first;
  arrow::ChunkedArray* second;
};

template <typename IP, typename... C>
struct RowViewCore : public IP, C... {
 public:
  using policy_t = IP;
  using table_t = o2::soa::Table<C...>;
  using all_columns = framework::pack<C...>;
  using persistent_columns_t = framework::selected_pack<is_persistent_t, C...>;
  using index_columns_t = framework::selected_pack<is_index_t, C...>;
  constexpr inline static bool has_index_v = framework::pack_size(index_columns_t{}) > 0;
  using external_index_columns_t = framework::selected_pack<is_external_index_t, C...>;
  using internal_index_columns_t = framework::selected_pack<is_self_index_t, C...>;

  RowViewCore(arrow::ChunkedArray* columnData[sizeof...(C)], IP&& policy)
    : IP{policy},
      C(columnData[framework::has_type_at_v<C>(all_columns{})])...
  {
    bind();
    // In case we have an index column might need to constrain the actual
    // number of rows in the view to the range provided by the index.
    // FIXME: we should really understand what happens to an index when we
    // have a RowViewFiltered.
    if constexpr (has_index_v) {
      this->limitRange(this->rangeStart(), this->rangeEnd());
    }
  }

  RowViewCore() = default;
  RowViewCore(RowViewCore<IP, C...> const& other)
    : IP{static_cast<IP const&>(other)},
      C(static_cast<C const&>(other))...
  {
    bind();
  }

  RowViewCore& operator=(RowViewCore other)
  {
    IP::operator=(static_cast<IP const&>(other));
    (void(static_cast<C&>(*this) = static_cast<C>(other)), ...);
    bind();
    return *this;
  }

  RowViewCore(RowViewCore<FilteredIndexPolicy, C...> const& other) requires std::is_same_v<IP, DefaultIndexPolicy>
    : IP{static_cast<IP const&>(other)},
      C(static_cast<C const&>(other))...
  {
    bind();
  }

  RowViewCore& operator++()
  {
    this->moveByIndex(1);
    return *this;
  }

  RowViewCore operator++(int)
  {
    RowViewCore<IP, C...> copy = *this;
    this->operator++();
    return copy;
  }

  RowViewCore& operator--()
  {
    this->moveByIndex(-1);
    return *this;
  }

  RowViewCore operator--(int)
  {
    RowViewCore<IP, C...> copy = *this;
    this->operator--();
    return copy;
  }

  /// Allow incrementing by more than one the iterator
  RowViewCore operator+(int64_t inc) const
  {
    RowViewCore copy = *this;
    copy.moveByIndex(inc);
    return copy;
  }

  RowViewCore operator-(int64_t dec) const
  {
    return operator+(-dec);
  }

  RowViewCore const& operator*() const
  {
    return *this;
  }

  template <typename... CL, typename TA>
  void doSetCurrentIndex(framework::pack<CL...>, TA* current)
  {
    (CL::setCurrent(current), ...);
  }

  template <typename CL>
  auto getCurrent() const
  {
    return CL::getCurrentRaw();
  }

  template <typename... Cs>
  auto getIndexBindingsImpl(framework::pack<Cs...>) const
  {
    return std::vector<o2::soa::Binding>{static_cast<Cs const&>(*this).getCurrentRaw()...};
  }

  auto getIndexBindings() const
  {
    return getIndexBindingsImpl(external_index_columns_t{});
  }

  template <typename... TA>
  void bindExternalIndices(TA*... current)
  {
    (doSetCurrentIndex(external_index_columns_t{}, current), ...);
  }

  template <typename... Cs>
  void doSetCurrentIndexRaw(framework::pack<Cs...> p, std::vector<o2::soa::Binding>&& ptrs)
  {
    (Cs::setCurrentRaw(ptrs[framework::has_type_at_v<Cs>(p)]), ...);
  }

  template <typename... Cs, typename I>
  void doSetCurrentInternal(framework::pack<Cs...>, I const* ptr)
  {
    o2::soa::Binding b;
    b.bind(ptr);
    (Cs::setCurrentRaw(b), ...);
  }

  void bindExternalIndicesRaw(std::vector<o2::soa::Binding>&& ptrs)
  {
    doSetCurrentIndexRaw(external_index_columns_t{}, std::forward<std::vector<o2::soa::Binding>>(ptrs));
  }

  template <typename I>
  void bindInternalIndices(I const* table)
  {
    doSetCurrentInternal(internal_index_columns_t{}, table);
  }

 private:
  /// Helper to move to the correct chunk, if needed.
  /// FIXME: not needed?
  template <typename... PC>
  void checkNextChunk(framework::pack<PC...>)
  {
    (PC::mColumnIterator.checkNextChunk(), ...);
  }

  /// Helper to move at the end of columns which actually have an iterator.
  template <typename... PC>
  void doMoveToEnd(framework::pack<PC...>)
  {
    (PC::mColumnIterator.moveToEnd(), ...);
  }

  /// Helper which binds all the ColumnIterators to the
  /// index of a the associated RowView
  void bind()
  {
    using namespace o2::soa;
    auto f = framework::overloaded  {
      [this]<typename T>(T*) -> void requires is_persistent_v<T> { T::mColumnIterator.mCurrentPos = &this->mRowIndex; },
      [this]<typename T>(T*) -> void requires is_dynamic_v<T> { bindDynamicColumn<T>(typename T::bindings_t{});},
      [this]<typename T>(T*) -> void {},
    };
    (f(static_cast<C*>(nullptr)), ...);
    if constexpr (has_index_v) {
      this->setIndices(this->getIndices());
      this->setOffsets(this->getOffsets());
    }
  }

  template <typename DC, typename... B>
  auto bindDynamicColumn(framework::pack<B...>)
  {
    DC::boundIterators = std::make_tuple(&(B::mColumnIterator)...);
  }
};

template <typename, typename = void>
constexpr bool is_type_with_policy_v = false;

template <typename T>
constexpr bool is_type_with_policy_v<T, std::void_t<decltype(sizeof(typename T::policy_t))>> = true;

struct ArrowHelpers {
  static std::shared_ptr<arrow::Table> joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables);
  static std::shared_ptr<arrow::Table> concatTables(std::vector<std::shared_ptr<arrow::Table>>&& tables);
};

template <typename... T>
using originals_pack_t = decltype(make_originals_from_type<T...>());

template <typename T, typename... Os>
constexpr bool are_bindings_compatible_v(framework::pack<Os...>&&)
{
  if constexpr (is_type_with_originals_v<T>) {
    return (are_bindings_compatible_v<Os>(originals_pack_t<T>{}) || ...);
  } else {
    return ((std::is_same_v<T, Os> || is_index_equivalent_v<T, Os>) || ...);
  }
}

template <typename T, typename B>
constexpr bool is_binding_compatible_v()
{
  return are_bindings_compatible_v<T>(originals_pack_t<B>{});
}

template <typename T, typename B>
struct is_binding_compatible : std::conditional_t<is_binding_compatible_v<T, typename B::binding_t>(), std::true_type, std::false_type> {
};

template <typename T>
static constexpr std::string getLabelFromType()
{
  if constexpr (soa::is_index_table_v<std::decay_t<T>>) {
    using TT = typename std::decay_t<T>::first_t;
    if constexpr (soa::is_type_with_originals_v<std::decay_t<TT>>) {
      using O = typename framework::pack_head_t<typename std::decay_t<TT>::originals>;
      using groupingMetadata = typename aod::MetadataTrait<O>::metadata;
      return std::string{groupingMetadata::tableLabel()};
    } else {
      using groupingMetadata = typename aod::MetadataTrait<TT>::metadata;
      return std::string{groupingMetadata::tableLabel()};
    }
  } else if constexpr (soa::is_type_with_originals_v<std::decay_t<T>>) {
    using TT = typename framework::pack_head_t<typename std::decay_t<T>::originals>;
    if constexpr (soa::is_with_base_table_v<typename aod::MetadataTrait<TT>::metadata>) {
      using TTT = typename aod::MetadataTrait<TT>::metadata::base_table_t;
      return getLabelFromType<TTT>();
    } else {
      using groupingMetadata = typename aod::MetadataTrait<TT>::metadata;
      return std::string{groupingMetadata::tableLabel()};
    }
  } else {
    if constexpr (soa::is_with_base_table_v<typename aod::MetadataTrait<T>::metadata>) {
      using TT = typename aod::MetadataTrait<T>::metadata::base_table_t;
      return getLabelFromType<TT>();
    } else {
      using groupingMetadata = typename aod::MetadataTrait<std::decay_t<T>>::metadata;
      return std::string{groupingMetadata::tableLabel()};
    }
  }
}

template <typename... C>
static constexpr auto hasColumnForKey(framework::pack<C...>, std::string const& key)
{
  return ((C::inherited_t::mLabel == key) || ...);
}

template <typename T>
static constexpr std::pair<bool, std::string> hasKey(std::string const& key)
{
  return {hasColumnForKey(typename T::persistent_columns_t{}, key), getLabelFromType<T>()};
}

template <typename... C>
static constexpr auto haveKey(framework::pack<C...>, std::string const& key)
{
  return std::vector{hasKey<C>(key)...};
}

void notFoundColumn(const char* label, const char* key);
void missingOptionalPreslice(const char* label, const char* key);

template <typename T, bool OPT = false>
static constexpr std::string getLabelFromTypeForKey(std::string const& key)
{
  if constexpr (soa::is_type_with_originals_v<std::decay_t<T>>) {
    using Os = typename std::decay_t<T>::originals;
    auto locate = haveKey(Os{}, key);
    auto it = std::find_if(locate.begin(), locate.end(), [](auto const& x) { return x.first; });
    if (it != locate.end()) {
      return it->second;
    }
  } else {
    auto locate = hasKey<std::decay_t<T>>(key);
    if (locate.first) {
      return locate.second;
    }
  }
  if constexpr (!OPT) {
    notFoundColumn(getLabelFromType<std::decay_t<T>>().data(), key.data());
  } else {
    return "[MISSING]";
  }
  O2_BUILTIN_UNREACHABLE();
}

template <typename B, typename... C>
consteval static bool hasIndexTo(framework::pack<C...>&&)
{
  return (o2::soa::is_binding_compatible_v<B, typename C::binding_t>() || ...);
}

template <typename B, typename... C>
consteval static bool hasSortedIndexTo(framework::pack<C...>&&)
{
  return ((C::sorted && o2::soa::is_binding_compatible_v<B, typename C::binding_t>()) || ...);
}

template <typename B, typename Z>
consteval static bool relatedByIndex()
{
  return hasIndexTo<B>(typename Z::table_t::external_index_columns_t{});
}

template <typename B, typename Z>
consteval static bool relatedBySortedIndex()
{
  return hasSortedIndexTo<B>(typename Z::table_t::external_index_columns_t{});
}
} // namespace o2::soa

namespace o2::framework
{
template <typename T, bool OPT = false, bool SORTED = true>
struct PresliceBase {
  constexpr static bool sorted = SORTED;
  constexpr static bool optional = OPT;
  using target_t = T;
  const std::string binding;

  PresliceBase(expressions::BindingNode index_)
    : binding{o2::soa::getLabelFromTypeForKey<T, OPT>(index_.name)},
      bindingKey{binding, index_.name} {}

  void updateSliceInfo(std::conditional_t<SORTED, SliceInfoPtr, SliceInfoUnsortedPtr>&& si)
  {
    sliceInfo = si;
  }

  std::shared_ptr<arrow::Table> getSliceFor(int value, std::shared_ptr<arrow::Table> const& input, uint64_t& offset) const
  {
    if constexpr (OPT) {
      if (isMissing()) {
        return nullptr;
      }
    }
    if constexpr (SORTED) {
      auto [offset_, count] = sliceInfo.getSliceFor(value);
      auto output = input->Slice(offset_, count);
      offset = static_cast<int64_t>(offset_);
      return output;
    } else {
      static_assert(SORTED, "Wrong method called for unsorted cache");
    }
  }

  gsl::span<const int64_t> getSliceFor(int value) const
  {
    if constexpr (OPT) {
      if (isMissing()) {
        return {};
      }
    }
    if constexpr (!SORTED) {
      return sliceInfo.getSliceFor(value);
    } else {
      static_assert(!SORTED, "Wrong method called for sorted cache");
    }
  }

  bool isMissing() const
  {
    return binding == "[MISSING]";
  }

  StringPair const& getBindingKey() const
  {
    return bindingKey;
  }

  std::conditional_t<SORTED, SliceInfoPtr, SliceInfoUnsortedPtr> sliceInfo;

  StringPair bindingKey;
};

template <typename T>
using PresliceUnsorted = PresliceBase<T, false, false>;
template <typename T>
using PresliceUnsortedOptional = PresliceBase<T, true, false>;
template <typename T>
using Preslice = PresliceBase<T, false, true>;
template <typename T>
using PresliceOptional = PresliceBase<T, true, true>;

} // namespace o2::framework

namespace o2::soa
{
//! Helper to check if a type T is an iterator
template <typename T>
inline constexpr bool is_soa_iterator_v = framework::is_base_of_template_v<RowViewCore, T> || framework::is_specialization_v<T, RowViewCore>;

template <typename T>
inline consteval bool is_soa_filtered_iterator_v()
{
  if constexpr (!is_soa_iterator_v<T>) {
    return false;
  } else {
    if constexpr (std::is_same_v<typename T::policy_t, soa::FilteredIndexPolicy>) {
      return true;
    } else {
      return false;
    }
  }
}

template <typename T>
using is_soa_table_t = typename framework::is_specialization<T, soa::Table>;

template <typename T>
inline constexpr bool is_soa_table_like_v = framework::is_base_of_template_v<soa::Table, T>;

template <typename T>
class FilteredBase;
template <typename T>
class Filtered;

template <typename T>
inline constexpr bool is_soa_filtered_v = framework::is_base_of_template_v<soa::FilteredBase, T>;

/// Helper function to extract bound indices
template <typename... Is>
static consteval auto extractBindings(framework::pack<Is...>)
{
  return framework::pack<typename Is::binding_t...>{};
}

SelectionVector selectionToVector(gandiva::Selection const& sel);

template <typename T, typename C, bool OPT, bool SORTED>
auto doSliceBy(T const* table, o2::framework::PresliceBase<C, OPT, SORTED> const& container, int value)
{
  if constexpr (o2::soa::is_binding_compatible_v<C, T>()) {
    if constexpr (OPT) {
      if (container.isMissing()) {
        missingOptionalPreslice(getLabelFromType<std::decay_t<T>>().data(), container.bindingKey.second.c_str());
      }
    }
    if constexpr (SORTED) {
      uint64_t offset = 0;
      auto out = container.getSliceFor(value, table->asArrowTable(), offset);
      auto t = typename T::self_t({out}, offset);
      table->copyIndexBindings(t);
      t.bindInternalIndicesTo(table);
      return t;
    } else {
      auto selection = container.getSliceFor(value);
      if constexpr (soa::is_soa_filtered_v<T>) {
        auto t = soa::Filtered<typename T::base_t>({table->asArrowTable()}, selection);
        table->copyIndexBindings(t);
        t.bindInternalIndicesTo(table);
        t.intersectWithSelection(table->getSelectedRows()); // intersect filters
        return t;
      } else {
        auto t = soa::Filtered<T>({table->asArrowTable()}, selection);
        table->copyIndexBindings(t);
        t.bindInternalIndicesTo(table);
        return t;
      }
    }
  } else {
    if constexpr (SORTED) {
      static_assert(o2::framework::always_static_assert_v<C>, "Wrong Preslice<> entry used: incompatible type");
    } else {
      static_assert(o2::framework::always_static_assert_v<C>, "Wrong PresliceUnsorted<> entry used: incompatible type");
    }
  }
}

template <typename T>
auto prepareFilteredSlice(T const* table, std::shared_ptr<arrow::Table> slice, uint64_t offset)
{
  if (offset >= table->tableSize()) {
    if constexpr (soa::is_soa_filtered_v<T>) {
      Filtered<typename T::base_t> fresult{{{slice}}, SelectionVector{}, 0};
      table->copyIndexBindings(fresult);
      return fresult;
    } else {
      typename T::self_t fresult{{{slice}}, SelectionVector{}, 0};
      table->copyIndexBindings(fresult);
      return fresult;
    }
  }
  auto start = offset;
  auto end = start + slice->num_rows();
  auto mSelectedRows = table->getSelectedRows();
  auto start_iterator = std::lower_bound(mSelectedRows.begin(), mSelectedRows.end(), start);
  auto stop_iterator = std::lower_bound(start_iterator, mSelectedRows.end(), end);
  SelectionVector slicedSelection{start_iterator, stop_iterator};
  std::transform(slicedSelection.begin(), slicedSelection.end(), slicedSelection.begin(),
                 [&start](int64_t idx) {
                   return idx - static_cast<int64_t>(start);
                 });
  if constexpr (soa::is_soa_filtered_v<T>) {
    Filtered<typename T::base_t> fresult{{{slice}}, std::move(slicedSelection), start};
    table->copyIndexBindings(fresult);
    return fresult;
  } else {
    typename T::self_t fresult{{{slice}}, std::move(slicedSelection), start};
    table->copyIndexBindings(fresult);
    return fresult;
  }
}

template <typename T, typename C, bool OPT>
auto doFilteredSliceBy(T const* table, o2::framework::PresliceBase<C, OPT> const& container, int value)
{
  if constexpr (o2::soa::is_binding_compatible_v<C, T>()) {
    if constexpr (OPT) {
      if (container.isMissing()) {
        missingOptionalPreslice(getLabelFromType<T>().data(), container.bindingKey.second.c_str());
      }
    }
    uint64_t offset = 0;
    auto slice = container.getSliceFor(value, table->asArrowTable(), offset);
    return prepareFilteredSlice(table, slice, offset);
  } else {
    static_assert(o2::framework::always_static_assert_v<C>, "Wrong Preslice<> entry used: incompatible type");
  }
}

template <typename T>
auto doSliceByCached(T const* table, framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache)
{
  auto localCache = cache.ptr->getCacheFor({o2::soa::getLabelFromTypeForKey<T>(node.name), node.name});
  auto [offset, count] = localCache.getSliceFor(value);
  auto t = typename T::self_t({table->asArrowTable()->Slice(static_cast<uint64_t>(offset), count)}, static_cast<uint64_t>(offset));
  table->copyIndexBindings(t);
  return t;
}

template <typename T>
auto doFilteredSliceByCached(T const* table, framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache)
{
  auto localCache = cache.ptr->getCacheFor({o2::soa::getLabelFromTypeForKey<T>(node.name), node.name});
  auto [offset, count] = localCache.getSliceFor(value);
  auto slice = table->asArrowTable()->Slice(static_cast<uint64_t>(offset), count);
  return prepareFilteredSlice(table, slice, offset);
}

template <typename T>
auto doSliceByCachedUnsorted(T const* table, framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache)
{
  auto localCache = cache.ptr->getCacheUnsortedFor({o2::soa::getLabelFromTypeForKey<T>(node.name), node.name});
  if constexpr (soa::is_soa_filtered_v<T>) {
    auto t = typename T::self_t({table->asArrowTable()}, localCache.getSliceFor(value));
    t.intersectWithSelection(table->getSelectedRows());
    table->copyIndexBindings(t);
    return t;
  } else {
    auto t = Filtered<T>({table->asArrowTable()}, localCache.getSliceFor(value));
    table->copyIndexBindings(t);
    return t;
  }
}

template <typename T>
auto select(T const& t, framework::expressions::Filter const& f)
{
  return Filtered<T>({t.asArrowTable()}, selectionToVector(framework::expressions::createSelection(t.asArrowTable(), f)));
}

arrow::ChunkedArray* getIndexFromLabel(arrow::Table* table, const char* label);

/// A Table class which observes an arrow::Table and provides
/// It is templated on a set of Column / DynamicColumn types.
template <typename... C>
class Table
{
 public:
  using self_t = Table<C...>;
  using table_t = Table<C...>;
  using columns = framework::pack<C...>;
  using column_types = framework::pack<typename C::type...>;
  using persistent_columns_t = framework::selected_pack<is_persistent_t, C...>;
  using external_index_columns_t = framework::selected_pack<is_external_index_t, C...>;
  using internal_index_columns_t = framework::selected_pack<is_self_index_t, C...>;

  static constexpr auto hashes()
  {
    return std::set{{o2::framework::TypeIdHelpers::uniqueId<C>()...}};
  }

  template <typename IP, typename Parent, typename... T>
  struct RowViewBase : public RowViewCore<IP, C...> {

    using external_index_columns_t = framework::selected_pack<is_external_index_t, C...>;
    using bindings_pack_t = decltype(extractBindings(external_index_columns_t{}));
    using parent_t = Parent;
    using originals = originals_pack_t<T...>;
    using policy_t = IP;

    RowViewBase() = default;

    RowViewBase(arrow::ChunkedArray* columnData[sizeof...(C)], IP&& policy)
      : RowViewCore<IP, C...>(columnData, std::forward<decltype(policy)>(policy))
    {
    }

    template <typename P, typename... O>
    RowViewBase& operator=(RowViewBase<IP, P, O...> other) requires std::is_same_v<typename P::table_t, typename Parent::table_t>
    {
      static_cast<RowViewCore<IP, C...>&>(*this) = static_cast<RowViewCore<IP, C...>>(other);
      return *this;
    }

    template <typename P>
    RowViewBase& operator=(RowViewBase<IP, P, T...> other)
    {
      static_cast<RowViewCore<IP, C...>&>(*this) = static_cast<RowViewCore<IP, C...>>(other);
      return *this;
    }

    template <typename P>
    RowViewBase& operator=(RowViewBase<FilteredIndexPolicy, P, T...> other) requires std::is_same_v<IP, DefaultIndexPolicy>
    {
      static_cast<RowViewCore<IP, C...>&>(*this) = static_cast<RowViewCore<FilteredIndexPolicy, C...>>(other);
      return *this;
    }

    template <typename P, typename... O>
    RowViewBase(RowViewBase<IP, P, O...> const& other) requires std::is_same_v<typename P::table_t, typename Parent::table_t>
    {
      *this = other;
    }

    template <typename P, typename... O>
    RowViewBase(RowViewBase<IP, P, O...>&& other) noexcept requires std::is_same_v<typename P::table_t, typename Parent::table_t>
    {
      *this = other;
    }

    template <typename P>
    RowViewBase(RowViewBase<IP, P, T...> const& other)
    {
      *this = other;
    }

    template <typename P>
    RowViewBase(RowViewBase<IP, P, T...>&& other) noexcept
    {
      *this = other;
    }

    template <typename P>
    RowViewBase(RowViewBase<FilteredIndexPolicy, P, T...> other) requires std::is_same_v<IP, DefaultIndexPolicy>
    {
      *this = other;
    }

    RowViewBase& operator=(RowViewSentinel const& other)
    {
      this->mRowIndex = other.index;
      return *this;
    }

    template <typename P>
    void matchTo(RowViewBase<IP, P, T...> const& other)
    {
      this->mRowIndex = other.mRowIndex;
    }

    template <typename P, typename... O>
    void matchTo(RowViewBase<IP, P, O...> const& other) requires std::is_same_v<typename P::table_t, typename Parent::table_t>
    {
      this->mRowIndex = other.mRowIndex;
    }

    template <typename TI>
    auto getId() const
    {
      using decayed = std::decay_t<TI>;
      if constexpr (framework::has_type<decayed>(bindings_pack_t{})) { // index to another table
        constexpr auto idx = framework::has_type_at_v<decayed>(bindings_pack_t{});
        return framework::pack_element_t<idx, external_index_columns_t>::getId();
      } else if constexpr (std::is_same_v<decayed, Parent>) { // self index
        return this->globalIndex();
      } else if constexpr (is_index_t<decayed>::value && decayed::mLabel == "Index") { // soa::Index<>
        return this->globalIndex();
      } else {
        return static_cast<int32_t>(-1);
      }
    }

    template <typename CD, typename... CDArgs>
    auto getDynamicColumn() const
    {
      using decayed = std::decay_t<CD>;
      static_assert(is_dynamic_t<decayed>(), "Requested column is not a dynamic column");
      return static_cast<decayed>(*this).template getDynamicValue<CDArgs...>();
    }

    template <typename B, typename CC>
    auto getValue() const
    {
      using COL = std::decay_t<CC>;
      static_assert(is_dynamic_t<COL>() || is_persistent_v<COL>, "Should be persistent or dynamic column with no argument that has a return type convertable to float");
      return static_cast<B>(static_cast<COL>(*this).get());
    }

    template <typename B, typename... CCs>
    std::array<B, sizeof...(CCs)> getValues() const
    {
      static_assert(std::is_same_v<B, float> || std::is_same_v<B, double>, "The common return type should be float or double");
      return {getValue<B, CCs>()...};
    }

    using IP::size;

    using RowViewCore<IP, C...>::operator++;

    /// Allow incrementing by more than one the iterator
    RowViewBase operator+(int64_t inc) const
    {
      RowViewBase copy = *this;
      copy.moveByIndex(inc);
      return copy;
    }

    RowViewBase operator-(int64_t dec) const
    {
      return operator+(-dec);
    }

    RowViewBase const& operator*() const
    {
      return *this;
    }
  };
  template <typename P, typename... Ts>
  using RowView = RowViewBase<DefaultIndexPolicy, P, Ts...>;

  template <typename P, typename... Ts>
  using RowViewFiltered = RowViewBase<FilteredIndexPolicy, P, Ts...>;

  using iterator = RowView<table_t, table_t>;
  using const_iterator = RowView<table_t, table_t>;
  using unfiltered_iterator = RowView<table_t, table_t>;
  using unfiltered_const_iterator = RowView<table_t, table_t>;
  using filtered_iterator = RowViewFiltered<table_t, table_t>;
  using filtered_const_iterator = RowViewFiltered<table_t, table_t>;

  Table(std::shared_ptr<arrow::Table> table, uint64_t offset = 0)
    : mTable(table),
      mEnd{table->num_rows()},
      mOffset(offset)
  {
    if (mTable->num_rows() == 0) {
      for (size_t ci = 0; ci < sizeof...(C); ++ci) {
        mColumnChunks[ci] = nullptr;
      }
      mBegin = mEnd;
    } else {
      arrow::ChunkedArray* lookups[] = {lookupColumn<C>()...};
      for (size_t ci = 0; ci < sizeof...(C); ++ci) {
        mColumnChunks[ci] = lookups[ci];
      }
      mBegin = unfiltered_iterator{mColumnChunks, {table->num_rows(), offset}};
      mBegin.bindInternalIndices(this);
    }
  }

  /// FIXME: this is to be able to construct a Filtered without explicit Join
  ///        so that Filtered<Table1,Table2, ...> always means a Join which
  ///        may or may not be a problem later
  Table(std::vector<std::shared_ptr<arrow::Table>>&& tables, uint64_t offset = 0)
    : Table(ArrowHelpers::joinTables(std::move(tables)), offset)
  {
  }

  template <typename Key>
  inline arrow::ChunkedArray* getIndexToKey()
  {
    if constexpr (framework::has_type_conditional<is_binding_compatible, Key>(external_index_columns_t{})) {
      using IC = framework::pack_element_t<framework::has_type_at_conditional<is_binding_compatible, Key>(external_index_columns_t{}), external_index_columns_t>;
      return mColumnChunks[framework::has_type_at<IC>(persistent_columns_t{})];
    } else if constexpr (std::is_same_v<table_t, Key>) {
      return nullptr;
    } else {
      static_assert(framework::always_static_assert_v<Key>, "This table does not have an index to this type");
    }
  }

  auto& cached_begin()
  {
    return mBegin;
  }

  auto const& cached_begin() const
  {
    return mBegin;
  }

  unfiltered_iterator begin()
  {
    return unfiltered_iterator(mBegin);
  }

  RowViewSentinel end()
  {
    return RowViewSentinel{mEnd};
  }

  filtered_iterator filtered_begin(gsl::span<int64_t const> selection)
  {
    // Note that the FilteredIndexPolicy will never outlive the selection which
    // is held by the table, so we are safe passing the bare pointer. If it does it
    // means that the iterator on a table is outliving the table itself, which is
    // a bad idea.
    return filtered_iterator(mColumnChunks, {selection, mTable->num_rows(), mOffset});
  }

  iterator iteratorAt(uint64_t i) const
  {
    return rawIteratorAt(i);
  }

  unfiltered_iterator rawIteratorAt(uint64_t i) const
  {
    auto it = mBegin;
    it.setCursor(i);
    return it;
  }

  unfiltered_const_iterator begin() const
  {
    return unfiltered_const_iterator(mBegin);
  }

  [[nodiscard]] RowViewSentinel end() const
  {
    return RowViewSentinel{mEnd};
  }

  /// Return a type erased arrow table backing store for / the type safe table.
  [[nodiscard]] std::shared_ptr<arrow::Table> asArrowTable() const
  {
    return mTable;
  }
  /// Return offset
  auto offset() const
  {
    return mOffset;
  }
  /// Size of the table, in rows.
  [[nodiscard]] int64_t size() const
  {
    return mTable->num_rows();
  }

  [[nodiscard]] int64_t tableSize() const
  {
    return size();
  }

  /// Bind the columns which refer to other tables
  /// to the associated tables.
  template <typename... TA>
  void bindExternalIndices(TA*... current)
  {
    mBegin.bindExternalIndices(current...);
  }

  template <typename I>
  void bindInternalIndicesTo(I const* ptr)
  {
    mBegin.bindInternalIndices(ptr);
  }

  void bindInternalIndicesExplicit(o2::soa::Binding binding)
  {
    doBindInternalIndicesExplicit(internal_index_columns_t{}, binding);
  }

  template <typename... Cs>
  void doBindInternalIndicesExplicit(framework::pack<Cs...>, o2::soa::Binding binding)
  {
    (static_cast<Cs>(mBegin).setCurrentRaw(binding), ...);
  }

  void bindExternalIndicesRaw(std::vector<o2::soa::Binding>&& ptrs)
  {
    mBegin.bindExternalIndicesRaw(std::forward<std::vector<o2::soa::Binding>>(ptrs));
  }

  template <typename T, typename... Cs>
  void doCopyIndexBindings(framework::pack<Cs...>, T& dest) const
  {
    dest.bindExternalIndicesRaw(mBegin.getIndexBindings());
  }

  template <typename T>
  void copyIndexBindings(T& dest) const
  {
    doCopyIndexBindings(external_index_columns_t{}, dest);
  }

  auto select(framework::expressions::Filter const& f) const
  {
    auto t = o2::soa::select(*this, f);
    copyIndexBindings(t);
    return t;
  }

  auto sliceByCached(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return doSliceByCached(this, node, value, cache);
  }

  auto sliceByCachedUnsorted(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return doSliceByCachedUnsorted(this, node, value, cache);
  }

  template <typename T1, bool OPT, bool SORTED>
  auto sliceBy(o2::framework::PresliceBase<T1, OPT, SORTED> const& container, int value) const
  {
    return doSliceBy(this, container, value);
  }

  auto rawSlice(uint64_t start, uint64_t end) const
  {
    return self_t{mTable->Slice(start, end - start + 1), start};
  }

  auto emptySlice() const
  {
    return self_t{mTable->Slice(0, 0), 0};
  }

 protected:
  /// Offset of the table within a larger table.
  uint64_t mOffset;

 private:
  template <typename T>
  arrow::ChunkedArray* lookupColumn()
  {
    if constexpr (T::persistent::value) {
      auto label = T::columnLabel();
      return getIndexFromLabel(mTable.get(), label);
    } else {
      return nullptr;
    }
  }
  std::shared_ptr<arrow::Table> mTable;
  // Cached pointers to the ChunkedArray associated to a column
  arrow::ChunkedArray* mColumnChunks[sizeof...(C)];
  /// Cached begin iterator for this table.
  unfiltered_iterator mBegin;
  /// Cached end iterator for this table.
  RowViewSentinel mEnd;
};

template <typename T>
struct PackToTable {
  static_assert(framework::always_static_assert_v<T>, "Not a pack");
};

template <typename... C>
struct PackToTable<framework::pack<C...>> {
  using table = o2::soa::Table<C...>;
};

template <typename... T>
struct TableWrap {
  using all_columns = framework::concatenated_pack_unique_t<typename T::columns...>;
  using table_t = typename PackToTable<all_columns>::table;
};

template <typename... T>
struct TableIntersect {
  using all_columns = framework::full_intersected_pack_t<typename T::columns...>;
  using table_t = typename PackToTable<all_columns>::table;
};

/// Template trait which allows to map a given
/// Table type to its O2 DataModel origin and description
template <typename INHERIT>
class TableMetadata
{
 public:
  static constexpr char const* tableLabel() { return INHERIT::mLabel; }
  static constexpr char const (&origin())[5] { return INHERIT::mOrigin; }
  static constexpr char const (&description())[16] { return INHERIT::mDescription; }
  static constexpr o2::header::DataHeader::SubSpecificationType version() { return INHERIT::mVersion; }
  static std::string sourceSpec() { return fmt::format("{}/{}/{}/{}", INHERIT::mLabel, INHERIT::mOrigin, INHERIT::mDescription, INHERIT::mVersion); };
};

/// Helper templates to define universal join and concat
template <typename... T>
constexpr auto join(T const&... t)
{
  return typename o2::soa::TableWrap<T...>::table_t(ArrowHelpers::joinTables({t.asArrowTable()...}));
}

template <typename... T>
constexpr auto concat(T const&... t)
{
  return typename o2::soa::TableIntersect<T...>::table_t(ArrowHelpers::concatTables({t.asArrowTable()...}));
}

template <typename T1, typename T2>
using ConcatBase = decltype(concat(std::declval<T1>(), std::declval<T2>()));

void notBoundTable(const char* tableName);

namespace row_helpers
{
template <typename... Cs>
std::array<arrow::ChunkedArray*, sizeof...(Cs)> getArrowColumns(arrow::Table* table, framework::pack<Cs...>)
{
  static_assert(std::conjunction_v<typename Cs::persistent...>, "Arrow columns: only persistent columns accepted (not dynamic and not index ones");
  return std::array<arrow::ChunkedArray*, sizeof...(Cs)>{o2::soa::getIndexFromLabel(table, Cs::columnLabel())...};
}

template <typename... Cs>
std::array<std::shared_ptr<arrow::Array>, sizeof...(Cs)> getChunks(arrow::Table* table, framework::pack<Cs...>, uint64_t ci)
{
  static_assert(std::conjunction_v<typename Cs::persistent...>, "Arrow chunks: only persistent columns accepted (not dynamic and not index ones");
  return std::array<std::shared_ptr<arrow::Array>, sizeof...(Cs)>{o2::soa::getIndexFromLabel(table, Cs::columnLabel())->chunk(ci)...};
}

template <typename T, typename C>
typename C::type getSingleRowPersistentData(arrow::Table* table, T& rowIterator, uint64_t ci = std::numeric_limits<uint64_t>::max(), uint64_t ai = std::numeric_limits<uint64_t>::max())
{
  if (ci == std::numeric_limits<uint64_t>::max() || ai == std::numeric_limits<uint64_t>::max()) {
    auto colIterator = static_cast<C>(rowIterator).getIterator();
    ci = colIterator.mCurrentChunk;
    ai = *(colIterator.mCurrentPos) - colIterator.mFirstIndex;
  }
  return std::static_pointer_cast<o2::soa::arrow_array_for_t<typename C::type>>(o2::soa::getIndexFromLabel(table, C::columnLabel())->chunk(ci))->raw_values()[ai];
}

template <typename T, typename C>
typename C::type getSingleRowDynamicData(T& rowIterator, uint64_t globalIndex = std::numeric_limits<uint64_t>::max())
{
  if (globalIndex != std::numeric_limits<uint64_t>::max() && globalIndex != *std::get<0>(rowIterator.getIndices())) {
    rowIterator.setCursor(globalIndex);
  }
  return rowIterator.template getDynamicColumn<C>();
}

template <typename T, typename C>
typename C::type getSingleRowIndexData(T& rowIterator, uint64_t globalIndex = std::numeric_limits<uint64_t>::max())
{
  if (globalIndex != std::numeric_limits<uint64_t>::max() && globalIndex != *std::get<0>(rowIterator.getIndices())) {
    rowIterator.setCursor(globalIndex);
  }
  return rowIterator.template getId<C>();
}

template <typename T, typename C>
typename C::type getSingleRowData(arrow::Table* table, T& rowIterator, uint64_t ci = -1, uint64_t ai = std::numeric_limits<uint64_t>::max(), uint64_t globalIndex = std::numeric_limits<uint64_t>::max())
{
  using decayed = std::decay_t<C>;
  if constexpr (decayed::persistent::value) {
    return getSingleRowPersistentData<T, C>(table, rowIterator, ci, ai);
  } else if constexpr (o2::soa::is_dynamic_t<decayed>()) {
    return getSingleRowDynamicData<T, C>(rowIterator, globalIndex);
  } else if constexpr (o2::soa::is_index_t<decayed>::value) {
    return getSingleRowIndexData<T, C>(rowIterator, globalIndex);
  } else {
    static_assert(!sizeof(decayed*), "Unrecognized column kind"); // A trick to delay static_assert until we actually instantiate this branch
  }
}

template <typename T, typename... Cs>
std::tuple<typename Cs::type...> getRowData(arrow::Table* table, T rowIterator, uint64_t ci = std::numeric_limits<uint64_t>::max(), uint64_t ai = std::numeric_limits<uint64_t>::max(), uint64_t globalIndex = std::numeric_limits<uint64_t>::max())
{
  return std::make_tuple(getSingleRowData<T, Cs>(table, rowIterator, ci, ai, globalIndex)...);
}
} // namespace row_helpers
} // namespace o2::soa

#define DECLARE_SOA_VERSIONING()                                                                    \
  template <typename T>                                                                             \
  constexpr int getVersion()                                                                        \
  {                                                                                                 \
    if constexpr (o2::soa::is_type_with_metadata_v<MetadataTrait<T>>) {                             \
      return MetadataTrait<T>::metadata::version();                                                 \
    } else if constexpr (o2::soa::is_type_with_originals_v<T>) {                                    \
      return MetadataTrait<o2::framework::pack_head_t<typename T::originals>>::metadata::version(); \
    } else {                                                                                        \
      static_assert(o2::framework::always_static_assert_v<T>, "Not a versioned type");              \
    }                                                                                               \
  }

#define DECLARE_EQUIVALENT_FOR_INDEX(_Base_, _Equiv_) \
  template <>                                         \
  struct EquivalentIndex<_Base_, _Equiv_> {           \
    constexpr static bool value = true;               \
  }

#define DECLARE_SOA_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_)                                                                                                                \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {                                                                                                                               \
    static constexpr const char* mLabel = _Label_;                                                                                                                                \
    static_assert(!((*(mLabel + 1) == 'I' && *(mLabel + 2) == 'n' && *(mLabel + 3) == 'd' && *(mLabel + 4) == 'e' && *(mLabel + 5) == 'x')), "Index is not a valid column name"); \
    using base = o2::soa::Column<_Type_, _Name_>;                                                                                                                                 \
    using type = _Type_;                                                                                                                                                          \
    using column_t = _Name_;                                                                                                                                                      \
    _Name_(arrow::ChunkedArray const* column)                                                                                                                                     \
      : o2::soa::Column<_Type_, _Name_>(o2::soa::ColumnIterator<type>(column))                                                                                                    \
    {                                                                                                                                                                             \
    }                                                                                                                                                                             \
                                                                                                                                                                                  \
    _Name_() = default;                                                                                                                                                           \
    _Name_(_Name_ const& other) = default;                                                                                                                                        \
    _Name_& operator=(_Name_ const& other) = default;                                                                                                                             \
                                                                                                                                                                                  \
    decltype(auto) _Getter_() const                                                                                                                                               \
    {                                                                                                                                                                             \
      return *mColumnIterator;                                                                                                                                                    \
    }                                                                                                                                                                             \
                                                                                                                                                                                  \
    decltype(auto) get() const                                                                                                                                                    \
    {                                                                                                                                                                             \
      return _Getter_();                                                                                                                                                          \
    }                                                                                                                                                                             \
  };                                                                                                                                                                              \
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_ { _Label_, o2::framework::TypeIdHelpers::uniqueId<_Name_>(),                                 \
                                                                                       o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_COLUMN(_Name_, _Getter_, _Type_) \
  DECLARE_SOA_COLUMN_FULL(_Name_, _Getter_, _Type_, "f" #_Name_)

/// A 'bitmap' column, i.e. a int-based column with custom accessors to check
/// individual bits
#define MAKEINT(_Size_) uint##_Size_##_t

#define DECLARE_SOA_BITMAP_COLUMN_FULL(_Name_, _Getter_, _Size_, _Label_)                                                                                                         \
  struct _Name_ : o2::soa::Column<MAKEINT(_Size_), _Name_> {                                                                                                                      \
    static constexpr const char* mLabel = _Label_;                                                                                                                                \
    static_assert(!((*(mLabel + 1) == 'I' && *(mLabel + 2) == 'n' && *(mLabel + 3) == 'd' && *(mLabel + 4) == 'e' && *(mLabel + 5) == 'x')), "Index is not a valid column name"); \
    using base = o2::soa::Column<MAKEINT(_Size_), _Name_>;                                                                                                                        \
    using type = MAKEINT(_Size_);                                                                                                                                                 \
    _Name_(arrow::ChunkedArray const* column)                                                                                                                                     \
      : o2::soa::Column<type, _Name_>(o2::soa::ColumnIterator<type>(column))                                                                                                      \
    {                                                                                                                                                                             \
    }                                                                                                                                                                             \
                                                                                                                                                                                  \
    _Name_() = default;                                                                                                                                                           \
    _Name_(_Name_ const& other) = default;                                                                                                                                        \
    _Name_& operator=(_Name_ const& other) = default;                                                                                                                             \
                                                                                                                                                                                  \
    decltype(auto) _Getter_##_raw() const                                                                                                                                         \
    {                                                                                                                                                                             \
      return *mColumnIterator;                                                                                                                                                    \
    }                                                                                                                                                                             \
                                                                                                                                                                                  \
    bool _Getter_##_bit(int bit) const                                                                                                                                            \
    {                                                                                                                                                                             \
      return (*mColumnIterator & (static_cast<type>(1) << bit)) >> bit;                                                                                                           \
    }                                                                                                                                                                             \
  };                                                                                                                                                                              \
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_ { _Label_, o2::framework::TypeIdHelpers::uniqueId<_Name_>(),                                 \
                                                                                       o2::framework::expressions::selectArrowType<MAKEINT(_Size_)>() }

#define DECLARE_SOA_BITMAP_COLUMN(_Name_, _Getter_, _Size_) \
  DECLARE_SOA_BITMAP_COLUMN_FULL(_Name_, _Getter_, _Size_, "f" #_Name_)

/// An 'expression' column. i.e. a column that can be calculated from other
/// columns with gandiva based on supplied C++ expression.
#define DECLARE_SOA_EXPRESSION_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_, _Expression_)                                                       \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {                                                                                               \
    static constexpr const char* mLabel = _Label_;                                                                                                \
    using base = o2::soa::Column<_Type_, _Name_>;                                                                                                 \
    using type = _Type_;                                                                                                                          \
    using column_t = _Name_;                                                                                                                      \
    using spawnable_t = std::true_type;                                                                                                           \
    _Name_(arrow::ChunkedArray const* column)                                                                                                     \
      : o2::soa::Column<_Type_, _Name_>(o2::soa::ColumnIterator<type>(column))                                                                    \
    {                                                                                                                                             \
    }                                                                                                                                             \
                                                                                                                                                  \
    _Name_() = default;                                                                                                                           \
    _Name_(_Name_ const& other) = default;                                                                                                        \
    _Name_& operator=(_Name_ const& other) = default;                                                                                             \
                                                                                                                                                  \
    decltype(auto) _Getter_() const                                                                                                               \
    {                                                                                                                                             \
      return *mColumnIterator;                                                                                                                    \
    }                                                                                                                                             \
    static o2::framework::expressions::Projector Projector()                                                                                      \
    {                                                                                                                                             \
      return _Expression_;                                                                                                                        \
    }                                                                                                                                             \
  };                                                                                                                                              \
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_ { _Label_, o2::framework::TypeIdHelpers::uniqueId<_Name_>(), \
                                                                                       o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_EXPRESSION_COLUMN(_Name_, _Getter_, _Type_, _Expression_) \
  DECLARE_SOA_EXPRESSION_COLUMN_FULL(_Name_, _Getter_, _Type_, "f" #_Name_, _Expression_);

/// An index column is a column of indices to elements / of another table named
/// _Name_##s. The column name will be _Name_##Id and will always be stored in
/// "fIndex"#_Table_#[_Suffix_]. If _Suffix_ is not empty it has to begin
/// with _ (underscore) to make the columns identifiable for the table merging
/// It will also have two special methods, setCurrent(...)
/// and getCurrent(...) which allow you to set / retrieve associated table.
/// It also exposes a getter _Getter_ which allows you to retrieve the pointed
/// object.
/// Notice how in order to define an index column, the table it points
/// to **must** be already declared. This is therefore only
/// useful to express child -> parent relationships. In case one
/// needs to go from parent to child, the only way is to either have
/// a separate "association" with the two indices, or to use the standard
/// grouping mechanism of AnalysisTask.
///
/// Normal index: returns iterator to a bound table
/// Slice  index: return an instance of the bound table type with a slice defined by the values in 0 and 1st elements
/// Array  index: return an array of iterators, defined by values in its elements

/// SLICE
#define DECLARE_SOA_SLICE_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Table_, _Suffix_)    \
  struct _Name_##IdSlice : o2::soa::Column<_Type_[2], _Name_##IdSlice> {                    \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");               \
    static_assert((*_Suffix_ == '\0') || (*_Suffix_ == '_'), "Suffix has to begin with _"); \
    static constexpr const char* mLabel = "fIndexSlice" #_Table_ _Suffix_;                  \
    using base = o2::soa::Column<_Type_[2], _Name_##IdSlice>;                               \
    using type = _Type_[2];                                                                 \
    using column_t = _Name_##IdSlice;                                                       \
    using binding_t = _Table_;                                                              \
    _Name_##IdSlice(arrow::ChunkedArray const* column)                                      \
      : o2::soa::Column<_Type_[2], _Name_##IdSlice>(o2::soa::ColumnIterator<type>(column))  \
    {                                                                                       \
    }                                                                                       \
                                                                                            \
    _Name_##IdSlice() = default;                                                            \
    _Name_##IdSlice(_Name_##IdSlice const& other) = default;                                \
    _Name_##IdSlice& operator=(_Name_##IdSlice const& other) = default;                     \
    std::array<_Type_, 2> inline getIds() const                                             \
    {                                                                                       \
      return _Getter_##Ids();                                                               \
    }                                                                                       \
                                                                                            \
    bool has_##_Getter_() const                                                             \
    {                                                                                       \
      auto a = *mColumnIterator;                                                            \
      return a[0] >= 0 && a[1] >= 0;                                                        \
    }                                                                                       \
                                                                                            \
    std::array<_Type_, 2> _Getter_##Ids() const                                             \
    {                                                                                       \
      auto a = *mColumnIterator;                                                            \
      return std::array{a[0], a[1]};                                                        \
    }                                                                                       \
                                                                                            \
    template <typename T>                                                                   \
    auto _Getter_##_as() const                                                              \
    {                                                                                       \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                   \
        o2::soa::notBoundTable(#_Table_);                                                   \
      }                                                                                     \
      auto t = mBinding.get<T>();                                                           \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                              \
        o2::soa::dereferenceWithWrongType();                                                \
      }                                                                                     \
      if (O2_BUILTIN_UNLIKELY(!has_##_Getter_())) {                                         \
        return t->emptySlice();                                                             \
      }                                                                                     \
      auto a = *mColumnIterator;                                                            \
      auto r = t->rawSlice(a[0], a[1]);                                                     \
      t->copyIndexBindings(r);                                                              \
      r.bindInternalIndicesTo(t);                                                           \
      return r;                                                                             \
    }                                                                                       \
                                                                                            \
    auto _Getter_() const                                                                   \
    {                                                                                       \
      return _Getter_##_as<binding_t>();                                                    \
    }                                                                                       \
                                                                                            \
    template <typename T>                                                                   \
    bool setCurrent(T const* current)                                                       \
    {                                                                                       \
      if constexpr (o2::soa::is_binding_compatible_v<T, binding_t>()) {                     \
        assert(current != nullptr);                                                         \
        this->mBinding.bind(current);                                                       \
        return true;                                                                        \
      }                                                                                     \
      return false;                                                                         \
    }                                                                                       \
                                                                                            \
    bool setCurrentRaw(o2::soa::Binding current)                                            \
    {                                                                                       \
      this->mBinding = current;                                                             \
      return true;                                                                          \
    }                                                                                       \
    binding_t const* getCurrent() const { return mBinding.get<binding_t>(); }               \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                             \
    o2::soa::Binding mBinding;                                                              \
  };

#define DECLARE_SOA_SLICE_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_SLICE_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, _Name_##s, "")

/// ARRAY
#define DECLARE_SOA_ARRAY_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Table_, _Suffix_)         \
  struct _Name_##Ids : o2::soa::Column<std::vector<_Type_>, _Name_##Ids> {                       \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");                    \
    static_assert((*_Suffix_ == '\0') || (*_Suffix_ == '_'), "Suffix has to begin with _");      \
    static constexpr const char* mLabel = "fIndexArray" #_Table_ _Suffix_;                       \
    using base = o2::soa::Column<std::vector<_Type_>, _Name_##Ids>;                              \
    using type = std::vector<_Type_>;                                                            \
    using column_t = _Name_##Ids;                                                                \
    using binding_t = _Table_;                                                                   \
    _Name_##Ids(arrow::ChunkedArray const* column)                                               \
      : o2::soa::Column<std::vector<_Type_>, _Name_##Ids>(o2::soa::ColumnIterator<type>(column)) \
    {                                                                                            \
    }                                                                                            \
                                                                                                 \
    _Name_##Ids() = default;                                                                     \
    _Name_##Ids(_Name_##Ids const& other) = default;                                             \
    _Name_##Ids& operator=(_Name_##Ids const& other) = default;                                  \
                                                                                                 \
    gsl::span<const _Type_> inline getIds() const                                                \
    {                                                                                            \
      return _Getter_##Ids();                                                                    \
    }                                                                                            \
                                                                                                 \
    gsl::span<const _Type_> _Getter_##Ids() const                                                \
    {                                                                                            \
      return *mColumnIterator;                                                                   \
    }                                                                                            \
                                                                                                 \
    bool has_##_Getter_() const                                                                  \
    {                                                                                            \
      return !(*mColumnIterator).empty();                                                        \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    auto _Getter_##_as() const                                                                   \
    {                                                                                            \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                        \
        o2::soa::notBoundTable(#_Table_);                                                        \
      }                                                                                          \
      auto t = mBinding.get<T>();                                                                \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                   \
        o2::soa::dereferenceWithWrongType();                                                     \
      }                                                                                          \
      return getIterators<T>();                                                                  \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    auto filtered_##_Getter_##_as() const                                                        \
    {                                                                                            \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                        \
        o2::soa::notBoundTable(#_Table_);                                                        \
      }                                                                                          \
      auto t = mBinding.get<T>();                                                                \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                   \
        o2::soa::dereferenceWithWrongType();                                                     \
      }                                                                                          \
      return getFilteredIterators<T>();                                                          \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    auto getIterators() const                                                                    \
    {                                                                                            \
      auto result = std::vector<typename T::unfiltered_iterator>();                              \
      for (auto& i : *mColumnIterator) {                                                         \
        result.push_back(mBinding.get<T>()->rawIteratorAt(i));                                   \
      }                                                                                          \
      return result;                                                                             \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    std::vector<typename T::iterator> getFilteredIterators() const                               \
    {                                                                                            \
      if constexpr (o2::soa::is_soa_filtered_v<T>) {                                             \
        auto result = std::vector<typename T::iterator>();                                       \
        for (auto const& i : *mColumnIterator) {                                                 \
          auto pos = mBinding.get<T>()->isInSelectedRows(i);                                     \
          if (pos > 0) {                                                                         \
            result.push_back(mBinding.get<T>()->iteratorAt(pos));                                \
          }                                                                                      \
        }                                                                                        \
        return result;                                                                           \
      } else {                                                                                   \
        static_assert(o2::framework::always_static_assert_v<T>, "T is not a Filtered type");     \
      }                                                                                          \
      return {};                                                                                 \
    }                                                                                            \
                                                                                                 \
    auto _Getter_() const                                                                        \
    {                                                                                            \
      return _Getter_##_as<binding_t>();                                                         \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    auto _Getter_##_first_as() const                                                             \
    {                                                                                            \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                        \
        o2::soa::notBoundTable(#_Table_);                                                        \
      }                                                                                          \
      auto t = mBinding.get<T>();                                                                \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                   \
        o2::soa::dereferenceWithWrongType();                                                     \
      }                                                                                          \
      return t->rawIteratorAt((*mColumnIterator)[0]);                                            \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    auto _Getter_##_last_as() const                                                              \
    {                                                                                            \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                        \
        o2::soa::notBoundTable(#_Table_);                                                        \
      }                                                                                          \
      auto t = mBinding.get<T>();                                                                \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                   \
        o2::soa::dereferenceWithWrongType();                                                     \
      }                                                                                          \
      return t->rawIteratorAt((*mColumnIterator).back());                                        \
    }                                                                                            \
                                                                                                 \
    auto _Getter_first() const                                                                   \
    {                                                                                            \
      return _Getter_##_first_as<binding_t>();                                                   \
    }                                                                                            \
                                                                                                 \
    auto _Getter_last() const                                                                    \
    {                                                                                            \
      return _Getter_##_last_as<binding_t>();                                                    \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    bool setCurrent(T const* current)                                                            \
    {                                                                                            \
      if constexpr (o2::soa::is_binding_compatible_v<T, binding_t>()) {                          \
        assert(current != nullptr);                                                              \
        this->mBinding.bind(current);                                                            \
        return true;                                                                             \
      }                                                                                          \
      return false;                                                                              \
    }                                                                                            \
                                                                                                 \
    bool setCurrentRaw(o2::soa::Binding current)                                                 \
    {                                                                                            \
      this->mBinding = current;                                                                  \
      return true;                                                                               \
    }                                                                                            \
    binding_t const* getCurrent() const { return mBinding.get<binding_t>(); }                    \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                                  \
    o2::soa::Binding mBinding;                                                                   \
  };

#define DECLARE_SOA_ARRAY_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_ARRAY_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, _Name_##s, "")

/// NORMAL
#define DECLARE_SOA_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Table_, _Suffix_)                                                                                           \
  struct _Name_##Id : o2::soa::Column<_Type_, _Name_##Id> {                                                                                                                  \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");                                                                                                \
    static_assert((*_Suffix_ == '\0') || (*_Suffix_ == '_'), "Suffix has to begin with _");                                                                                  \
    static constexpr const char* mLabel = "fIndex" #_Table_ _Suffix_;                                                                                                        \
    using base = o2::soa::Column<_Type_, _Name_##Id>;                                                                                                                        \
    using type = _Type_;                                                                                                                                                     \
    using column_t = _Name_##Id;                                                                                                                                             \
    using binding_t = _Table_;                                                                                                                                               \
    _Name_##Id(arrow::ChunkedArray const* column)                                                                                                                            \
      : o2::soa::Column<_Type_, _Name_##Id>(o2::soa::ColumnIterator<type>(column))                                                                                           \
    {                                                                                                                                                                        \
    }                                                                                                                                                                        \
                                                                                                                                                                             \
    _Name_##Id() = default;                                                                                                                                                  \
    _Name_##Id(_Name_##Id const& other) = default;                                                                                                                           \
    _Name_##Id& operator=(_Name_##Id const& other) = default;                                                                                                                \
    type inline getId() const                                                                                                                                                \
    {                                                                                                                                                                        \
      return _Getter_##Id();                                                                                                                                                 \
    }                                                                                                                                                                        \
                                                                                                                                                                             \
    type _Getter_##Id() const                                                                                                                                                \
    {                                                                                                                                                                        \
      return *mColumnIterator;                                                                                                                                               \
    }                                                                                                                                                                        \
                                                                                                                                                                             \
    bool has_##_Getter_() const                                                                                                                                              \
    {                                                                                                                                                                        \
      return *mColumnIterator >= 0;                                                                                                                                          \
    }                                                                                                                                                                        \
                                                                                                                                                                             \
    template <typename T>                                                                                                                                                    \
    auto _Getter_##_as() const                                                                                                                                               \
    {                                                                                                                                                                        \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                                                                                                    \
        o2::soa::notBoundTable(#_Table_);                                                                                                                                    \
      }                                                                                                                                                                      \
      if (O2_BUILTIN_UNLIKELY(!has_##_Getter_())) {                                                                                                                          \
        o2::soa::accessingInvalidIndexFor(#_Getter_);                                                                                                                        \
      }                                                                                                                                                                      \
      auto t = mBinding.get<T>();                                                                                                                                            \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                                                                                               \
        o2::soa::dereferenceWithWrongType();                                                                                                                                 \
      }                                                                                                                                                                      \
      return t->rawIteratorAt(*mColumnIterator);                                                                                                                             \
    }                                                                                                                                                                        \
                                                                                                                                                                             \
    auto _Getter_() const                                                                                                                                                    \
    {                                                                                                                                                                        \
      return _Getter_##_as<binding_t>();                                                                                                                                     \
    }                                                                                                                                                                        \
                                                                                                                                                                             \
    template <typename T>                                                                                                                                                    \
    bool setCurrent(T* current)                                                                                                                                              \
    {                                                                                                                                                                        \
      if constexpr (o2::soa::is_binding_compatible_v<T, binding_t>()) {                                                                                                      \
        assert(current != nullptr);                                                                                                                                          \
        this->mBinding.bind(current);                                                                                                                                        \
        return true;                                                                                                                                                         \
      }                                                                                                                                                                      \
      return false;                                                                                                                                                          \
    }                                                                                                                                                                        \
                                                                                                                                                                             \
    bool setCurrentRaw(o2::soa::Binding current)                                                                                                                             \
    {                                                                                                                                                                        \
      this->mBinding = current;                                                                                                                                              \
      return true;                                                                                                                                                           \
    }                                                                                                                                                                        \
    binding_t const* getCurrent() const { return mBinding.get<binding_t>(); }                                                                                                \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                                                                                                              \
    o2::soa::Binding mBinding;                                                                                                                                               \
  };                                                                                                                                                                         \
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_##Id { "fIndex" #_Table_ _Suffix_, o2::framework::TypeIdHelpers::uniqueId<_Name_##Id>(), \
                                                                                           o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, _Name_##s, "")

/// SELF
#define DECLARE_SOA_SELF_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_)                                                                                      \
  struct _Name_##Id : o2::soa::Column<_Type_, _Name_##Id> {                                                                                                        \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");                                                                                      \
    static constexpr const char* mLabel = "fIndex" _Label_;                                                                                                        \
    using base = o2::soa::Column<_Type_, _Name_##Id>;                                                                                                              \
    using type = _Type_;                                                                                                                                           \
    using column_t = _Name_##Id;                                                                                                                                   \
    using self_index_t = std::true_type;                                                                                                                           \
    _Name_##Id(arrow::ChunkedArray const* column)                                                                                                                  \
      : o2::soa::Column<_Type_, _Name_##Id>(o2::soa::ColumnIterator<type>(column))                                                                                 \
    {                                                                                                                                                              \
    }                                                                                                                                                              \
                                                                                                                                                                   \
    _Name_##Id() = default;                                                                                                                                        \
    _Name_##Id(_Name_##Id const& other) = default;                                                                                                                 \
    _Name_##Id& operator=(_Name_##Id const& other) = default;                                                                                                      \
    type inline getId() const                                                                                                                                      \
    {                                                                                                                                                              \
      return _Getter_##Id();                                                                                                                                       \
    }                                                                                                                                                              \
                                                                                                                                                                   \
    type _Getter_##Id() const                                                                                                                                      \
    {                                                                                                                                                              \
      return *mColumnIterator;                                                                                                                                     \
    }                                                                                                                                                              \
                                                                                                                                                                   \
    bool has_##_Getter_() const                                                                                                                                    \
    {                                                                                                                                                              \
      return *mColumnIterator >= 0;                                                                                                                                \
    }                                                                                                                                                              \
                                                                                                                                                                   \
    template <typename T>                                                                                                                                          \
    auto _Getter_##_as() const                                                                                                                                     \
    {                                                                                                                                                              \
      if (O2_BUILTIN_UNLIKELY(!has_##_Getter_())) {                                                                                                                \
        o2::soa::accessingInvalidIndexFor(#_Getter_);                                                                                                              \
      }                                                                                                                                                            \
      auto t = mBinding.get<T>();                                                                                                                                  \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                                                                                     \
        o2::soa::dereferenceWithWrongType();                                                                                                                       \
      }                                                                                                                                                            \
      return t->rawIteratorAt(*mColumnIterator);                                                                                                                   \
    }                                                                                                                                                              \
                                                                                                                                                                   \
    bool setCurrentRaw(o2::soa::Binding current)                                                                                                                   \
    {                                                                                                                                                              \
      this->mBinding = current;                                                                                                                                    \
      return true;                                                                                                                                                 \
    }                                                                                                                                                              \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                                                                                                    \
    o2::soa::Binding mBinding;                                                                                                                                     \
  };                                                                                                                                                               \
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_##Id { "fIndex" _Label_, o2::framework::TypeIdHelpers::uniqueId<_Name_##Id>(), \
                                                                                           o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_SELF_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_SELF_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, #_Name_)
/// SELF SLICE
#define DECLARE_SOA_SELF_SLICE_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_)        \
  struct _Name_##IdSlice : o2::soa::Column<_Type_[2], _Name_##IdSlice> {                   \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");              \
    static constexpr const char* mLabel = "fIndexSlice" _Label_;                           \
    using base = o2::soa::Column<_Type_[2], _Name_##IdSlice>;                              \
    using type = _Type_[2];                                                                \
    using column_t = _Name_##IdSlice;                                                      \
    using self_index_t = std::true_type;                                                   \
    _Name_##IdSlice(arrow::ChunkedArray const* column)                                     \
      : o2::soa::Column<_Type_[2], _Name_##IdSlice>(o2::soa::ColumnIterator<type>(column)) \
    {                                                                                      \
    }                                                                                      \
                                                                                           \
    _Name_##IdSlice() = default;                                                           \
    _Name_##IdSlice(_Name_##IdSlice const& other) = default;                               \
    _Name_##IdSlice& operator=(_Name_##IdSlice const& other) = default;                    \
    std::array<_Type_, 2> inline getIds() const                                            \
    {                                                                                      \
      return _Getter_##Ids();                                                              \
    }                                                                                      \
                                                                                           \
    bool has_##_Getter_() const                                                            \
    {                                                                                      \
      auto a = *mColumnIterator;                                                           \
      return a[0] >= 0 && a[1] >= 0;                                                       \
    }                                                                                      \
                                                                                           \
    std::array<_Type_, 2> _Getter_##Ids() const                                            \
    {                                                                                      \
      auto a = *mColumnIterator;                                                           \
      return std::array{a[0], a[1]};                                                       \
    }                                                                                      \
                                                                                           \
    template <typename T>                                                                  \
    auto _Getter_##_as() const                                                             \
    {                                                                                      \
      auto t = mBinding.get<T>();                                                          \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                             \
        o2::soa::dereferenceWithWrongType();                                               \
      }                                                                                    \
      if (O2_BUILTIN_UNLIKELY(!has_##_Getter_())) {                                        \
        return t->emptySlice();                                                            \
      }                                                                                    \
      auto a = *mColumnIterator;                                                           \
      auto r = t->rawSlice(a[0], a[1]);                                                    \
      t->copyIndexBindings(r);                                                             \
      r.bindInternalIndicesTo(t);                                                          \
      return r;                                                                            \
    }                                                                                      \
                                                                                           \
    bool setCurrentRaw(o2::soa::Binding current)                                           \
    {                                                                                      \
      this->mBinding = current;                                                            \
      return true;                                                                         \
    }                                                                                      \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                            \
    o2::soa::Binding mBinding;                                                             \
  };

#define DECLARE_SOA_SELF_SLICE_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_SELF_SLICE_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, "_" #_Name_)
/// SELF ARRAY
#define DECLARE_SOA_SELF_ARRAY_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_)              \
  struct _Name_##Ids : o2::soa::Column<std::vector<_Type_>, _Name_##Ids> {                       \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");                    \
    static constexpr const char* mLabel = "fIndexArray" _Label_;                                 \
    using base = o2::soa::Column<std::vector<_Type_>, _Name_##Ids>;                              \
    using type = std::vector<_Type_>;                                                            \
    using column_t = _Name_##Ids;                                                                \
    using self_index_t = std::true_type;                                                         \
    _Name_##Ids(arrow::ChunkedArray const* column)                                               \
      : o2::soa::Column<std::vector<_Type_>, _Name_##Ids>(o2::soa::ColumnIterator<type>(column)) \
    {                                                                                            \
    }                                                                                            \
                                                                                                 \
    _Name_##Ids() = default;                                                                     \
    _Name_##Ids(_Name_##Ids const& other) = default;                                             \
    _Name_##Ids& operator=(_Name_##Ids const& other) = default;                                  \
    gsl::span<const _Type_> inline getIds() const                                                \
    {                                                                                            \
      return _Getter_##Ids();                                                                    \
    }                                                                                            \
                                                                                                 \
    gsl::span<const _Type_> _Getter_##Ids() const                                                \
    {                                                                                            \
      return *mColumnIterator;                                                                   \
    }                                                                                            \
                                                                                                 \
    bool has_##_Getter_() const                                                                  \
    {                                                                                            \
      return !(*mColumnIterator).empty();                                                        \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    auto _Getter_##_as() const                                                                   \
    {                                                                                            \
      auto t = mBinding.get<T>();                                                                \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                   \
        o2::soa::dereferenceWithWrongType();                                                     \
      }                                                                                          \
      return getIterators<T>();                                                                  \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    auto getIterators() const                                                                    \
    {                                                                                            \
      auto result = std::vector<typename T::unfiltered_iterator>();                              \
      for (auto& i : *mColumnIterator) {                                                         \
        result.push_back(mBinding.get<T>()->rawIteratorAt(i));                                   \
      }                                                                                          \
      return result;                                                                             \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    auto _Getter_##_first_as() const                                                             \
    {                                                                                            \
      return mBinding.get<T>()->rawIteratorAt((*mColumnIterator)[0]);                            \
    }                                                                                            \
                                                                                                 \
    template <typename T>                                                                        \
    auto _Getter_##_last_as() const                                                              \
    {                                                                                            \
      return mBinding.get<T>()->rawIteratorAt((*mColumnIterator).back());                        \
    }                                                                                            \
                                                                                                 \
    bool setCurrentRaw(o2::soa::Binding current)                                                 \
    {                                                                                            \
      this->mBinding = current;                                                                  \
      return true;                                                                               \
    }                                                                                            \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                                  \
    o2::soa::Binding mBinding;                                                                   \
  };

#define DECLARE_SOA_SELF_ARRAY_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_SELF_ARRAY_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, "_" #_Name_)

/// A dynamic column is a column whose values are derived
/// from those of other real columns. These can be used for
/// example to provide different coordinate systems (e.g. polar,
/// cylindrical) from a persister representation (e.g. cartesian).
/// _Name_ is a unique typename which will be associated with the
/// column. _Getter_ is a mnemonic to retrieve the value of
/// the column in a given row. The variadic template argument
/// (...) is used to capture a lambda or callable object which
/// will be used to perform the transformation.
/// Notice that the macro will define a template type _Name_
/// which will have as template argument the types of the columns
/// to be bound for the operation.
///
/// For example, let's assume you have:
///
/// \code{.cpp}
///
/// namespace point {
/// DECLARE_SOA_COLUMN(X, x, float, "fX");
/// DECLARE_SOA_COLUMN(Y, y, float, "fY");
/// }
///
/// DECLARE_SOA_DYNAMIC_COLUMN(R2, r2, [](x, y) { return x*x + y+y; });
///
/// DECLARE_SOA_TABLE(Point, "MISC", "POINT",
///                   X, Y, (R2<X,Y>));
/// \endcode
///
#define DECLARE_SOA_DYNAMIC_COLUMN(_Name_, _Getter_, ...)                                                                  \
  struct _Name_##Callback {                                                                                                \
    static inline constexpr auto getLambda() { return __VA_ARGS__; }                                                       \
  };                                                                                                                       \
                                                                                                                           \
  struct _Name_##Helper {                                                                                                  \
    using callable_t = decltype(o2::framework::FunctionMetadata(std::declval<decltype(_Name_##Callback::getLambda())>())); \
    using return_type = typename callable_t::return_type;                                                                  \
  };                                                                                                                       \
  template <typename... Bindings>                                                                                          \
  struct _Name_ : o2::soa::DynamicColumn<typename _Name_##Helper::callable_t::type, _Name_<Bindings...>> {                 \
    using base = o2::soa::DynamicColumn<typename _Name_##Helper::callable_t::type, _Name_<Bindings...>>;                   \
    using helper = _Name_##Helper;                                                                                         \
    using callback_holder_t = _Name_##Callback;                                                                            \
    using callable_t = helper::callable_t;                                                                                 \
    using callback_t = callable_t::type;                                                                                   \
                                                                                                                           \
    _Name_(arrow::ChunkedArray const*)                                                                                     \
    {                                                                                                                      \
    }                                                                                                                      \
    _Name_() = default;                                                                                                    \
    _Name_(_Name_ const& other) = default;                                                                                 \
    _Name_& operator=(_Name_ const& other) = default;                                                                      \
    static constexpr const char* mLabel = #_Name_;                                                                         \
    using type = typename callable_t::return_type;                                                                         \
                                                                                                                           \
    template <typename... FreeArgs>                                                                                        \
    type _Getter_(FreeArgs... freeArgs) const                                                                              \
    {                                                                                                                      \
      return boundGetter(std::make_index_sequence<std::tuple_size_v<decltype(boundIterators)>>{}, freeArgs...);            \
    }                                                                                                                      \
    template <typename... FreeArgs>                                                                                        \
    type getDynamicValue(FreeArgs... freeArgs) const                                                                       \
    {                                                                                                                      \
      return boundGetter(std::make_index_sequence<std::tuple_size_v<decltype(boundIterators)>>{}, freeArgs...);            \
    }                                                                                                                      \
                                                                                                                           \
    type get() const                                                                                                       \
    {                                                                                                                      \
      return _Getter_();                                                                                                   \
    }                                                                                                                      \
                                                                                                                           \
    template <size_t... Is, typename... FreeArgs>                                                                          \
    type boundGetter(std::integer_sequence<size_t, Is...>&&, FreeArgs... freeArgs) const                                   \
    {                                                                                                                      \
      return __VA_ARGS__((**std::get<Is>(boundIterators))..., freeArgs...);                                                \
    }                                                                                                                      \
                                                                                                                           \
    using bindings_t = typename o2::framework::pack<Bindings...>;                                                          \
    std::tuple<o2::soa::ColumnIterator<typename Bindings::type> const*...> boundIterators;                                 \
  }

#define DECLARE_SOA_TABLE_FULL_VERSIONED(_Name_, _Label_, _Origin_, _Description_, _Version_, ...) \
  using _Name_ = o2::soa::Table<__VA_ARGS__>;                                                      \
                                                                                                   \
  struct _Name_##Metadata : o2::soa::TableMetadata<_Name_##Metadata> {                             \
    using table_t = _Name_;                                                                        \
    static constexpr o2::header::DataHeader::SubSpecificationType mVersion = _Version_;            \
    static constexpr char const* mLabel = _Label_;                                                 \
    static constexpr char const mOrigin[5] = _Origin_;                                             \
    static constexpr char const mDescription[16] = _Description_;                                  \
  };                                                                                               \
                                                                                                   \
  template <>                                                                                      \
  struct MetadataTrait<_Name_> {                                                                   \
    using metadata = _Name_##Metadata;                                                             \
  };                                                                                               \
                                                                                                   \
  template <>                                                                                      \
  struct MetadataTrait<_Name_::unfiltered_iterator> {                                              \
    using metadata = _Name_##Metadata;                                                             \
  };

#define DECLARE_SOA_TABLE_FULL(_Name_, _Label_, _Origin_, _Description_, ...) \
  DECLARE_SOA_TABLE_FULL_VERSIONED(_Name_, _Label_, _Origin_, _Description_, 0, __VA_ARGS__);
#define DECLARE_SOA_TABLE(_Name_, _Origin_, _Description_, ...) \
  DECLARE_SOA_TABLE_FULL(_Name_, #_Name_, _Origin_, _Description_, __VA_ARGS__);
#define DECLARE_SOA_TABLE_VERSIONED(_Name_, _Origin_, _Description_, _Version_, ...) \
  DECLARE_SOA_TABLE_FULL_VERSIONED(_Name_, #_Name_, _Origin_, _Description_, _Version_, __VA_ARGS__);

#define DECLARE_SOA_EXTENDED_TABLE_FULL(_Name_, _Table_, _Origin_, _Description_, ...)                                          \
  struct _Name_##Extension : o2::soa::Table<__VA_ARGS__> {                                                                      \
    using base_t = o2::soa::Table<__VA_ARGS__>;                                                                                 \
    _Name_##Extension(std::shared_ptr<arrow::Table> table, uint64_t offset = 0) : o2::soa::Table<__VA_ARGS__>(table, offset){}; \
    _Name_##Extension(_Name_##Extension const&) = default;                                                                      \
    _Name_##Extension(_Name_##Extension&&) = default;                                                                           \
    using expression_pack_t = framework::pack<__VA_ARGS__>;                                                                     \
    using iterator = typename base_t::template RowView<_Name_##Extension, _Name_##Extension>;                                   \
    using const_iterator = iterator;                                                                                            \
  };                                                                                                                            \
  using _Name_ = o2::soa::Join<_Name_##Extension, _Table_>;                                                                     \
                                                                                                                                \
  struct _Name_##ExtensionMetadata : o2::soa::TableMetadata<_Name_##ExtensionMetadata> {                                        \
    using table_t = _Name_##Extension;                                                                                          \
    using base_table_t = _Table_;                                                                                               \
    using expression_pack_t = typename _Name_##Extension::expression_pack_t;                                                    \
    using originals = soa::originals_pack_t<_Table_>;                                                                           \
    using sources = originals;                                                                                                  \
    static constexpr o2::header::DataHeader::SubSpecificationType mVersion = getVersion<_Table_>();                             \
    static constexpr char const* mLabel = #_Name_ "Extension";                                                                  \
    static constexpr char const mOrigin[5] = _Origin_;                                                                          \
    static constexpr char const mDescription[16] = _Description_;                                                               \
  };                                                                                                                            \
                                                                                                                                \
  template <>                                                                                                                   \
  struct MetadataTrait<_Name_##Extension> {                                                                                     \
    using metadata = _Name_##ExtensionMetadata;                                                                                 \
  };

#define DECLARE_SOA_EXTENDED_TABLE(_Name_, _Table_, _Description_, ...) \
  DECLARE_SOA_EXTENDED_TABLE_FULL(_Name_, _Table_, "DYN", _Description_, __VA_ARGS__)

#define DECLARE_SOA_EXTENDED_TABLE_USER(_Name_, _Table_, _Description_, ...) \
  DECLARE_SOA_EXTENDED_TABLE_FULL(_Name_, _Table_, "AOD", _Description_, __VA_ARGS__)

#define DECLARE_SOA_INDEX_TABLE_FULL(_Name_, _Key_, _Origin_, _Description_, _Exclusive_, ...)                                   \
  struct _Name_ : o2::soa::IndexTable<_Key_, __VA_ARGS__> {                                                                      \
    _Name_(std::shared_ptr<arrow::Table> table, uint64_t offset = 0) : o2::soa::IndexTable<_Key_, __VA_ARGS__>(table, offset){}; \
    _Name_(_Name_ const&) = default;                                                                                             \
    _Name_(_Name_&&) = default;                                                                                                  \
    using iterator = typename base_t::template RowView<_Name_, _Name_>;                                                          \
    using const_iterator = iterator;                                                                                             \
  };                                                                                                                             \
                                                                                                                                 \
  struct _Name_##Metadata : o2::soa::TableMetadata<_Name_##Metadata> {                                                           \
    using Key = _Key_;                                                                                                           \
    using index_pack_t = framework::pack<__VA_ARGS__>;                                                                           \
    using originals = decltype(soa::extractBindings(index_pack_t{}));                                                            \
    using sources = typename _Name_::sources_t;                                                                                  \
    static constexpr o2::header::DataHeader::SubSpecificationType mVersion = 0;                                                  \
    static constexpr char const* mLabel = #_Name_;                                                                               \
    static constexpr char const mOrigin[5] = _Origin_;                                                                           \
    static constexpr char const mDescription[16] = _Description_;                                                                \
    static constexpr bool exclusive = _Exclusive_;                                                                               \
  };                                                                                                                             \
                                                                                                                                 \
  template <>                                                                                                                    \
  struct MetadataTrait<_Name_> {                                                                                                 \
    using metadata = _Name_##Metadata;                                                                                           \
  };                                                                                                                             \
                                                                                                                                 \
  template <>                                                                                                                    \
  struct MetadataTrait<_Name_::iterator> {                                                                                       \
    using metadata = _Name_##Metadata;                                                                                           \
  };

#define DECLARE_SOA_INDEX_TABLE(_Name_, _Key_, _Description_, ...) \
  DECLARE_SOA_INDEX_TABLE_FULL(_Name_, _Key_, "IDX", _Description_, false, __VA_ARGS__)

#define DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(_Name_, _Key_, _Description_, ...) \
  DECLARE_SOA_INDEX_TABLE_FULL(_Name_, _Key_, "IDX", _Description_, true, __VA_ARGS__)

#define DECLARE_SOA_INDEX_TABLE_USER(_Name_, _Key_, _Description_, ...) \
  DECLARE_SOA_INDEX_TABLE_FULL(_Name_, _Key_, "AOD", _Description_, false, __VA_ARGS__)

#define DECLARE_SOA_INDEX_TABLE_EXCLUSIVE_USER(_Name_, _Key_, _Description_, ...) \
  DECLARE_SOA_INDEX_TABLE_FULL(_Name_, _Key_, "AOD", _Description_, true, __VA_ARGS__)

namespace o2::soa
{
template <typename T>
class FilteredBase;

template <typename... Ts>
struct Join : TableWrap<Ts...>::table_t {
  using base = typename TableWrap<Ts...>::table_t;
  using originals = originals_pack_t<Ts...>;

  Join(std::vector<std::shared_ptr<arrow::Table>>&& tables, uint64_t offset = 0)
    : base{ArrowHelpers::joinTables(std::move(tables)), offset}
  {
    bindInternalIndicesTo(this);
  }
  using base::bindExternalIndices;
  using base::bindInternalIndicesTo;

  using self_t = Join<Ts...>;
  using table_t = base;
  using persistent_columns_t = typename table_t::persistent_columns_t;
  using iterator = decltype([]<typename... Os>(framework::pack<Os...>) { return typename table_t::template RowView<Join<Ts...>, Os...>{}; }(originals{}));
  using const_iterator = iterator;
  using unfiltered_iterator = iterator;
  using unfiltered_const_iterator = const_iterator;
  using filtered_iterator = decltype([]<typename... Os>(framework::pack<Os...>) { return typename table_t::template RowViewFiltered<Filtered<Join<Ts...>>, Os...>{}; }(originals{}));
  using filtered_const_iterator = filtered_iterator;

  iterator begin()
  {
    return iterator{this->cached_begin()};
  }

  const_iterator begin() const
  {
    return const_iterator{this->cached_begin()};
  }

  auto sliceByCached(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return doSliceByCached(this, node, value, cache);
  }

  auto sliceByCachedUnsorted(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return doSliceByCachedUnsorted(this, node, value, cache);
  }

  template <typename T1, bool OPT, bool SORTED>
  auto sliceBy(o2::framework::PresliceBase<T1, OPT, SORTED> const& container, int value) const
  {
    return doSliceBy(this, container, value);
  }

  iterator rawIteratorAt(uint64_t i) const
  {
    auto it = iterator{this->cached_begin()};
    it.setCursor(i);
    return it;
  }

  iterator iteratorAt(uint64_t i) const
  {
    return rawIteratorAt(i);
  }

  auto rawSlice(uint64_t start, uint64_t end) const
  {
    return self_t{{this->asArrowTable()->Slice(start, end - start + 1)}, start};
  }

  auto emptySlice() const
  {
    return self_t{{this->asArrowTable()->Slice(0, 0)}, 0};
  }

  template <typename T>
  static consteval bool contains()
  {
    if constexpr (is_type_with_originals_v<T>) {
      return contains(typename T::originals{});
    } else {
      return framework::has_type<T>(originals{});
    }
  }

  template <typename... TTs>
  static constexpr bool contains(framework::pack<TTs...>)
  {
    return (contains<TTs>() || ...);
  }
};

template <typename T1, typename T2>
struct Concat : ConcatBase<T1, T2> {
  Concat(std::shared_ptr<arrow::Table> t1, std::shared_ptr<arrow::Table> t2, uint64_t offset = 0)
    : ConcatBase<T1, T2>{ArrowHelpers::concatTables({t1, t2}), offset}
  {
    bindInternalIndicesTo(this);
  }
  Concat(std::vector<std::shared_ptr<arrow::Table>> tables, uint64_t offset = 0)
    : ConcatBase<T1, T2>{ArrowHelpers::concatTables(std::move(tables)), offset}
  {
    bindInternalIndicesTo(this);
  }

  using base = ConcatBase<T1, T2>;
  using originals = framework::concatenated_pack_t<originals_pack_t<T1>, originals_pack_t<T2>>;

  using base::bindExternalIndices;
  using base::bindInternalIndicesTo;

  // FIXME: can be remove when we do the same treatment we did for Join to Concatenate
  using left_t = T1;
  using right_t = T2;
  using table_t = ConcatBase<T1, T2>;
  using persistent_columns_t = typename table_t::persistent_columns_t;

  using iterator = typename table_t::template RowView<Concat<T1, T2>, T1, T2>;
  using filtered_iterator = typename table_t::template RowViewFiltered<Concat<T1, T2>, T1, T2>;
};

template <typename T>
using is_soa_join_t = framework::is_specialization<T, soa::Join>;

template <typename T>
using is_soa_concat_t = framework::is_specialization<T, soa::Concat>;

template <typename T>
inline constexpr bool is_soa_join_v = is_soa_join_t<T>::value;

template <typename T>
inline constexpr bool is_soa_concat_v = is_soa_concat_t<T>::value;

template <typename T>
class FilteredBase : public T
{
 public:
  using self_t = FilteredBase<T>;
  using originals = originals_pack_t<T>;
  using table_t = typename T::table_t;
  using persistent_columns_t = typename T::persistent_columns_t;
  using external_index_columns_t = typename T::external_index_columns_t;

  using iterator = decltype([]<typename... Os>(framework::pack<Os...>) { return typename table_t::template RowViewFiltered<FilteredBase<T>, Os...>{}; }(originals{}));
  using unfiltered_iterator = decltype([]<typename... Os>(framework::pack<Os...>) { return typename table_t::template RowView<FilteredBase<T>, Os...>{}; }(originals{}));
  using const_iterator = iterator;

  FilteredBase(std::vector<std::shared_ptr<arrow::Table>>&& tables, gandiva::Selection const& selection, uint64_t offset = 0)
    : T{std::move(tables), offset},
      mSelectedRows{getSpan(selection)}
  {
    if (this->tableSize() != 0) {
      mFilteredBegin = table_t::filtered_begin(mSelectedRows);
    }
    resetRanges();
    mFilteredBegin.bindInternalIndices(this);
  }

  FilteredBase(std::vector<std::shared_ptr<arrow::Table>>&& tables, SelectionVector&& selection, uint64_t offset = 0)
    : T{std::move(tables), offset},
      mSelectedRowsCache{std::move(selection)},
      mCached{true}
  {
    mSelectedRows = gsl::span{mSelectedRowsCache};
    if (this->tableSize() != 0) {
      mFilteredBegin = table_t::filtered_begin(mSelectedRows);
    }
    resetRanges();
    mFilteredBegin.bindInternalIndices(this);
  }

  FilteredBase(std::vector<std::shared_ptr<arrow::Table>>&& tables, gsl::span<int64_t const> const& selection, uint64_t offset = 0)
    : T{std::move(tables), offset},
      mSelectedRows{selection}
  {
    if (this->tableSize() != 0) {
      mFilteredBegin = table_t::filtered_begin(mSelectedRows);
    }
    resetRanges();
    mFilteredBegin.bindInternalIndices(this);
  }

  iterator begin()
  {
    return iterator(mFilteredBegin);
  }

  const_iterator begin() const
  {
    return const_iterator(mFilteredBegin);
  }

  unfiltered_iterator rawIteratorAt(uint64_t i) const
  {
    auto it = unfiltered_iterator{mFilteredBegin};
    it.setCursor(i);
    return it;
  }

  [[nodiscard]] RowViewSentinel end() const
  {
    return RowViewSentinel{*mFilteredEnd};
  }

  auto& cached_begin()
  {
    return mFilteredBegin;
  }

  auto const& cached_begin() const
  {
    return mFilteredBegin;
  }

  iterator iteratorAt(uint64_t i) const
  {
    return mFilteredBegin + i;
  }

  [[nodiscard]] int64_t size() const
  {
    return mSelectedRows.size();
  }

  [[nodiscard]] int64_t tableSize() const
  {
    return table_t::asArrowTable()->num_rows();
  }

  auto const& getSelectedRows() const
  {
    return mSelectedRows;
  }

  auto rawSlice(uint64_t start, uint64_t end) const
  {
    SelectionVector newSelection;
    newSelection.resize(static_cast<int64_t>(end - start + 1));
    std::iota(newSelection.begin(), newSelection.end(), start);
    return self_t{{this->asArrowTable()}, std::move(newSelection), 0};
  }

  auto emptySlice() const
  {
    return self_t{{this->asArrowTable()}, SelectionVector{}, 0};
  }

  static inline auto getSpan(gandiva::Selection const& sel)
  {
    if (sel == nullptr) {
      return gsl::span<int64_t const>{};
    }
    auto array = std::static_pointer_cast<arrow::Int64Array>(sel->ToArray());
    auto start = array->raw_values();
    auto stop = start + array->length();
    return gsl::span{start, stop};
  }

  /// Bind the columns which refer to other tables
  /// to the associated tables.
  template <typename... TA>
  void bindExternalIndices(TA*... current)
  {
    table_t::bindExternalIndices(current...);
    mFilteredBegin.bindExternalIndices(current...);
  }

  void bindExternalIndicesRaw(std::vector<o2::soa::Binding>&& ptrs)
  {
    mFilteredBegin.bindExternalIndicesRaw(std::forward<std::vector<o2::soa::Binding>>(ptrs));
  }

  template <typename I>
  void bindInternalIndicesTo(I const* ptr)
  {
    mFilteredBegin.bindInternalIndices(ptr);
  }

  template <typename T1, typename... Cs>
  void doCopyIndexBindings(framework::pack<Cs...>, T1& dest) const
  {
    dest.bindExternalIndicesRaw(mFilteredBegin.getIndexBindings());
  }

  template <typename T1>
  void copyIndexBindings(T1& dest) const
  {
    doCopyIndexBindings(external_index_columns_t{}, dest);
  }

  template <typename T1>
  auto rawSliceBy(o2::framework::Preslice<T1> const& container, int value) const
  {
    return (table_t)this->sliceBy(container, value);
  }

  auto sliceByCached(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return doFilteredSliceByCached(this, node, value, cache);
  }

  auto sliceByCachedUnsorted(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return doSliceByCachedUnsorted(this, node, value, cache);
  }

  template <typename T1, bool OPT, bool SORTED>
  auto sliceBy(o2::framework::PresliceBase<T1, OPT, SORTED> const& container, int value) const
  {
    if constexpr (SORTED) {
      return doFilteredSliceBy(this, container, value);
    } else {
      return doSliceBy(this, container, value);
    }
  }

  auto select(framework::expressions::Filter const& f) const
  {
    auto t = o2::soa::select(*this, f);
    copyIndexBindings(t);
    return t;
  }

  int isInSelectedRows(int i) const
  {
    auto locate = std::find(mSelectedRows.begin(), mSelectedRows.end(), i);
    if (locate == mSelectedRows.end()) {
      return -1;
    }
    return static_cast<int>(std::distance(mSelectedRows.begin(), locate));
  }

  void sumWithSelection(SelectionVector const& selection)
  {
    mCached = true;
    SelectionVector rowsUnion;
    std::set_union(mSelectedRows.begin(), mSelectedRows.end(), selection.begin(), selection.end(), std::back_inserter(rowsUnion));
    mSelectedRowsCache.clear();
    mSelectedRowsCache = rowsUnion;
    resetRanges();
  }

  void intersectWithSelection(SelectionVector const& selection)
  {
    mCached = true;
    SelectionVector intersection;
    std::set_intersection(mSelectedRows.begin(), mSelectedRows.end(), selection.begin(), selection.end(), std::back_inserter(intersection));
    mSelectedRowsCache.clear();
    mSelectedRowsCache = intersection;
    resetRanges();
  }

  void sumWithSelection(gsl::span<int64_t const> const& selection)
  {
    mCached = true;
    SelectionVector rowsUnion;
    std::set_union(mSelectedRows.begin(), mSelectedRows.end(), selection.begin(), selection.end(), std::back_inserter(rowsUnion));
    mSelectedRowsCache.clear();
    mSelectedRowsCache = rowsUnion;
    resetRanges();
  }

  void intersectWithSelection(gsl::span<int64_t const> const& selection)
  {
    mCached = true;
    SelectionVector intersection;
    std::set_intersection(mSelectedRows.begin(), mSelectedRows.end(), selection.begin(), selection.end(), std::back_inserter(intersection));
    mSelectedRowsCache.clear();
    mSelectedRowsCache = intersection;
    resetRanges();
  }

  bool isCached() const
  {
    return mCached;
  }

 private:
  void resetRanges()
  {
    if (mCached) {
      mSelectedRows = gsl::span{mSelectedRowsCache};
    }
    mFilteredEnd.reset(new RowViewSentinel{static_cast<int64_t>(mSelectedRows.size())});
    if (tableSize() == 0) {
      mFilteredBegin = *mFilteredEnd;
    } else {
      mFilteredBegin.resetSelection(mSelectedRows);
    }
  }

  gsl::span<int64_t const> mSelectedRows;
  SelectionVector mSelectedRowsCache;
  bool mCached = false;
  iterator mFilteredBegin;
  std::shared_ptr<RowViewSentinel> mFilteredEnd;
};

template <typename T>
class Filtered : public FilteredBase<T>
{
 public:
  using base_t = T;
  using self_t = Filtered<T>;
  using table_t = typename FilteredBase<T>::table_t;
  using originals = originals_pack_t<T>;

  using iterator = decltype([]<typename... Os>(framework::pack<Os...>) { return typename table_t::template RowViewFiltered<Filtered<T>, Os...>{}; }(originals{}));
  using unfiltered_iterator = decltype([]<typename... Os>(framework::pack<Os...>) { return typename table_t::template RowView<Filtered<T>, Os...>{}; }(originals{}));
  using const_iterator = iterator;

  iterator begin()
  {
    return iterator(this->cached_begin());
  }

  const_iterator begin() const
  {
    return const_iterator(this->cached_begin());
  }

  Filtered(std::vector<std::shared_ptr<arrow::Table>>&& tables, gandiva::Selection const& selection, uint64_t offset = 0)
    : FilteredBase<T>(std::move(tables), selection, offset) {}

  Filtered(std::vector<std::shared_ptr<arrow::Table>>&& tables, SelectionVector&& selection, uint64_t offset = 0)
    : FilteredBase<T>(std::move(tables), std::forward<SelectionVector>(selection), offset) {}

  Filtered(std::vector<std::shared_ptr<arrow::Table>>&& tables, gsl::span<int64_t const> const& selection, uint64_t offset = 0)
    : FilteredBase<T>(std::move(tables), selection, offset) {}

  Filtered<T> operator+(SelectionVector const& selection)
  {
    Filtered<T> copy(*this);
    copy.sumWithSelection(selection);
    return copy;
  }

  Filtered<T> operator+(gsl::span<int64_t const> const& selection)
  {
    Filtered<T> copy(*this);
    copy.sumWithSelection(selection);
    return copy;
  }

  Filtered<T> operator+(Filtered<T> const& other)
  {
    return operator+(other.getSelectedRows());
  }

  Filtered<T> operator+=(SelectionVector const& selection)
  {
    this->sumWithSelection(selection);
    return *this;
  }

  Filtered<T> operator+=(gsl::span<int64_t const> const& selection)
  {
    this->sumWithSelection(selection);
    return *this;
  }

  Filtered<T> operator+=(Filtered<T> const& other)
  {
    return operator+=(other.getSelectedRows());
  }

  Filtered<T> operator*(SelectionVector const& selection)
  {
    Filtered<T> copy(*this);
    copy.intersectWithSelection(selection);
    return copy;
  }

  Filtered<T> operator*(gsl::span<int64_t const> const& selection)
  {
    Filtered<T> copy(*this);
    copy.intersectWithSelection(selection);
    return copy;
  }

  Filtered<T> operator*(Filtered<T> const& other)
  {
    return operator*(other.getSelectedRows());
  }

  Filtered<T> operator*=(SelectionVector const& selection)
  {
    this->intersectWithSelection(selection);
    return *this;
  }

  Filtered<T> operator*=(gsl::span<int64_t const> const& selection)
  {
    this->intersectWithSelection(selection);
    return *this;
  }

  Filtered<T> operator*=(Filtered<T> const& other)
  {
    return operator*=(other.getSelectedRows());
  }

  unfiltered_iterator rawIteratorAt(uint64_t i) const
  {
    auto it = unfiltered_iterator{this->cached_begin()};
    it.setCursor(i);
    return it;
  }

  using FilteredBase<T>::getSelectedRows;

  auto rawSlice(uint64_t start, uint64_t end) const
  {
    SelectionVector newSelection;
    newSelection.resize(static_cast<int64_t>(end - start + 1));
    std::iota(newSelection.begin(), newSelection.end(), start);
    return self_t{{this->asArrowTable()}, std::move(newSelection), 0};
  }

  auto emptySlice() const
  {
    return self_t{{this->asArrowTable()}, SelectionVector{}, 0};
  }

  template <typename T1>
  auto rawSliceBy(o2::framework::Preslice<T1> const& container, int value) const
  {
    return (table_t)this->sliceBy(container, value);
  }

  auto sliceByCached(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return doFilteredSliceByCached(this, node, value, cache);
  }

  auto sliceByCachedUnsorted(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return doSliceByCachedUnsorted(this, node, value, cache);
  }

  template <typename T1, bool OPT, bool SORTED>
  auto sliceBy(o2::framework::PresliceBase<T1, OPT, SORTED> const& container, int value) const
  {
    if constexpr (SORTED) {
      return doFilteredSliceBy(this, container, value);
    } else {
      return doSliceBy(this, container, value);
    }
  }

  auto select(framework::expressions::Filter const& f) const
  {
    auto t = o2::soa::select(*this, f);
    copyIndexBindings(t);
    return t;
  }
};

template <typename T>
class Filtered<Filtered<T>> : public FilteredBase<typename T::table_t>
{
 public:
  using self_t = Filtered<Filtered<T>>;
  using base_t = T;
  using table_t = typename FilteredBase<typename T::table_t>::table_t;
  using originals = originals_pack_t<T>;
  using iterator = decltype([]<typename... Os>(framework::pack<Os...>) { return typename table_t::template RowViewFiltered<Filtered<Filtered<T>>, Os...>{}; }(originals{}));
  using unfiltered_iterator = decltype([]<typename... Os>(framework::pack<Os...>) { return typename table_t::template RowView<Filtered<Filtered<T>>, Os...>{}; }(originals{}));
  using const_iterator = iterator;

  iterator begin()
  {
    return iterator(this->cached_begin());
  }

  const_iterator begin() const
  {
    return const_iterator(this->cached_begin());
  }

  Filtered(std::vector<Filtered<T>>&& tables, gandiva::Selection const& selection, uint64_t offset = 0)
    : FilteredBase<typename T::table_t>(std::move(extractTablesFromFiltered(tables)), selection, offset)
  {
    for (auto& table : tables) {
      *this *= table;
    }
  }

  Filtered(std::vector<Filtered<T>>&& tables, SelectionVector&& selection, uint64_t offset = 0)
    : FilteredBase<typename T::table_t>(std::move(extractTablesFromFiltered(tables)), std::forward<SelectionVector>(selection), offset)
  {
    for (auto& table : tables) {
      *this *= table;
    }
  }

  Filtered(std::vector<Filtered<T>>&& tables, gsl::span<int64_t const> const& selection, uint64_t offset = 0)
    : FilteredBase<typename T::table_t>(std::move(extractTablesFromFiltered(tables)), selection, offset)
  {
    for (auto& table : tables) {
      *this *= table;
    }
  }

  Filtered<Filtered<T>> operator+(SelectionVector const& selection)
  {
    Filtered<Filtered<T>> copy(*this);
    copy.sumWithSelection(selection);
    return copy;
  }

  Filtered<Filtered<T>> operator+(gsl::span<int64_t const> const& selection)
  {
    Filtered<Filtered<T>> copy(*this);
    copy.sumWithSelection(selection);
    return copy;
  }

  Filtered<Filtered<T>> operator+(Filtered<T> const& other)
  {
    return operator+(other.getSelectedRows());
  }

  Filtered<Filtered<T>> operator+=(SelectionVector const& selection)
  {
    this->sumWithSelection(selection);
    return *this;
  }

  Filtered<Filtered<T>> operator+=(gsl::span<int64_t const> const& selection)
  {
    this->sumWithSelection(selection);
    return *this;
  }

  Filtered<Filtered<T>> operator+=(Filtered<T> const& other)
  {
    return operator+=(other.getSelectedRows());
  }

  Filtered<Filtered<T>> operator*(SelectionVector const& selection)
  {
    Filtered<Filtered<T>> copy(*this);
    copy.intersectionWithSelection(selection);
    return copy;
  }

  Filtered<Filtered<T>> operator*(gsl::span<int64_t const> const& selection)
  {
    Filtered<Filtered<T>> copy(*this);
    copy.intersectionWithSelection(selection);
    return copy;
  }

  Filtered<Filtered<T>> operator*(Filtered<T> const& other)
  {
    return operator*(other.getSelectedRows());
  }

  Filtered<Filtered<T>> operator*=(SelectionVector const& selection)
  {
    this->intersectWithSelection(selection);
    return *this;
  }

  Filtered<Filtered<T>> operator*=(gsl::span<int64_t const> const& selection)
  {
    this->intersectWithSelection(selection);
    return *this;
  }

  Filtered<Filtered<T>> operator*=(Filtered<T> const& other)
  {
    return operator*=(other.getSelectedRows());
  }

  unfiltered_iterator rawIteratorAt(uint64_t i) const
  {
    auto it = unfiltered_iterator{this->cached_begin()};
    it.setCursor(i);
    return it;
  }

  auto rawSlice(uint64_t start, uint64_t end) const
  {
    SelectionVector newSelection;
    newSelection.resize(static_cast<int64_t>(end - start + 1));
    std::iota(newSelection.begin(), newSelection.end(), start);
    return self_t{{this->asArrowTable()}, std::move(newSelection), 0};
  }

  auto emptySlice() const
  {
    return self_t{{this->asArrowTable()}, SelectionVector{}, 0};
  }

  auto sliceByCached(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return doFilteredSliceByCached(this, node, value, cache);
  }

  auto sliceByCachedUnsorted(framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache) const
  {
    return doSliceByCachedUnsorted(this, node, value, cache);
  }

  template <typename T1, bool OPT, bool SORTED>
  auto sliceBy(o2::framework::PresliceBase<T1, OPT, SORTED> const& container, int value) const
  {
    if constexpr (SORTED) {
      return doFilteredSliceBy(this, container, value);
    } else {
      return doSliceBy(this, container, value);
    }
  }

 private:
  std::vector<std::shared_ptr<arrow::Table>> extractTablesFromFiltered(std::vector<Filtered<T>>& tables)
  {
    std::vector<std::shared_ptr<arrow::Table>> outTables;
    for (auto& table : tables) {
      outTables.push_back(table.asArrowTable());
    }
    return outTables;
  }
};

/// Template for building an index table to access matching rows from non-
/// joinable, but compatible tables, e.g. Collisions and ZDCs.
/// First argument is the key table (BCs for the Collisions+ZDCs case), the rest
/// are index columns defined for the required tables.
/// First index will be used by process() as the grouping
template <typename Key, typename H, typename... Ts>
struct IndexTable : Table<soa::Index<>, H, Ts...> {
  using base_t = Table<soa::Index<>, H, Ts...>;
  using table_t = base_t;
  using safe_base_t = Table<H, Ts...>;
  using indexing_t = Key;
  using first_t = typename H::binding_t;
  using rest_t = framework::pack<typename Ts::binding_t...>;
  using sources_t = originals_pack_t<Key, first_t, typename Ts::binding_t...>;

  IndexTable(std::shared_ptr<arrow::Table> table, uint64_t offset = 0)
    : base_t{table, offset}
  {
  }

  IndexTable(IndexTable const&) = default;
  IndexTable(IndexTable&&) = default;
  IndexTable& operator=(IndexTable const&) = default;
  IndexTable& operator=(IndexTable&&) = default;

  using iterator = typename base_t::template RowView<IndexTable<Key, H, Ts...>, IndexTable<Key, H, Ts...>>;
  using const_iterator = iterator;
};

template <typename T>
inline constexpr bool is_soa_index_table_v = framework::is_base_of_template_v<soa::IndexTable, T>;

template <typename T, bool APPLY>
struct SmallGroupsBase : public Filtered<T> {
  static constexpr bool applyFilters = APPLY;
  SmallGroupsBase(std::vector<std::shared_ptr<arrow::Table>>&& tables, gandiva::Selection const& selection, uint64_t offset = 0)
    : Filtered<T>(std::move(tables), selection, offset) {}

  SmallGroupsBase(std::vector<std::shared_ptr<arrow::Table>>&& tables, SelectionVector&& selection, uint64_t offset = 0)
    : Filtered<T>(std::move(tables), std::forward<SelectionVector>(selection), offset) {}

  SmallGroupsBase(std::vector<std::shared_ptr<arrow::Table>>&& tables, gsl::span<int64_t const> const& selection, uint64_t offset = 0)
    : Filtered<T>(std::move(tables), selection, offset) {}
};

template <typename T>
using SmallGroups = SmallGroupsBase<T, true>;

template <typename T>
using SmallGroupsUnfiltered = SmallGroupsBase<T, false>;

template <typename T>
struct is_smallgroups_t {
  static constexpr bool value = false;
};

template <typename T, bool F>
struct is_smallgroups_t<SmallGroupsBase<T, F>> {
  static constexpr bool value = true;
};

template <typename T>
constexpr bool is_smallgroups_v = is_smallgroups_t<T>::value;
} // namespace o2::soa

namespace o2::framework
{
using ListVector = std::vector<std::vector<int64_t>>;

std::string cutString(std::string&& str);

void sliceByColumnGeneric(
  char const* key,
  char const* target,
  std::shared_ptr<arrow::Table> const& input,
  int32_t fullSize,
  ListVector* groups,
  ListVector* unassigned = nullptr);
} // namespace o2::framework

#endif // O2_FRAMEWORK_ASOA_H_
