// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_ASOA_H_
#define O2_FRAMEWORK_ASOA_H_

#include "Framework/Pack.h"
#include "Framework/CheckTypes.h"
#include "Framework/FunctionalHelpers.h"
#include "Framework/CompilerBuiltins.h"
#include "Framework/Traits.h"
#include "Framework/Expressions.h"
#include "Framework/ArrowTypes.h"
#include "Framework/RuntimeError.h"
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/util/variant.h>
#include <arrow/compute/kernel.h>
#include <arrow/compute/api_aggregate.h>
#include <gandiva/selection_vector.h>
#include <cassert>
#include <fmt/format.h>

namespace o2::soa
{
template <typename... C>
auto createSchemaFromColumns(framework::pack<C...>)
{
  return std::make_shared<arrow::Schema>(std::vector<std::shared_ptr<arrow::Field>>{C::asArrowField()...});
}
using SelectionVector = std::vector<int64_t>;

template <typename, typename = void>
constexpr bool is_index_column_v = false;

template <typename T>
constexpr bool is_index_column_v<T, std::void_t<decltype(sizeof(typename T::binding_t))>> = true;

template <typename, typename = void>
constexpr bool is_type_with_originals_v = false;

template <typename T>
constexpr bool is_type_with_originals_v<T, std::void_t<decltype(sizeof(typename T::originals))>> = true;

template <typename T, typename = void>
constexpr bool is_type_with_parent_v = false;

template <typename T>
constexpr bool is_type_with_parent_v<T, std::void_t<decltype(sizeof(typename T::parent_t))>> = true;

template <typename, typename = void>
constexpr bool is_type_with_metadata_v = false;

template <typename T>
constexpr bool is_type_with_metadata_v<T, std::void_t<decltype(sizeof(typename T::metadata))>> = true;

template <typename, typename = void>
constexpr bool is_type_with_binding_v = false;

template <typename T>
constexpr bool is_type_with_binding_v<T, std::void_t<decltype(sizeof(typename T::binding_t))>> = true;

template <typename, typename = void>
constexpr bool is_type_spawnable_v = false;

template <typename T>
constexpr bool is_type_spawnable_v<T, std::void_t<decltype(sizeof(typename T::spawnable_t))>> = true;

template <typename T, typename = void>
constexpr bool is_index_table_v = false;

template <typename T>
constexpr bool is_index_table_v<T, std::void_t<decltype(sizeof(typename T::indexing_t))>> = true;

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
      mCurrentPos{nullptr},
      mFirstIndex{0},
      mCurrentChunk{0}
  {
    auto array = getCurrentArray();
    mCurrent = reinterpret_cast<T const*>(array->values()->data()) + array->offset();
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
    mCurrent = reinterpret_cast<T const*>(array->values()->data()) + array->offset() - (mFirstIndex >> SCALE_FACTOR);
    mLast = mCurrent + array->length() + (mFirstIndex >> SCALE_FACTOR);
  }

  void prevChunk() const
  {
    auto previousArray = getCurrentArray();
    mFirstIndex -= previousArray->length();

    mCurrentChunk--;
    auto array = getCurrentArray();
    mCurrent = reinterpret_cast<T const*>(array->values()->data()) + array->offset() - (mFirstIndex >> SCALE_FACTOR);
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
    mCurrent = reinterpret_cast<T const*>(array->values()->data()) + array->offset() - (mFirstIndex >> SCALE_FACTOR);
    mLast = mCurrent + array->length() + (mFirstIndex >> SCALE_FACTOR);
  }

  decltype(auto) operator*() const
  {
    if constexpr (ChunkingPolicy::chunked) {
      if (O2_BUILTIN_UNLIKELY(((mCurrent + (*mCurrentPos >> SCALE_FACTOR)) >= mLast))) {
        nextChunk();
      }
    }
    if constexpr (std::is_same_v<bool, std::decay_t<T>>) {
      // FIXME: notice that due to the type punning we cannot simply convert the
      //        masked char to a bool, because it's undefined behavior.
      // FIXME: check if shifting the masked bit to the first position is better than != 0
      return (*((char*)mCurrent + (*mCurrentPos >> SCALE_FACTOR)) & (1 << (*mCurrentPos & 0x7))) != 0;
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

  mutable T const* mCurrent;
  int64_t const* mCurrentPos;
  mutable T const* mLast;
  arrow::ChunkedArray const* mColumn;
  mutable int mFirstIndex;
  mutable int mCurrentChunk;

 private:
  /// get pointer to mCurrentChunk chunk
  auto getCurrentArray() const
  {
    std::shared_ptr<arrow::Array> chunkToUse = mColumn->chunk(mCurrentChunk);
    if constexpr (std::is_same_v<arrow_array_for_t<T>, arrow::FixedSizeListArray>) {
      chunkToUse = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(chunkToUse)->values();
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

  int64_t index() const
  {
    return index<0>();
  }

  int64_t filteredIndex() const
  {
    return index<1>();
  }

  int64_t globalIndex() const
  {
    return index<0>() + offsets<0>();
  }

  template <int N = 0>
  int64_t index() const
  {
    return *std::get<N>(rowIndices);
  }

  template <int N = 0>
  int64_t offsets() const
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

template <typename T>
using is_persistent_t = typename std::decay_t<T>::persistent::type;

template <typename T>
using is_external_index_t = typename std::conditional<is_index_column_v<T>, std::true_type, std::false_type>::type;

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
  uint64_t const index;
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

  void limitRange(int64_t start, int64_t end)
  {
    this->setCursor(start);
    if (end >= 0) {
      mMaxRow = std::min(end, mMaxRow);
    }
  }

  std::tuple<int64_t const*, int64_t const*>
    getIndices() const
  {
    return std::make_tuple(&mRowIndex, &mRowIndex);
  }

  std::tuple<uint64_t const*>
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

  bool operator!=(DefaultIndexPolicy const& other) const
  {
    return O2_BUILTIN_LIKELY(this->mRowIndex != other.mRowIndex);
  }

  bool operator==(DefaultIndexPolicy const& other) const
  {
    return O2_BUILTIN_UNLIKELY(this->mRowIndex == other.mRowIndex);
  }

  bool operator!=(RowViewSentinel const& sentinel) const
  {
    return O2_BUILTIN_LIKELY(this->mRowIndex != sentinel.index);
  }

  bool operator==(RowViewSentinel const& sentinel) const
  {
    return O2_BUILTIN_UNLIKELY(this->mRowIndex == sentinel.index);
  }

  auto size() const
  {
    return mMaxRow;
  }

  int64_t mMaxRow = 0;
};

struct FilteredIndexPolicy : IndexPolicyBase {
  // We use -1 in the IndexPolicyBase to indicate that the index is
  // invalid. What will validate the index is the this->setCursor()
  // which happens below which will properly setup the first index
  // by remapping the filtered index 0 to whatever unfiltered index
  // it belongs to.
  FilteredIndexPolicy(SelectionVector selection, uint64_t offset = 0)
    : IndexPolicyBase{-1, offset},
      mSelectedRows(selection),
      mMaxSelection(selection.size())
  {
    this->setCursor(0);
  }

  FilteredIndexPolicy() = default;
  FilteredIndexPolicy(FilteredIndexPolicy&&) = default;
  FilteredIndexPolicy(FilteredIndexPolicy const&) = default;
  FilteredIndexPolicy& operator=(FilteredIndexPolicy const&) = default;
  FilteredIndexPolicy& operator=(FilteredIndexPolicy&&) = default;

  std::tuple<int64_t const*, int64_t const*>
    getIndices() const
  {
    return std::make_tuple(&mRowIndex, &mSelectionRow);
  }

  std::tuple<uint64_t const*>
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

  bool operator!=(FilteredIndexPolicy const& other) const
  {
    return O2_BUILTIN_LIKELY(mSelectionRow != other.mSelectionRow);
  }

  bool operator==(FilteredIndexPolicy const& other) const
  {
    return O2_BUILTIN_UNLIKELY(mSelectionRow == other.mSelectionRow);
  }

  bool operator!=(RowViewSentinel const& sentinel) const
  {
    return O2_BUILTIN_LIKELY(mSelectionRow != sentinel.index);
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

  auto getSelectionRow() const
  {
    return mSelectionRow;
  }

  auto size() const
  {
    return mMaxSelection;
  }

 private:
  inline void updateRow()
  {
    this->mRowIndex = O2_BUILTIN_LIKELY(mSelectionRow < mMaxSelection) ? mSelectedRows[mSelectionRow] : -1;

    // this->mRowIndex = O2_BUILTIN_LIKELY(mMaxSelection != 0) ?
    //  (mSelectionRow < mMaxSelection ? mSelectedRows[mSelectionRow] : -1)
    //  : mSelectionRow;
  }
  SelectionVector mSelectedRows;
  int64_t mSelectionRow = 0;
  int64_t mMaxSelection = 0;
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
  using dynamic_columns_t = framework::selected_pack<is_dynamic_t, C...>;
  using index_columns_t = framework::selected_pack<is_index_t, C...>;
  constexpr inline static bool has_index_v = framework::pack_size(index_columns_t{}) > 0;
  using external_index_columns_t = framework::selected_pack<is_external_index_t, C...>;

  RowViewCore(arrow::ChunkedArray* columnData[sizeof...(C)], IP&& policy)
    : IP{policy},
      C(columnData[framework::has_type_at_v<C>(all_columns{})])...
  {
    bindIterators(persistent_columns_t{});
    bindAllDynamicColumns(dynamic_columns_t{});
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
    bindIterators(persistent_columns_t{});
    bindAllDynamicColumns(dynamic_columns_t{});
  }

  RowViewCore(RowViewCore&& other) noexcept
  {
    IP::operator=(static_cast<IP&&>(other));
    (void(static_cast<C&>(*this) = static_cast<C&&>(other)), ...);
    bindIterators(persistent_columns_t{});
    bindAllDynamicColumns(dynamic_columns_t{});
  }

  RowViewCore& operator=(RowViewCore const& other)
  {
    IP::operator=(static_cast<IP const&>(other));
    (void(static_cast<C&>(*this) = static_cast<C const&>(other)), ...);
    bindIterators(persistent_columns_t{});
    bindAllDynamicColumns(dynamic_columns_t{});
    return *this;
  }

  RowViewCore& operator=(RowViewCore&& other) noexcept
  {
    IP::operator=(static_cast<IP&&>(other));
    (void(static_cast<C&>(*this) = static_cast<C&&>(other)), ...);
    return *this;
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

  /// Inequality operator. Actual implementation
  /// depend on the policy we use for the index.
  using IP::operator!=;

  /// Equality operator. Actual implementation
  /// depend on the policy we use for the index.
  using IP::operator==;

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
    return std::vector<void*>{static_cast<Cs const&>(*this).getCurrentRaw()...};
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
  void doSetCurrentIndexRaw(framework::pack<Cs...> p, std::vector<void*>&& ptrs)
  {
    (Cs::setCurrentRaw(ptrs[framework::has_type_at_v<Cs>(p)]), ...);
  }

  void bindExternalIndicesRaw(std::vector<void*>&& ptrs)
  {
    doSetCurrentIndexRaw(external_index_columns_t{}, std::forward<std::vector<void*>>(ptrs));
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
  template <typename... PC>
  auto bindIterators(framework::pack<PC...>)
  {
    using namespace o2::soa;
    (void(PC::mColumnIterator.mCurrentPos = &this->mRowIndex), ...);
  }

  template <typename... DC>
  auto bindAllDynamicColumns(framework::pack<DC...>)
  {
    using namespace o2::soa;
    (bindDynamicColumn<DC>(typename DC::bindings_t{}), ...);
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

template <typename T>
using is_soa_iterator_t = typename framework::is_base_of_template<RowViewCore, T>;

template <typename T>
constexpr bool is_soa_iterator_v()
{
  return is_soa_iterator_t<T>::value || framework::is_specialization<T, RowViewCore>::value;
}

template <typename T>
using is_soa_table_t = typename framework::is_specialization<T, soa::Table>;

template <typename T>
using is_soa_table_like_t = typename framework::is_base_of_template<soa::Table, T>;

/// Helper function to extract bound indices
template <typename... Is>
static constexpr auto extractBindings(framework::pack<Is...>)
{
  return framework::pack<typename Is::binding_t...>{};
}

template <typename T>
class Filtered;

template <typename T>
auto select(T const& t, framework::expressions::Filter&& f)
{
  return Filtered<T>({t.asArrowTable()}, framework::expressions::createExpressionTree(
                                           framework::expressions::createOperations(f),
                                           t.asArrowTable()->schema()));
}

namespace
{
auto getSliceFor(int value, char const* key, std::shared_ptr<arrow::Table> const& input, std::shared_ptr<arrow::Table>& output, uint64_t& offset)
{
  arrow::Datum value_counts;
  auto options = arrow::compute::CountOptions::Defaults();
  ARROW_ASSIGN_OR_RAISE(value_counts,
                        arrow::compute::CallFunction("value_counts", {input->GetColumnByName(key)},
                                                     &options));
  auto pair = static_cast<arrow::StructArray>(value_counts.array());
  auto values = static_cast<arrow::NumericArray<arrow::Int32Type>>(pair.field(0)->data());
  auto counts = static_cast<arrow::NumericArray<arrow::Int64Type>>(pair.field(1)->data());

  int slice;
  for (slice = 0; slice < values.length(); ++slice) {
    if (values.Value(slice) == value) {
      offset = slice;
      output = input->Slice(slice, counts.Value(slice));
      return arrow::Status::OK();
    }
  }
  output = input->Slice(0, 0);
  return arrow::Status::OK();
}
} // namespace

template <typename T>
auto sliceBy(T const& t, framework::expressions::BindingNode const& node, int value)
{
  uint64_t offset = 0;
  std::shared_ptr<arrow::Table> result = nullptr;
  auto status = getSliceFor(value, node.name.c_str(), t.asArrowTable(), result, offset);
  if (status.ok()) {
    return T({result}, offset);
  }
  throw std::runtime_error("Failed to slice table");
}

/// A Table class which observes an arrow::Table and provides
/// It is templated on a set of Column / DynamicColumn types.
template <typename... C>
class Table
{
 public:
  using table_t = Table<C...>;
  using columns = framework::pack<C...>;
  using column_types = framework::pack<typename C::type...>;
  using persistent_columns_t = framework::selected_pack<is_persistent_t, C...>;
  using external_index_columns_t = framework::selected_pack<is_external_index_t, C...>;

  template <typename IP, typename Parent, typename... T>
  struct RowViewBase : public RowViewCore<IP, C...> {

    using external_index_columns_t = framework::selected_pack<is_external_index_t, C...>;
    using bindings_pack_t = decltype(extractBindings(external_index_columns_t{}));
    using parent_t = Parent;
    using originals = originals_pack_t<T...>;

    RowViewBase(arrow::ChunkedArray* columnData[sizeof...(C)], IP&& policy)
      : RowViewCore<IP, C...>(columnData, std::forward<decltype(policy)>(policy))
    {
    }

    template <typename Tbl = table_t>
    RowViewBase(RowViewBase<IP, Tbl, Tbl> const& other)
      : RowViewCore<IP, C...>(other)
    {
    }

    template <typename Tbl = table_t>
    RowViewBase(RowViewBase<IP, Tbl, Tbl>&& other) noexcept
      : RowViewCore<IP, C...>(other)
    {
    }

    RowViewBase() = default;
    RowViewBase(RowViewBase const&) = default;
    RowViewBase(RowViewBase&&) = default;

    RowViewBase& operator=(RowViewBase const&) = default;
    RowViewBase& operator=(RowViewBase&&) = default;

    RowViewBase& operator=(RowViewSentinel const& other)
    {
      this->mRowIndex = other.index;
      return *this;
    }

    void matchTo(RowViewBase const& other)
    {
      this->mRowIndex = other.mRowIndex;
    }

    template <typename TI>
    auto getId() const
    {
      if constexpr (framework::has_type_v<std::decay_t<TI>, bindings_pack_t>) {
        constexpr auto idx = framework::has_type_at_v<std::decay_t<TI>>(bindings_pack_t{});
        return framework::pack_element_t<idx, external_index_columns_t>::getId();
      } else if constexpr (std::is_same_v<std::decay_t<TI>, Parent>) {
        return this->globalIndex();
      } else {
        return static_cast<int32_t>(-1);
      }
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
      mEnd{static_cast<uint64_t>(table->num_rows())},
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
    }
  }

  /// FIXME: this is to be able to construct a Filtered without explicit Join
  ///        so that Filtered<Table1,Table2, ...> always means a Join which
  ///        may or may not be a problem later
  Table(std::vector<std::shared_ptr<arrow::Table>>&& tables, uint64_t offset = 0)
    : Table(ArrowHelpers::joinTables(std::move(tables)), offset)
  {
  }

  unfiltered_iterator begin()
  {
    return unfiltered_iterator(mBegin);
  }

  RowViewSentinel end()
  {
    return RowViewSentinel{mEnd};
  }

  filtered_iterator filtered_begin(SelectionVector selection)
  {
    // Note that the FilteredIndexPolicy will never outlive the selection which
    // is held by the table, so we are safe passing the bare pointer. If it does it
    // means that the iterator on a table is outliving the table itself, which is
    // a bad idea.
    return filtered_iterator(mColumnChunks, {selection, mOffset});
  }

  unfiltered_iterator iteratorAt(uint64_t i) const
  {
    return mBegin + (i - mOffset);
  }

  unfiltered_const_iterator begin() const
  {
    return unfiltered_const_iterator(mBegin);
  }

  RowViewSentinel end() const
  {
    return RowViewSentinel{mEnd};
  }

  /// Return a type erased arrow table backing store for / the type safe table.
  std::shared_ptr<arrow::Table> asArrowTable() const
  {
    return mTable;
  }
  /// Return offset
  auto offset() const
  {
    return mOffset;
  }
  /// Size of the table, in rows.
  int64_t size() const
  {
    return mTable->num_rows();
  }

  int64_t tableSize() const
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

  void bindExternalIndicesRaw(std::vector<void*>&& ptrs)
  {
    mBegin.bindExternalIndicesRaw(std::forward<std::vector<void*>>(ptrs));
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

  auto select(framework::expressions::Filter&& f) const
  {
    auto t = o2::soa::select(*this, std::forward<framework::expressions::Filter>(f));
    copyIndexBindings(t);
    return t;
  }

  auto sliceBy(framework::expressions::BindingNode const& node, int value) const
  {
    auto t = o2::soa::sliceBy(*this, node, value);
    copyIndexBindings(t);
    return t;
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
      auto index = mTable->schema()->GetAllFieldIndices(label);
      if (index.empty() == true) {
        throw o2::framework::runtime_error_f("Unable to find column with label %s", label);
      }
      return mTable->column(index[0]).get();
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

/// Template trait which allows to map a given
/// Table type to its O2 DataModel origin and description
template <typename INHERIT>
class TableMetadata
{
 public:
  static constexpr char const* tableLabel() { return INHERIT::mLabel; }
  static constexpr char const (&origin())[4] { return INHERIT::mOrigin; }
  static constexpr char const (&description())[16] { return INHERIT::mDescription; }
  static std::string sourceSpec() { return fmt::format("{}/{}/{}", INHERIT::mLabel, INHERIT::mOrigin, INHERIT::mDescription); };
};

/// Helper template to define universal join
template <typename Key, typename H, typename... Ts>
struct IndexTable;

template <typename... C1, typename... C2>
constexpr auto joinTables(o2::soa::Table<C1...> const& t1, o2::soa::Table<C2...> const& t2)
{
  return o2::soa::Table<C1..., C2...>(ArrowHelpers::joinTables({t1.asArrowTable(), t2.asArrowTable()}));
}

// special case for appending an index
template <typename... C1, typename Key, typename H, typename... C2>
constexpr auto joinTables(o2::soa::Table<C1...> const& t1, o2::soa::IndexTable<Key, H, C2...> const& t2)
{
  return joinTables(t1, o2::soa::Table<H, C2...>{t2.asArrowTable()});
}

template <typename T, typename... C, typename... O>
constexpr auto joinLeft(T const& t1, o2::soa::Table<C...> const& t2, framework::pack<O...>)
{
  return typename o2::soa::TableWrap<O..., o2::soa::Table<C...>>::table_t(ArrowHelpers::joinTables({t1.asArrowTable(), t2.asArrowTable()}));
}

template <typename T, typename... C, typename... O>
constexpr auto joinRight(o2::soa::Table<C...> const& t1, T const& t2, framework::pack<O...>)
{
  return typename o2::soa::TableWrap<o2::soa::Table<C...>, O...>::table_t(ArrowHelpers::joinTables({t1.asArrowTable(), t2.asArrowTable()}));
}

template <typename T1, typename T2, typename... O1, typename... O2>
constexpr auto joinBoth(T1 const& t1, T2 const& t2, framework::pack<O1...>, framework::pack<O2...>)
{
  return typename o2::soa::TableWrap<O1..., O2...>::table_t(ArrowHelpers::joinTables({t1.asArrowTable(), t2.asArrowTable()}));
}

template <typename T1, typename T2>
constexpr auto join(T1 const& t1, T2 const& t2)
{
  if constexpr (soa::is_type_with_originals_v<T1>) {
    if constexpr (soa::is_type_with_originals_v<T2>) {
      return joinBoth(t1, t2, typename T1::originals{}, typename T2::originals{});
    } else {
      return joinLeft(t1, t2, typename T1::originals{});
    }
  } else {
    if constexpr (soa::is_type_with_originals_v<T2>) {
      return joinRight(t1, t2, typename T2::originals{});
    } else {
      return joinTables(t1, t2);
    }
  }
}

template <typename T1, typename T2, typename... Ts>
constexpr auto join(T1 const& t1, T2 const& t2, Ts const&... ts)
{
  return join(t1, join(t2, ts...));
}

template <typename T1, typename T2>
constexpr auto concat(T1&& t1, T2&& t2)
{
  using table_t = typename PackToTable<framework::intersected_pack_t<typename T1::columns, typename T2::columns>>::table;
  return table_t(ArrowHelpers::concatTables({t1.asArrowTable(), t2.asArrowTable()}));
}

template <typename... Ts>
using JoinBase = decltype(join(std::declval<Ts>()...));

template <typename T1, typename T2>
using ConcatBase = decltype(concat(std::declval<T1>(), std::declval<T2>()));

template <typename T1, typename T2>
constexpr auto is_binding_compatible_v()
{
  return framework::pack_size(
           framework::intersected_pack_t<originals_pack_t<T1>, originals_pack_t<T2>>{}) > 0;
}

} // namespace o2::soa

#define DECLARE_SOA_STORE()          \
  template <typename T>              \
  struct MetadataTrait {             \
    using metadata = std::void_t<T>; \
  }

#define DECLARE_SOA_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_)             \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {                            \
    static constexpr const char* mLabel = _Label_;                             \
    using base = o2::soa::Column<_Type_, _Name_>;                              \
    using type = _Type_;                                                       \
    using column_t = _Name_;                                                   \
    _Name_(arrow::ChunkedArray const* column)                                  \
      : o2::soa::Column<_Type_, _Name_>(o2::soa::ColumnIterator<type>(column)) \
    {                                                                          \
    }                                                                          \
                                                                               \
    _Name_() = default;                                                        \
    _Name_(_Name_ const& other) = default;                                     \
    _Name_& operator=(_Name_ const& other) = default;                          \
                                                                               \
    decltype(auto) _Getter_() const                                            \
    {                                                                          \
      return *mColumnIterator;                                                 \
    }                                                                          \
  };                                                                           \
  static const o2::framework::expressions::BindingNode _Getter_ { _Label_,     \
                                                                  o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_COLUMN(_Name_, _Getter_, _Type_) \
  DECLARE_SOA_COLUMN_FULL(_Name_, _Getter_, _Type_, "f" #_Name_)

/// An 'expression' column. i.e. a column that can be calculated from other
/// columns with gandiva based on supplied C++ expression.
#define DECLARE_SOA_EXPRESSION_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_, _Expression_) \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {                                         \
    static constexpr const char* mLabel = _Label_;                                          \
    using base = o2::soa::Column<_Type_, _Name_>;                                           \
    using type = _Type_;                                                                    \
    using column_t = _Name_;                                                                \
    using spawnable_t = std::true_type;                                                     \
    _Name_(arrow::ChunkedArray const* column)                                               \
      : o2::soa::Column<_Type_, _Name_>(o2::soa::ColumnIterator<type>(column))              \
    {                                                                                       \
    }                                                                                       \
                                                                                            \
    _Name_() = default;                                                                     \
    _Name_(_Name_ const& other) = default;                                                  \
    _Name_& operator=(_Name_ const& other) = default;                                       \
                                                                                            \
    decltype(auto) _Getter_() const                                                         \
    {                                                                                       \
      return *mColumnIterator;                                                              \
    }                                                                                       \
    static o2::framework::expressions::Projector Projector()                                \
    {                                                                                       \
      return _Expression_;                                                                  \
    }                                                                                       \
  };                                                                                        \
  static const o2::framework::expressions::BindingNode _Getter_ { _Label_,                  \
                                                                  o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_EXPRESSION_COLUMN(_Name_, _Getter_, _Type_, _Expression_) \
  DECLARE_SOA_EXPRESSION_COLUMN_FULL(_Name_, _Getter_, _Type_, "f" #_Name_, _Expression_);

/// An index column is a column of indices to elements / of another table named
/// _Name_##s. The column name will be _Name_##Id and will always be stored in
/// "f"#_Name__#Id . It will also have two special methods, setCurrent(...)
/// and getCurrent(...) which allow you to set / retrieve associated table.
/// It also exposes a getter _Getter_ which allows you to retrieve the pointed
/// object.
/// Notice how in order to define an index column, the table it points
/// to **must** be already declared. This is therefore only
/// useful to express child -> parent relationships. In case one
/// needs to go from parent to child, the only way is to either have
/// a separate "association" with the two indices, or to use the standard
/// grouping mechanism of AnalysisTask.
#define DECLARE_SOA_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Table_, _Label_)  \
  struct _Name_##Id : o2::soa::Column<_Type_, _Name_##Id> {                        \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");      \
    static constexpr const char* mLabel = _Label_;                                 \
    using base = o2::soa::Column<_Type_, _Name_##Id>;                              \
    using type = _Type_;                                                           \
    using column_t = _Name_##Id;                                                   \
    using binding_t = _Table_;                                                     \
    _Name_##Id(arrow::ChunkedArray const* column)                                  \
      : o2::soa::Column<_Type_, _Name_##Id>(o2::soa::ColumnIterator<type>(column)) \
    {                                                                              \
    }                                                                              \
                                                                                   \
    _Name_##Id() = default;                                                        \
    _Name_##Id(_Name_##Id const& other) = default;                                 \
    _Name_##Id& operator=(_Name_##Id const& other) = default;                      \
    type inline getId() const                                                      \
    {                                                                              \
      return _Getter_##Id();                                                       \
    }                                                                              \
                                                                                   \
    type _Getter_##Id() const                                                      \
    {                                                                              \
      return *mColumnIterator;                                                     \
    }                                                                              \
                                                                                   \
    bool has_##_Getter_() const                                                    \
    {                                                                              \
      return *mColumnIterator >= 0;                                                \
    }                                                                              \
                                                                                   \
    template <typename T>                                                          \
    auto _Getter_##_as() const                                                     \
    {                                                                              \
      assert(mBinding != nullptr);                                                 \
      return static_cast<T*>(mBinding)->begin() + *mColumnIterator;                \
    }                                                                              \
                                                                                   \
    auto _Getter_() const                                                          \
    {                                                                              \
      return _Getter_##_as<binding_t>();                                           \
    }                                                                              \
                                                                                   \
    template <typename T>                                                          \
    bool setCurrent(T* current)                                                    \
    {                                                                              \
      if constexpr (o2::soa::is_binding_compatible_v<T, binding_t>()) {            \
        assert(current != nullptr);                                                \
        this->mBinding = current;                                                  \
        return true;                                                               \
      }                                                                            \
      return false;                                                                \
    }                                                                              \
                                                                                   \
    bool setCurrentRaw(void* current)                                              \
    {                                                                              \
      this->mBinding = current;                                                    \
      return true;                                                                 \
    }                                                                              \
    binding_t* getCurrent() const { return static_cast<binding_t*>(mBinding); }    \
    void* getCurrentRaw() const { return mBinding; }                               \
    void* mBinding = nullptr;                                                      \
  };                                                                               \
  static const o2::framework::expressions::BindingNode _Getter_##Id { _Label_,     \
                                                                      o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, _Name_##s, "f" #_Name_ "sID")
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

#define DECLARE_SOA_TABLE_FULL(_Name_, _Label_, _Origin_, _Description_, ...) \
  using _Name_ = o2::soa::Table<__VA_ARGS__>;                                 \
                                                                              \
  struct _Name_##Metadata : o2::soa::TableMetadata<_Name_##Metadata> {        \
    using table_t = _Name_;                                                   \
    static constexpr char const* mLabel = _Label_;                            \
    static constexpr char const mOrigin[4] = _Origin_;                        \
    static constexpr char const mDescription[16] = _Description_;             \
  };                                                                          \
                                                                              \
  template <>                                                                 \
  struct MetadataTrait<_Name_> {                                              \
    using metadata = _Name_##Metadata;                                        \
  };                                                                          \
                                                                              \
  template <>                                                                 \
  struct MetadataTrait<_Name_::unfiltered_iterator> {                         \
    using metadata = _Name_##Metadata;                                        \
  };

#define DECLARE_SOA_TABLE(_Name_, _Origin_, _Description_, ...) \
  DECLARE_SOA_TABLE_FULL(_Name_, #_Name_, _Origin_, _Description_, __VA_ARGS__);

#define DECLARE_SOA_EXTENDED_TABLE_FULL(_Name_, _Table_, _Origin_, _Description_, ...)   \
  using _Name_##Extension = o2::soa::Table<__VA_ARGS__>;                                 \
  using _Name_ = o2::soa::Join<_Name_##Extension, _Table_>;                              \
                                                                                         \
  struct _Name_##ExtensionMetadata : o2::soa::TableMetadata<_Name_##ExtensionMetadata> { \
    using table_t = _Name_##Extension;                                                   \
    using base_table_t = typename _Table_::table_t;                                      \
    using expression_pack_t = framework::pack<__VA_ARGS__>;                              \
    using originals = soa::originals_pack_t<_Table_>;                                    \
    static constexpr char const* mLabel = #_Name_ "Extension";                           \
    static constexpr char const mOrigin[4] = _Origin_;                                   \
    static constexpr char const mDescription[16] = _Description_;                        \
  };                                                                                     \
                                                                                         \
  template <>                                                                            \
  struct MetadataTrait<_Name_##Extension> {                                              \
    using metadata = _Name_##ExtensionMetadata;                                          \
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
    static constexpr char const* mLabel = #_Name_;                                                                               \
    static constexpr char const mOrigin[4] = _Origin_;                                                                           \
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
template <typename... Ts>
struct Join : JoinBase<Ts...> {
  Join(std::vector<std::shared_ptr<arrow::Table>>&& tables, uint64_t offset = 0)
    : JoinBase<Ts...>{ArrowHelpers::joinTables(std::move(tables)), offset} {}

  template <typename... ATs>
  Join(uint64_t offset, std::shared_ptr<arrow::Table> t1, std::shared_ptr<arrow::Table> t2, ATs... ts)
    : Join<Ts...>(std::vector<std::shared_ptr<arrow::Table>>{t1, t2, ts...}, offset)
  {
  }

  using base = JoinBase<Ts...>;
  using originals = originals_pack_t<Ts...>;

  template <typename... TA>
  void bindExternalIndices(TA*... externals)
  {
    base::bindExternalIndices(externals...);
  }

  using table_t = base;
  using persistent_columns_t = typename table_t::persistent_columns_t;
  using iterator = typename table_t::template RowView<Join<Ts...>, Ts...>;
  using const_iterator = iterator;
  using filtered_iterator = typename table_t::template RowViewFiltered<Join<Ts...>, Ts...>;
  using filtered_const_iterator = filtered_iterator;
};

template <typename T1, typename T2>
struct Concat : ConcatBase<T1, T2> {
  Concat(std::shared_ptr<arrow::Table> t1, std::shared_ptr<arrow::Table> t2, uint64_t offset = 0)
    : ConcatBase<T1, T2>{ArrowHelpers::concatTables({t1, t2}), offset} {}
  Concat(std::vector<std::shared_ptr<arrow::Table>> tables, uint64_t offset = 0)
    : ConcatBase<T1, T2>{ArrowHelpers::concatTables(std::move(tables)), offset} {}

  using base = ConcatBase<T1, T2>;
  using originals = framework::concatenated_pack_t<originals_pack_t<T1>, originals_pack_t<T2>>;

  template <typename... TA>
  void bindExternalIndices(TA*... externals)
  {
    base::bindExternalIndices(externals...);
  }

  // FIXME: can be remove when we do the same treatment we did for Join to Concatenate
  using left_t = T1;
  using right_t = T2;
  using table_t = ConcatBase<T1, T2>;
  using persistent_columns_t = typename table_t::persistent_columns_t;

  using iterator = typename table_t::template RowView<Concat<T1, T2>, T1, T2>;
  using filtered_iterator = typename table_t::template RowViewFiltered<Concat<T1, T2>, T1, T2>;
};

template <typename T>
using is_soa_join_t = typename framework::is_specialization<T, soa::Join>;

template <typename T>
using is_soa_concat_t = typename framework::is_specialization<T, soa::Concat>;

template <typename T>
class FilteredPolicy : public T
{
 public:
  using originals = originals_pack_t<T>;
  using table_t = typename T::table_t;
  using persistent_columns_t = typename T::persistent_columns_t;
  using external_index_columns_t = typename T::external_index_columns_t;

  template <typename P, typename... Os>
  constexpr static auto make_it(framework::pack<Os...> const&)
  {
    return typename table_t::template RowViewFiltered<P, Os...>{};
  }
  using iterator = decltype(make_it<FilteredPolicy<T>>(originals{}));
  using const_iterator = iterator;

  FilteredPolicy(std::vector<std::shared_ptr<arrow::Table>>&& tables, SelectionVector&& selection, uint64_t offset = 0)
    : T{std::move(tables), offset},
      mSelectedRows{std::forward<SelectionVector>(selection)}
  {
    resetRanges();
  }

  FilteredPolicy(std::vector<std::shared_ptr<arrow::Table>>&& tables, framework::expressions::Selection selection, uint64_t offset = 0)
    : T{std::move(tables), offset},
      mSelectedRows{copySelection(selection)}
  {
    resetRanges();
  }

  FilteredPolicy(std::vector<std::shared_ptr<arrow::Table>>&& tables, gandiva::NodePtr const& tree, uint64_t offset = 0)
    : T{std::move(tables), offset},
      mSelectedRows{copySelection(framework::expressions::createSelection(this->asArrowTable(),
                                                                          framework::expressions::createFilter(this->asArrowTable()->schema(),
                                                                                                               framework::expressions::makeCondition(tree))))}
  {
    resetRanges();
  }

  iterator begin()
  {
    return iterator(mFilteredBegin);
  }

  RowViewSentinel end()
  {
    return RowViewSentinel{*mFilteredEnd};
  }

  const_iterator begin() const
  {
    return const_iterator(mFilteredBegin);
  }

  RowViewSentinel end() const
  {
    return RowViewSentinel{*mFilteredEnd};
  }

  iterator iteratorAt(uint64_t i)
  {
    return mFilteredBegin + (i - this->mOffset);
  }

  int64_t size() const
  {
    return mSelectedRows.size();
  }

  int64_t tableSize() const
  {
    return table_t::asArrowTable()->num_rows();
  }

  SelectionVector const& getSelectedRows() const
  {
    return mSelectedRows;
  }

  static inline SelectionVector copySelection(framework::expressions::Selection const& sel)
  {
    SelectionVector rows;
    for (auto i = 0; i < sel->GetNumSlots(); ++i) {
      rows.push_back(sel->GetIndex(i));
    }
    return rows;
  }

  /// Bind the columns which refer to other tables
  /// to the associated tables.
  template <typename... TA>
  void bindExternalIndices(TA*... current)
  {
    table_t::bindExternalIndices(current...);
    mFilteredBegin.bindExternalIndices(current...);
  }

  void bindExternalIndicesRaw(std::vector<void*>&& ptrs)
  {
    mFilteredBegin.bindExternalIndicesRaw(std::forward<std::vector<void*>>(ptrs));
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

 protected:
  void sumWithSelection(SelectionVector const& selection)
  {
    SelectionVector rowsUnion;
    std::set_union(mSelectedRows.begin(), mSelectedRows.end(), selection.begin(), selection.end(), std::back_inserter(rowsUnion));
    mSelectedRows = rowsUnion;
    resetRanges();
  }

  void intersectWithSelection(SelectionVector const& selection)
  {
    SelectionVector intersection;
    std::set_intersection(mSelectedRows.begin(), mSelectedRows.end(), selection.begin(), selection.end(), std::back_inserter(intersection));
    mSelectedRows = intersection;
    resetRanges();
  }

 private:
  void resetRanges()
  {
    mFilteredEnd.reset(new RowViewSentinel{mSelectedRows.size()});
    if (tableSize() == 0) {
      mFilteredBegin = *mFilteredEnd;
    } else {
      mFilteredBegin = table_t::filtered_begin(mSelectedRows);
    }
  }

  SelectionVector mSelectedRows;
  iterator mFilteredBegin;
  std::shared_ptr<RowViewSentinel> mFilteredEnd;
};

template <typename T>
class Filtered : public FilteredPolicy<T>
{
 public:
  Filtered(std::vector<std::shared_ptr<arrow::Table>>&& tables, SelectionVector&& selection, uint64_t offset = 0)
    : FilteredPolicy<T>(std::move(tables), std::forward<SelectionVector>(selection), offset) {}

  Filtered(std::vector<std::shared_ptr<arrow::Table>>&& tables, framework::expressions::Selection selection, uint64_t offset = 0)
    : FilteredPolicy<T>(std::move(tables), selection, offset) {}

  Filtered(std::vector<std::shared_ptr<arrow::Table>>&& tables, gandiva::NodePtr const& tree, uint64_t offset = 0)
    : FilteredPolicy<T>(std::move(tables), tree, offset) {}

  Filtered<T> operator+(SelectionVector const& selection)
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

  Filtered<T> operator*(Filtered<T> const& other)
  {
    return operator*(other.getSelectedRows());
  }

  Filtered<T> operator*=(SelectionVector const& selection)
  {
    this->intersectWithSelection(selection);
    return *this;
  }

  Filtered<T> operator*=(Filtered<T> const& other)
  {
    return operator*=(other.getSelectedRows());
  }
};

template <typename T>
class Filtered<Filtered<T>> : public FilteredPolicy<typename T::table_t>
{
 public:
  using table_t = typename FilteredPolicy<typename T::table_t>::table_t;

  Filtered(std::vector<Filtered<T>>&& tables, SelectionVector&& selection, uint64_t offset = 0)
    : FilteredPolicy<typename T::table_t>(std::move(extractTablesFromFiltered(std::move(tables))), std::forward<SelectionVector>(selection), offset)
  {
    for (auto& table : tables) {
      *this *= table;
    }
  }

  Filtered(std::vector<Filtered<T>>&& tables, framework::expressions::Selection selection, uint64_t offset = 0)
    : FilteredPolicy<typename T::table_t>(std::move(extractTablesFromFiltered(std::move(tables))), selection, offset)
  {
    for (auto& table : tables) {
      *this *= table;
    }
  }

  Filtered(std::vector<Filtered<T>>&& tables, gandiva::NodePtr const& tree, uint64_t offset = 0)
    : FilteredPolicy<typename T::table_t>(std::move(extractTablesFromFiltered(std::move(tables))), tree, offset)
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

  Filtered<Filtered<T>> operator+(Filtered<T> const& other)
  {
    return operator+(other.getSelectedRows());
  }

  Filtered<Filtered<T>> operator+=(SelectionVector const& selection)
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

  Filtered<Filtered<T>> operator*(Filtered<T> const& other)
  {
    return operator*(other.getSelectedRows());
  }

  Filtered<Filtered<T>> operator*=(SelectionVector const& selection)
  {
    this->intersectWithSelection(selection);
    return *this;
  }

  Filtered<Filtered<T>> operator*=(Filtered<T> const& other)
  {
    return operator*=(other.getSelectedRows());
  }

 private:
  std::vector<std::shared_ptr<arrow::Table>> extractTablesFromFiltered(std::vector<Filtered<T>>&& tables)
  {
    std::vector<std::shared_ptr<arrow::Table>> outTables;
    for (auto& table : tables) {
      outTables.push_back(table.asArrowTable());
    }
    return outTables;
  }
};

template <typename T>
using is_soa_filtered_t = typename framework::is_base_of_template<soa::FilteredPolicy, T>;

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
  using sources_t = framework::pack<Key, typename H::binding_t, typename Ts::binding_t...>;

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
using is_soa_index_table_t = typename framework::is_base_of_template<soa::IndexTable, T>;
} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOA_H_
