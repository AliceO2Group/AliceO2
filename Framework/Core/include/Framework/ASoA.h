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
#include "Framework/Kernels.h"
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/util/variant.h>
#include <arrow/compute/context.h>
#include <arrow/compute/kernel.h>
#include <gandiva/selection_vector.h>
#include <cassert>

namespace o2::soa
{

template <typename, typename = void>
constexpr bool is_index_column_v = false;

template <typename T>
constexpr bool is_index_column_v<T, std::void_t<decltype(sizeof(typename T::binding_t))>> = true;

template <typename, typename = void>
constexpr bool is_type_with_originals_v = false;

template <typename T>
constexpr bool is_type_with_originals_v<T, std::void_t<decltype(sizeof(typename T::originals))>> = true;

template <typename, typename = void>
constexpr bool is_type_with_metadata_v = false;

template <typename T>
constexpr bool is_type_with_metadata_v<T, std::void_t<decltype(sizeof(typename T::metadata))>> = true;

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

template <typename T>
constexpr auto make_originals_from_type()
{
  using decayed = std::decay_t<T>;
  if constexpr (is_type_with_originals_v<decayed>) {
    return typename decayed::originals{};
  } else if constexpr (is_type_with_originals_v<typename decayed::table_t>) {
    return typename decayed::table_t::originals{};
  } else {
    return framework::pack<decayed>{};
  }
}

template <typename T>
struct arrow_array_for {
};
template <>
struct arrow_array_for<int8_t> {
  using type = arrow::Int8Array;
};
template <>
struct arrow_array_for<uint8_t> {
  using type = arrow::UInt8Array;
};
template <>
struct arrow_array_for<int16_t> {
  using type = arrow::Int16Array;
};
template <>
struct arrow_array_for<uint16_t> {
  using type = arrow::UInt16Array;
};
template <>
struct arrow_array_for<int32_t> {
  using type = arrow::Int32Array;
};
template <>
struct arrow_array_for<int64_t> {
  using type = arrow::Int64Array;
};
template <>
struct arrow_array_for<uint32_t> {
  using type = arrow::UInt32Array;
};
template <>
struct arrow_array_for<uint64_t> {
  using type = arrow::UInt64Array;
};
template <>
struct arrow_array_for<float> {
  using type = arrow::FloatArray;
};
template <>
struct arrow_array_for<double> {
  using type = arrow::DoubleArray;
};

template <typename T>
using arrow_array_for_t = typename arrow_array_for<T>::type;

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
 public:
  /// Constructor of the column iterator. Notice how it takes a pointer
  /// to the arrow::Column (for the data store) and to the index inside
  /// it. This means that a ColumnIterator is actually only available
  /// as part of a RowView.
  ColumnIterator(arrow::Column const* column)
    : mColumn{column},
      mCurrentPos{nullptr},
      mFirstIndex{0},
      mCurrentChunk{0}
  {
    auto chunks = mColumn->data();
    auto array = std::static_pointer_cast<arrow_array_for_t<T>>(chunks->chunk(mCurrentChunk));
    mCurrent = array->raw_values();
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
    auto chunks = mColumn->data();
    auto previousArray = std::static_pointer_cast<arrow_array_for_t<T>>(chunks->chunk(mCurrentChunk));
    mFirstIndex += previousArray->length();
    mCurrentChunk++;
    auto array = std::static_pointer_cast<arrow_array_for_t<T>>(chunks->chunk(mCurrentChunk));
    mCurrent = array->raw_values() - mFirstIndex;
    mLast = mCurrent + array->length() + mFirstIndex;
  }

  void prevChunk() const
  {
    auto chunks = mColumn->data();
    auto previousArray = std::static_pointer_cast<arrow_array_for_t<T>>(chunks->chunk(mCurrentChunk));
    mFirstIndex -= previousArray->length();
    mCurrentChunk--;
    auto array = std::static_pointer_cast<arrow_array_for_t<T>>(chunks->chunk(mCurrentChunk));
    mCurrent = array->raw_values() - mFirstIndex;
    mLast = mCurrent + array->length() + mFirstIndex;
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
    mCurrentChunk = mColumn->data()->num_chunks() - 1;
    auto chunks = mColumn->data();
    auto array = std::static_pointer_cast<arrow_array_for_t<T>>(chunks->chunk(mCurrentChunk));
    assert(array.get());
    mFirstIndex = mColumn->length() - array->length();
    mCurrent = array->raw_values() - mFirstIndex;
    mLast = mCurrent + array->length() + mFirstIndex;
  }

  T const& operator*() const
  {
    if constexpr (ChunkingPolicy::chunked) {
      if (O2_BUILTIN_UNLIKELY(((mCurrent + *mCurrentPos) >= mLast))) {
        nextChunk();
      }
    }
    return *(mCurrent + *mCurrentPos);
  }

  // Move to the chunk which containts element pos
  ColumnIterator<T>& moveToPos()
  {
    // If we get outside range of the current chunk, go to the next.
    if constexpr (ChunkingPolicy::chunked) {
      while (O2_BUILTIN_UNLIKELY((mCurrent + *mCurrentPos) >= mLast)) {
        nextChunk();
      }
    }
    return *this;
  }

  // Move to the chunk which containts element pos
  ColumnIterator<T>& checkNextChunk()
  {
    if constexpr (ChunkingPolicy::chunked) {
      if (O2_BUILTIN_LIKELY((mCurrent + *mCurrentPos) <= mLast)) {
        return *this;
      }
      nextChunk();
    }
    return *this;
  }

  mutable T const* mCurrent;
  int64_t const* mCurrentPos;
  mutable T const* mLast;
  arrow::Column const* mColumn;
  mutable int mFirstIndex;
  mutable int mCurrentChunk;
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

  using persistent = std::true_type;
  using type = T;
  static constexpr const char* const& label() { return INHERIT::mLabel; }
  ColumnIterator<T> const& getIterator() const
  {
    return mColumnIterator;
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
  static constexpr const char* const& label() { return INHERIT::mLabel; }
};

template <typename INHERIT>
struct IndexColumn {
  using inherited_t = INHERIT;

  using persistent = std::false_type;
  static constexpr const char* const& label() { return INHERIT::mLabel; }
};

template <int64_t START = 0, int64_t END = -1>
struct Index : o2::soa::IndexColumn<Index<START, END>> {
  using base = o2::soa::IndexColumn<Index<START, END>>;
  constexpr inline static int64_t start = START;
  constexpr inline static int64_t end = END;

  Index() = default;
  Index(arrow::Column const* column)
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
  int64_t mMaxRow = 0;
};

struct FilteredIndexPolicy : IndexPolicyBase {
  // We use -1 in the IndexPolicyBase to indicate that the index is
  // invalid. What will validate the index is the this->setCursor()
  // which happens below which will properly setup the first index
  // by remapping the filtered index 0 to whatever unfiltered index
  // it belongs to.
  FilteredIndexPolicy(gandiva::SelectionVector* selection = nullptr, uint64_t offset = 0)
    : IndexPolicyBase{-1, offset},
      mSelection(selection),
      mMaxSelection(selection ? selection->GetNumSlots() : 0)
  {
    this->setCursor(0);
  }

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
    this->mRowIndex = mSelection ? mSelection->GetIndex(mSelectionRow) : mSelectionRow;
  }

  void moveByIndex(int64_t i)
  {
    mSelectionRow += i;
    this->mRowIndex = mSelection ? mSelection->GetIndex(mSelectionRow) : mSelectionRow;
  }

  bool operator!=(FilteredIndexPolicy const& other) const
  {
    return O2_BUILTIN_LIKELY(this->mSelectionRow != other.mSelectionRow);
  }

  bool operator==(FilteredIndexPolicy const& other) const
  {
    return O2_BUILTIN_UNLIKELY(this->mSelectionRow == other.mSelectionRow);
  }

  /// Move iterator to one after the end. Since this is a view
  /// we move the mSelectionRow to one past the view size and
  /// the mRowIndex to one past the last entry in the selection
  void moveToEnd()
  {
    this->mSelectionRow = this->mMaxSelection;
    this->mRowIndex = -1;
  }

  int64_t getSelectionRow() const
  {
    return mSelectionRow;
  }

 private:
  int64_t mSelectionRow = 0;
  int64_t mMaxSelection = 0;
  gandiva::SelectionVector* mSelection = nullptr;
};

template <typename... C>
class Table;

template <typename IP, typename... C>
struct RowViewBase : public IP, C... {
 public:
  using policy_t = IP;
  using table_t = o2::soa::Table<C...>;
  using persistent_columns_t = framework::selected_pack<is_persistent_t, C...>;
  using dynamic_columns_t = framework::selected_pack<is_dynamic_t, C...>;
  using index_columns_t = framework::selected_pack<is_index_t, C...>;
  constexpr inline static bool has_index_v = !std::is_same_v<index_columns_t, framework::pack<>>;
  using external_index_columns_t = framework::selected_pack<is_external_index_t, C...>;

  RowViewBase(std::tuple<std::pair<C*, arrow::Column*>...> const& columnIndex, IP&& policy)
    : IP{policy},
      C(std::get<std::pair<C*, arrow::Column*>>(columnIndex).second)...
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

  RowViewBase() = default;
  RowViewBase(RowViewBase<IP, C...> const& other)
    : IP{static_cast<IP>(other)},
      C(static_cast<C const&>(other))...
  {
    bindIterators(persistent_columns_t{});
    bindAllDynamicColumns(dynamic_columns_t{});
  }

  RowViewBase(RowViewBase<IP, C...>&& other)
  {
    IP::operator=(static_cast<IP>(other));
    (void(static_cast<C&>(*this) = static_cast<C const&>(other)), ...);
  }

  RowViewBase<IP, C...>& operator=(RowViewBase<IP, C...> const& other)
  {
    IP::operator=(static_cast<IP>(other));
    (void(static_cast<C&>(*this) = static_cast<C const&>(other)), ...);
    bindIterators(persistent_columns_t{});
    bindAllDynamicColumns(dynamic_columns_t{});
    return *this;
  }

  RowViewBase& operator=(RowViewBase<IP, C...>&& other)
  {
    IP::operator=(static_cast<IP>(other));
    (void(static_cast<C&>(*this) = static_cast<C const&>(other)), ...);
    return *this;
  }

  RowViewBase<IP, C...>& operator[](int64_t i)
  {
    this->setCursor(i);
    return *this;
  }

  RowViewBase<IP, C...>& operator++()
  {
    this->moveByIndex(1);
    return *this;
  }

  RowViewBase<IP, C...> operator++(int)
  {
    RowViewBase<IP, C...> copy = *this;
    this->operator++();
    return copy;
  }

  /// Allow incrementing by more than one the iterator
  RowViewBase<IP, C...> operator+(int64_t inc) const
  {
    RowViewBase<IP, C...> copy = *this;
    copy.moveByIndex(inc);
    return copy;
  }

  RowViewBase<IP, C...> operator-(int64_t dec) const
  {
    return operator+(-dec);
  }

  RowViewBase<IP, C...> const& operator*() const
  {
    return *this;
  }

  /// Inequality operator. Actual implementation
  /// depend on the policy we use for the index.
  bool operator!=(RowViewBase<IP, C...> const& other) const
  {
    return IP::operator!=(static_cast<IP>(other));
  }

  /// Equality operator. Actual implementation
  /// depend on the policy we use for the index.
  bool operator==(RowViewBase<IP, C...> const& other) const
  {
    return IP::operator==(static_cast<IP>(other));
  }

  template <typename... CL, typename TA>
  void doSetCurrentIndex(framework::pack<CL...>, TA* current)
  {
    (CL::setCurrent(current), ...);
  }

  template <typename... TA>
  void bindExternalIndices(TA*... current)
  {
    (doSetCurrentIndex(external_index_columns_t{}, current), ...);
  }

 private:
  /// Helper to move to the correct chunk, if needed.
  /// FIXME: not needed?
  template <typename... PC>
  void checkNextChunk(framework::pack<PC...> pack)
  {
    (PC::mColumnIterator.checkNextChunk(), ...);
  }

  /// Helper to move at the end of columns which actually have an iterator.
  template <typename... PC>
  void doMoveToEnd(framework::pack<PC...> pack)
  {
    (PC::mColumnIterator.moveToEnd(), ...);
  }

  /// Helper which binds all the ColumnIterators to the
  /// index of a the associated RowView
  template <typename... PC>
  auto bindIterators(framework::pack<PC...> pack)
  {
    using namespace o2::soa;
    (void(PC::mColumnIterator.mCurrentPos = &this->mRowIndex), ...);
  }

  template <typename... DC>
  auto bindAllDynamicColumns(framework::pack<DC...> pack)
  {
    using namespace o2::soa;
    (bindDynamicColumn<DC>(typename DC::bindings_t{}), ...);
    if constexpr (has_index_v) {
      this->setIndices(this->getIndices());
      this->setOffsets(this->getOffsets());
    }
  }

  template <typename DC, typename... B>
  auto bindDynamicColumn(framework::pack<B...> bindings)
  {
    DC::boundIterators = std::make_tuple(&(B::mColumnIterator)...);
  }
};

template <typename, typename = void>
constexpr bool is_type_with_policy_v = false;

template <typename T>
constexpr bool is_type_with_policy_v<T, std::void_t<decltype(sizeof(typename T::policy_t))>> = true;

template <typename... C>
using RowView = RowViewBase<DefaultIndexPolicy, C...>;

template <typename... C>
auto&& makeRowView(std::tuple<std::pair<C*, arrow::Column*>...> const& columnIndex, int64_t numRows, uint64_t offset)
{
  return std::move(RowViewBase<DefaultIndexPolicy, C...>{columnIndex, DefaultIndexPolicy{numRows, offset}});
}

template <typename... C>
using RowViewFiltered = RowViewBase<FilteredIndexPolicy, C...>;

template <typename... C>
auto&& makeRowViewFiltered(std::tuple<std::pair<C*, arrow::Column*>...> const& columnIndex, gandiva::SelectionVector* selection, uint64_t offset)
{
  return std::move(RowViewBase<FilteredIndexPolicy, C...>{columnIndex, FilteredIndexPolicy{selection, offset}});
}

struct ArrowHelpers {
  static std::shared_ptr<arrow::Table> joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables);
  static std::shared_ptr<arrow::Table> concatTables(std::vector<std::shared_ptr<arrow::Table>>&& tables);
};

/// A Table class which observes an arrow::Table and provides
/// It is templated on a set of Column / DynamicColumn types.
template <typename... C>
class Table
{
 public:
  using iterator = RowView<C...>;
  using const_iterator = RowView<C...>;
  using unfiltered_iterator = RowView<C...>;
  using unfiltered_const_iterator = RowView<C...>;
  using filtered_iterator = RowViewFiltered<C...>;
  using filtered_const_iterator = RowViewFiltered<C...>;
  using table_t = Table<C...>;
  using columns = framework::pack<C...>;
  using persistent_columns_t = framework::selected_pack<is_persistent_t, C...>;

  Table(std::shared_ptr<arrow::Table> table, uint64_t offset = 0)
    : mTable(table),
      mColumnIndex{
        std::pair<C*, arrow::Column*>{nullptr,
                                      lookupColumn<C>()}...},
      mBegin(mColumnIndex, {table->num_rows(), offset}),
      mEnd(mColumnIndex, {table->num_rows(), offset}),
      mOffset(offset)
  {
    mEnd.moveToEnd();
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

  unfiltered_iterator end()
  {
    return unfiltered_iterator{mEnd};
  }

  filtered_iterator filtered_begin(framework::expressions::Selection selection)
  {
    // Note that the FilteredIndexPolicy will never outlive the selection which
    // is held by the table, so we are safe passing the bare pointer. If it does it
    // means that the iterator on a table is outliving the table itself, which is
    // a bad idea.
    return filtered_iterator(mColumnIndex, {selection.get(), mOffset});
  }

  filtered_iterator filtered_end(framework::expressions::Selection selection)
  {
    auto end = filtered_iterator(mColumnIndex, {selection.get(), mOffset});
    end.moveToEnd();
    return end;
  }

  unfiltered_const_iterator begin() const
  {
    return unfiltered_const_iterator(mBegin);
  }

  unfiltered_const_iterator end() const
  {
    return unfiltered_const_iterator{mEnd};
  }

  /// Return a type erased arrow table backing store for / the type safe table.
  std::shared_ptr<arrow::Table> asArrowTable() const
  {
    return mTable;
  }

  /// Size of the table, in rows.
  int64_t size() const
  {
    return mTable->num_rows();
  }

  /// Bind the columns which refer to other tables
  /// to the associated tables.
  template <typename... TA>
  void bindExternalIndices(TA*... current)
  {
    mBegin.bindExternalIndices(current...);
    mEnd.bindExternalIndices(current...);
  }

 private:
  template <typename T>
  arrow::Column* lookupColumn()
  {
    if constexpr (T::persistent::value) {
      auto label = T::label();
      auto index = mTable->schema()->GetFieldIndex(label);
      if (index == -1) {
        throw std::runtime_error(std::string("Unable to find column with label ") + label);
      }
      return mTable->column(index).get();
    } else {
      return nullptr;
    }
  }
  std::shared_ptr<arrow::Table> mTable;
  /// This is a cached lookup of the column index in a given
  std::tuple<std::pair<C*, arrow::Column*>...>
    mColumnIndex;
  /// Cached begin iterator for this table.
  unfiltered_iterator mBegin;
  /// Cached end iterator for this table.
  unfiltered_iterator mEnd;
  /// Offset of the table within a larger table.
  uint64_t mOffset;
};

template <typename T>
struct PackToTable {
  static_assert(framework::always_static_assert_v<T>, "Not a pack");
};

template <typename... C>
struct PackToTable<framework::pack<C...>> {
  using table = o2::soa::Table<C...>;
};

template <typename T>
struct FilterPersistentColumns {
  static_assert(framework::always_static_assert_v<T>, "Not a soa::Table");
};

template <typename... C>
struct FilterPersistentColumns<soa::Table<C...>> {
  using columns = typename soa::Table<C...>::columns;
  using persistent_columns_pack = framework::selected_pack<is_persistent_t, C...>;
  using persistent_table_t = typename PackToTable<persistent_columns_pack>::table;
};

/// Template trait which allows to map a given
/// Table type to its O2 DataModel origin and description
template <typename INHERIT>
class TableMetadata
{
 public:
  static constexpr char const* label() { return INHERIT::mLabel; }
  static constexpr char const (&origin())[4] { return INHERIT::mOrigin; }
  static constexpr char const (&description())[16] { return INHERIT::mDescription; }
};

} // namespace o2::soa

#define DECLARE_SOA_STORE()          \
  template <typename T>              \
  struct MetadataTrait {             \
    using metadata = std::void_t<T>; \
  }

#define DECLARE_SOA_COLUMN(_Name_, _Getter_, _Type_, _Label_)                  \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {                            \
    static constexpr const char* mLabel = _Label_;                             \
    using base = o2::soa::Column<_Type_, _Name_>;                              \
    using type = _Type_;                                                       \
    using column_t = _Name_;                                                   \
    _Name_(arrow::Column const* column)                                        \
      : o2::soa::Column<_Type_, _Name_>(o2::soa::ColumnIterator<type>(column)) \
    {                                                                          \
    }                                                                          \
                                                                               \
    _Name_() = default;                                                        \
    _Name_(_Name_ const& other) = default;                                     \
    _Name_& operator=(_Name_ const& other) = default;                          \
                                                                               \
    _Type_ const _Getter_() const                                              \
    {                                                                          \
      return *mColumnIterator;                                                 \
    }                                                                          \
  };                                                                           \
  static const o2::framework::expressions::BindingNode _Getter_ { _Label_,     \
                                                                  o2::framework::expressions::selectArrowType<_Type_>() }

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
    _Name_##Id(arrow::Column const* column)                                        \
      : o2::soa::Column<_Type_, _Name_##Id>(o2::soa::ColumnIterator<type>(column)) \
    {                                                                              \
    }                                                                              \
                                                                                   \
    _Name_##Id() = default;                                                        \
    _Name_##Id(_Name_##Id const& other) = default;                                 \
    _Name_##Id& operator=(_Name_##Id const& other) = default;                      \
                                                                                   \
    type const _Getter_##Id() const                                                \
    {                                                                              \
      return *mColumnIterator;                                                     \
    }                                                                              \
                                                                                   \
    binding_t::iterator _Getter_() const                                           \
    {                                                                              \
      assert(mBinding != 0);                                                       \
      return mBinding->begin() + *mColumnIterator;                                 \
    }                                                                              \
    template <typename T>                                                          \
    bool setCurrent(T* current)                                                    \
    {                                                                              \
      if constexpr (std::is_same_v<T, binding_t>) {                                \
        assert(current != 0);                                                      \
        this->mBinding = current;                                                  \
        return true;                                                               \
      }                                                                            \
      return false;                                                                \
    }                                                                              \
    binding_t* getCurrent() { return mBinding; }                                   \
    binding_t* mBinding = nullptr;                                                 \
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
    _Name_(arrow::Column const*)                                                                                           \
    {                                                                                                                      \
    }                                                                                                                      \
    _Name_() = default;                                                                                                    \
    _Name_(_Name_ const& other) = default;                                                                                 \
    _Name_& operator=(_Name_ const& other) = default;                                                                      \
    static constexpr const char* mLabel = #_Name_;                                                                         \
    using type = typename callable_t::return_type;                                                                         \
                                                                                                                           \
    template <typename... FreeArgs>                                                                                        \
    type const _Getter_(FreeArgs... freeArgs) const                                                                        \
    {                                                                                                                      \
      return boundGetter(std::make_index_sequence<std::tuple_size_v<decltype(boundIterators)>>{}, freeArgs...);            \
    }                                                                                                                      \
                                                                                                                           \
    template <size_t... Is, typename... FreeArgs>                                                                          \
    type const boundGetter(std::integer_sequence<size_t, Is...>&& index, FreeArgs... freeArgs) const                       \
    {                                                                                                                      \
      return __VA_ARGS__((**std::get<Is>(boundIterators))..., freeArgs...);                                                \
    }                                                                                                                      \
                                                                                                                           \
    using bindings_t = typename o2::framework::pack<Bindings...>;                                                          \
    std::tuple<o2::soa::ColumnIterator<typename Bindings::type> const*...> boundIterators;                                 \
  }

#define DECLARE_SOA_TABLE(_Name_, _Origin_, _Description_, ...)        \
  using _Name_ = o2::soa::Table<__VA_ARGS__>;                          \
                                                                       \
  struct _Name_##Metadata : o2::soa::TableMetadata<_Name_##Metadata> { \
    using table_t = _Name_;                                            \
    static constexpr char const* mLabel = #_Name_;                     \
    static constexpr char const mOrigin[4] = _Origin_;                 \
    static constexpr char const mDescription[16] = _Description_;      \
  };                                                                   \
                                                                       \
  template <>                                                          \
  struct MetadataTrait<_Name_> {                                       \
    using metadata = _Name_##Metadata;                                 \
  };                                                                   \
                                                                       \
  template <>                                                          \
  struct MetadataTrait<_Name_::unfiltered_iterator> {                  \
    using metadata = _Name_##Metadata;                                 \
  };                                                                   \
                                                                       \
  template <>                                                          \
  struct MetadataTrait<o2::soa::Filtered<_Name_>::iterator> {          \
    using metadata = _Name_##Metadata;                                 \
  };

namespace o2::soa
{

template <typename... C1, typename... C2>
constexpr auto join(o2::soa::Table<C1...> const& t1, o2::soa::Table<C2...> const& t2)
{
  return o2::soa::Table<C1..., C2...>(ArrowHelpers::joinTables({t1.asArrowTable(), t2.asArrowTable()}));
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

template <typename T>
using originals_pack_t = decltype(make_originals_from_type<T>());

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
  using originals = framework::concatenated_pack_t<originals_pack_t<Ts>...>;

  template <typename... TA>
  void bindExternalIndices(TA*... externals)
  {
    base::bindExternalIndices(externals...);
  }

  using table_t = JoinBase<Ts...>;
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
};

template <typename T>
class Filtered : public T
{
 public:
  using originals = originals_pack_t<T>;
  using table_t = typename T::table_t;
  using iterator = typename table_t::filtered_iterator;
  using const_iterator = typename table_t::filtered_const_iterator;

  Filtered(std::vector<std::shared_ptr<arrow::Table>>&& tables, framework::expressions::Selection selection, uint64_t offset = 0)
    : T{std::move(tables), offset},
      mSelection{selection},
      mFilteredBegin{table_t::filtered_begin(mSelection)},
      mFilteredEnd{table_t::filtered_end(mSelection)}
  {
  }

  Filtered(std::vector<std::shared_ptr<arrow::Table>>&& tables, gandiva::NodePtr const& tree, uint64_t offset = 0)
    : T{std::move(tables), offset},
      mSelection{framework::expressions::createSelection(this->asArrowTable(),
                                                         framework::expressions::createFilter(this->asArrowTable()->schema(),
                                                                                              framework::expressions::createCondition(tree)))},
      mFilteredBegin{table_t::filtered_begin(mSelection)},
      mFilteredEnd{table_t::filtered_end(mSelection)}
  {
  }

  iterator begin()
  {
    return iterator(mFilteredBegin);
  }

  iterator end()
  {
    return iterator{mFilteredEnd};
  }

  const_iterator begin() const
  {
    return const_iterator(mFilteredBegin);
  }

  const_iterator end() const
  {
    return const_iterator{mFilteredEnd};
  }

  int64_t size() const
  {
    return mSelection->GetNumSlots();
  }

  int64_t tableSize() const
  {
    return table_t::asArrowTable()->num_rows();
  }

  framework::expressions::Selection getSelection() const
  {
    return mSelection;
  }

 private:
  framework::expressions::Selection mSelection;
  iterator mFilteredBegin;
  iterator mFilteredEnd;
};

template <typename T>
auto filter(T&& t, framework::expressions::Filter const& expr)
{
  return Filtered<T>(t.asArrowTable(), expr);
}

template <typename T>
std::vector<std::decay_t<T>> slice(T&& t, std::string const& columnName)
{
  arrow::compute::FunctionContext ctx;
  std::vector<arrow::compute::Datum> splittedDatums;
  std::vector<std::decay_t<T>> splittedTables;
  std::vector<uint64_t> offsets;
  auto status = framework::sliceByColumn(&ctx, columnName, arrow::compute::Datum(t.asArrowTable()), &splittedDatums, &offsets);
  if (status.ok() == false) {
    throw std::runtime_error("Unable to slice table");
  }
  splittedTables.reserve(splittedDatums.size());
  for (size_t ti = 0; ti < splittedDatums.size(); ++ti) {
    auto table = arrow::util::get<std::shared_ptr<arrow::Table>>(splittedDatums[ti].value);
    auto offset = offsets[ti];
    splittedTables.emplace_back(table, offset);
  }
  return splittedTables;
}

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOA_H_
