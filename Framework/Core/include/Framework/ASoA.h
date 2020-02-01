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

#include "Framework/FunctionalHelpers.h"
#include "Framework/CompilerBuiltins.h"
#include "Framework/Traits.h"
#include "Framework/Expressions.h"
#include <arrow/table.h>
#include <arrow/array.h>
#include <gandiva/selection_vector.h>
#include <cassert>

namespace o2::soa
{

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
  Index(arrow::Column const*)
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

  template <int N = 0>
  int64_t index() const
  {
    return *std::get<N>(rowIndices);
  }

  void setIndices(std::tuple<int64_t const*, int64_t const*> indices)
  {
    rowIndices = indices;
  }

  static constexpr const char* mLabel = "Index";
  using type = int64_t;

  using bindings_t = typename o2::framework::pack<>;
  std::tuple<> boundIterators;
  std::tuple<int64_t const*, int64_t const*> rowIndices;
};

template <typename T>
using is_dynamic_t = framework::is_specialization<typename T::base, DynamicColumn>;

template <typename T>
using is_persistent_t = typename std::decay_t<T>::persistent::type;

template <typename T, template <auto...> class Ref>
struct is_index : std::false_type {
};

template <template <auto...> class Ref, auto... Args>
struct is_index<Ref<Args...>, Ref> : std::true_type {
};

template <typename T>
using is_index_t = is_index<T, Index>;

struct IndexPolicyBase {
  int64_t mRowIndex = 0;
};

struct DefaultIndexPolicy : IndexPolicyBase {
  /// Needed to be able to copy the policy
  DefaultIndexPolicy() = default;
  DefaultIndexPolicy(DefaultIndexPolicy&&) = default;
  DefaultIndexPolicy(DefaultIndexPolicy const&) = default;
  DefaultIndexPolicy& operator=(DefaultIndexPolicy const&) = default;
  DefaultIndexPolicy& operator=(DefaultIndexPolicy&&) = default;

  /// mMaxRow is one behind the last row, so effectively equal to the number of
  /// rows @a nRows.
  DefaultIndexPolicy(int64_t nRows)
    : mMaxRow(nRows)
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
  FilteredIndexPolicy()
    : mSelection(nullptr)
  {
  }

  FilteredIndexPolicy(gandiva::SelectionVector* selection)
    : mSelection(selection),
      mMaxSelection(selection->GetNumSlots())
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
    this->mRowIndex = mSelection->GetIndex(mSelectionRow);
  }

  void moveByIndex(int64_t i)
  {
    mSelectionRow += i;
    this->mRowIndex = mSelection->GetIndex(mSelectionRow);
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
    }
  }

  template <typename DC, typename... B>
  auto bindDynamicColumn(framework::pack<B...> bindings)
  {
    DC::boundIterators = std::make_tuple(&(B::mColumnIterator)...);
  }
};

template <typename... C>
using RowView = RowViewBase<DefaultIndexPolicy, C...>;

template <typename... C>
auto&& makeRowView(std::tuple<std::pair<C*, arrow::Column*>...> const& columnIndex, int64_t numRows)
{
  return std::move(RowViewBase<DefaultIndexPolicy, C...>{columnIndex, DefaultIndexPolicy{numRows}});
}

template <typename... C>
using RowViewFiltered = RowViewBase<FilteredIndexPolicy, C...>;

template <typename... C>
auto&& makeRowViewFiltered(std::tuple<std::pair<C*, arrow::Column*>...> const& columnIndex, gandiva::SelectionVector* selection)
{
  return std::move(RowViewBase<FilteredIndexPolicy, C...>{columnIndex, FilteredIndexPolicy{selection}});
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
  using filtered_iterator = RowViewFiltered<C...>;
  using filtered_const_iterator = RowViewFiltered<C...>;
  using table_t = Table<C...>;
  using columns = framework::pack<C...>;
  using persistent_columns_t = framework::selected_pack<is_persistent_t, C...>;

  Table(std::shared_ptr<arrow::Table> table)
    : mTable(table),
      mColumnIndex{
        std::pair<C*, arrow::Column*>{nullptr,
                                      lookupColumn<C>()}...},
      mBegin(mColumnIndex, table->num_rows()),
      mEnd(mColumnIndex, table->num_rows())
  {
    mEnd.moveToEnd();
  }

  iterator begin()
  {
    return iterator(mBegin);
  }

  iterator end()
  {
    return iterator{mEnd};
  }

  filtered_iterator filtered_begin(framework::expressions::Selection selection)
  {
    // Note that the FilteredIndexPolicy will never outlive the selection which
    // is held by the table, so we are safe passing the bare pointer. If it does it
    // means that the iterator on a table is outliving the table itself, which is
    // a bad idea.
    return filtered_iterator(mColumnIndex, selection.get());
  }

  filtered_iterator filtered_end(framework::expressions::Selection selection)
  {
    auto end = filtered_iterator(mColumnIndex, selection.get());
    end.moveToEnd();
    return end;
  }

  const_iterator begin() const
  {
    return const_iterator(mBegin);
  }

  const_iterator end() const
  {
    return const_iterator{mEnd};
  }

  std::shared_ptr<arrow::Table> asArrowTable() const
  {
    return mTable;
  }

  int64_t size() const
  {
    return mTable->num_rows();
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
  std::tuple<std::pair<C*, arrow::Column*>...> mColumnIndex;
  /// Cached begin iterator for this table.
  iterator mBegin;
  /// Cached end iterator for this table.
  iterator mEnd;
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
  struct MetadataTrait<_Name_::iterator> {                             \
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
constexpr auto join(o2::soa::Table<C1...>&& t1, o2::soa::Table<C2...>&& t2)
{
  return o2::soa::Table<C1..., C2...>(ArrowHelpers::joinTables({t1.asArrowTable(), t2.asArrowTable()}));
}

template <typename T1, typename T2>
constexpr auto concat(T1&& t1, T2&& t2)
{
  using table_t = typename PackToTable<framework::intersected_pack_t<typename T1::columns, typename T2::columns>>::table;
  return table_t(ArrowHelpers::concatTables({t1.asArrowTable(), t2.asArrowTable()}));
}

template <typename T1, typename T2>
using JoinBase = decltype(join(std::declval<T1>(), std::declval<T2>()));

template <typename T1, typename T2>
using ConcatBase = decltype(concat(std::declval<T1>(), std::declval<T2>()));

template <typename T1, typename T2>
struct Join : JoinBase<T1, T2> {
  Join(std::shared_ptr<arrow::Table> t1, std::shared_ptr<arrow::Table> t2)
    : JoinBase<T1, T2>{ArrowHelpers::joinTables({t1, t2})} {}
  Join(std::vector<std::shared_ptr<arrow::Table>> tables)
    : JoinBase<T1, T2>{ArrowHelpers::joinTables(std::move(tables))} {}

  using left_t = T1;
  using right_t = T2;
  using table_t = JoinBase<T1, T2>;
};

template <typename T1, typename T2>
struct Concat : ConcatBase<T1, T2> {
  Concat(std::shared_ptr<arrow::Table> t1, std::shared_ptr<arrow::Table> t2)
    : ConcatBase<T1, T2>{ArrowHelpers::concatTables({t1, t2})} {}
  Concat(std::vector<std::shared_ptr<arrow::Table>> tables)
    : ConcatBase<T1, T2>{ArrowHelpers::concatTables(std::move(tables))} {}

  using left_t = T1;
  using right_t = T2;
  using table_t = ConcatBase<T1, T2>;
};

template <typename T>
class Filtered : public T
{
 public:
  using iterator = typename T::filtered_iterator;
  using const_iterator = typename T::filtered_const_iterator;

  Filtered(std::shared_ptr<arrow::Table> table, framework::expressions::Selection selection)
    : T{table},
      mSelection{selection},
      mFilteredBegin{T::filtered_begin(mSelection)},
      mFilteredEnd{T::filtered_end(mSelection)}
  {
  }

  Filtered(std::shared_ptr<arrow::Table> table, framework::expressions::Filter const& expression)
    : Filtered(table, createSelection(table, expression))
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
} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOA_H_
