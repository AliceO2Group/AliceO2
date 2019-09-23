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
#include <arrow/table.h>
#include <arrow/array.h>
#include <cassert>

namespace o2
{
namespace soa
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
template <typename T, typename ChunkingPolicy = Flat>
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
    mCurrentChunk++;
    auto chunks = mColumn->data();
    auto array = std::static_pointer_cast<arrow_array_for_t<T>>(chunks->chunk(mCurrentChunk));
    mFirstIndex += array->length();
    mCurrent = array->raw_values() - mFirstIndex;
    mLast = mCurrent + array->length() + mFirstIndex;
  }

  void prevChunk()
  {
    mCurrentChunk--;
    auto chunks = mColumn->data();
    auto array = std::static_pointer_cast<arrow_array_for_t<T>>(chunks->chunk(mCurrentChunk));
    mFirstIndex -= array->length();
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
      if (O2_BUILTIN_UNLIKELY(((mCurrent + *mCurrentPos) > mLast))) {
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
      while (O2_BUILTIN_UNLIKELY((mCurrent + *mCurrentPos) > mLast)) {
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
  size_t const* mCurrentPos;
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

template <typename T>
using is_not_persistent_t = typename std::decay_t<T>::persistent::type;

template <typename T>
using is_persistent_t = std::is_same<typename std::decay_t<T>::persistent::type, std::false_type>;

template <typename... C>
struct RowView : public C... {
 public:
  using persistent_columns_t = framework::filtered_pack<is_not_persistent_t, C...>;
  using dynamic_columns_t = framework::filtered_pack<is_persistent_t, C...>;

  RowView(std::tuple<std::pair<C*, arrow::Column*>...> const& columnIndex, size_t numRows)
    : C(std::get<std::pair<C*, arrow::Column*>>(columnIndex).second)...
  {
    bindIterators(persistent_columns_t{});
    bindAllDynamicColumns(dynamic_columns_t{});
    mMaxRow = numRows;
  }

  RowView(RowView<C...> const& other)
    : C(static_cast<C const&>(other))...
  {
    mRowIndex = other.mRowIndex;
    mMaxRow = other.mMaxRow;
    bindIterators(persistent_columns_t{});
    bindAllDynamicColumns(dynamic_columns_t{});
  }

  RowView() = default;
  RowView<C...>& operator=(RowView<C...> const& other)
  {
    mRowIndex = other.mRowIndex;
    mMaxRow = other.mMaxRow;
    (void(static_cast<C&>(*this) = static_cast<C const&>(other)), ...);
    bindIterators(persistent_columns_t{});
    bindAllDynamicColumns(dynamic_columns_t{});
    return *this;
  }

  RowView(RowView<C...>&& other)
  {
    mRowIndex = other.mRowIndex;
    mMaxRow = other.mMaxRow;
    (void(static_cast<C&>(*this) = static_cast<C const&>(other)), ...);
  }

  RowView& operator=(RowView<C...>&& other)
  {
    mRowIndex = other.mRowIndex;
    mMaxRow = other.mMaxRow;
    (void(static_cast<C&>(*this) = static_cast<C const&>(other)), ...);
    return *this;
  }

  RowView<C...>& operator++()
  {
    ++mRowIndex;
    return *this;
  }

  RowView<C...> operator++(int)
  {
    RowView<C...> copy = *this;
    operator++();
    return copy;
  }

  RowView<C...> const& operator*() const
  {
    return *this;
  }

  /// Notice this relies on the fact that the iterators in the column are all
  /// in sync and therefore we check only for the first one.
  bool operator!=(RowView<C...> const& other) const
  {
    return O2_BUILTIN_LIKELY(mRowIndex != other.mRowIndex);
  }

  bool operator==(RowView<C...> const& other) const
  {
    return O2_BUILTIN_UNLIKELY(mRowIndex == other.mRowIndex);
  }

  void moveToEnd()
  {
    mRowIndex = mMaxRow;
  }

 private:
  size_t mRowIndex = 0;
  size_t mMaxRow = 0;
  /// Helper to move to the correct chunk, if needed.
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
    (void(PC::mColumnIterator.mCurrentPos = &mRowIndex), ...);
  }

  template <typename... DC>
  auto bindAllDynamicColumns(framework::pack<DC...> pack)
  {
    using namespace o2::soa;
    (bindDynamicColumn<DC>(typename DC::bindings_t{}), ...);
  }

  template <typename DC, typename... B>
  auto bindDynamicColumn(framework::pack<B...> bindings)
  {
    DC::boundIterators = std::make_tuple(&(B::mColumnIterator)...);
  }
};

/// A Table class which observes an arrow::Table and provides
/// It is templated on a set of Column / DynamicColumn types.
template <typename... C>
class Table
{
 public:
  using iterator = RowView<C...>;
  using const_iterator = RowView<C...>;
  using columns = framework::pack<C...>;

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

  std::size_t size() const
  {
    return mTable->num_rows();
  }

 private:
  template <typename T>
  arrow::Column* lookupColumn()
  {
    if constexpr (T::persistent::value) {
      return mTable->column(mTable->schema()->GetFieldIndex(T::label())).get();
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
  using persistent_columns_pack = framework::filtered_pack<is_not_persistent_t, C...>;
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
} // namespace soa

} // namespace o2

#define DECLARE_SOA_STORE()          \
  template <typename T>              \
  struct MetadataTrait {             \
    using metadata = std::void_t<T>; \
  }

#define DECLARE_SOA_COLUMN(_Name_, _Getter_, _Type_, _Label_)                  \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {                            \
    static constexpr const char* mLabel = _Label_;                             \
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
  }

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
  }

#endif // O2_FRAMEWORK_ASOA_H_
