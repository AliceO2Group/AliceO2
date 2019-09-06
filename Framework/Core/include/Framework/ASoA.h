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

template <typename T>
class ColumnIterator
{
 public:
  ColumnIterator(std::shared_ptr<arrow::Column> const& column)
    : mColumn{column},
      mCurrentChunk{0},
      mLastChunk{mColumn->data()->num_chunks() - 1},
      mLast{nullptr},
      mCurrent{nullptr}
  {
    moveToChunk(mCurrentChunk);
  }

  void moveToChunk(int chunk)
  {
    assert(mColumn.get());
    assert(mColumn->data());
    mCurrentChunk = chunk;
    auto chunks = mColumn->data();
    auto array = std::static_pointer_cast<arrow_array_for_t<T>>(chunks->chunk(chunk));
    assert(array.get());
    mCurrent = array->raw_values();
    mLast = mCurrent + array->length();
  }

  void moveToEnd()
  {
    mCurrentChunk = mLastChunk;
    auto chunks = mColumn->data();
    auto array = std::static_pointer_cast<arrow_array_for_t<T>>(chunks->chunk(mCurrentChunk));
    assert(array.get());
    mLast = array->raw_values() + array->length();
    mCurrent = mLast;
  }

  T const& operator*() const
  {
    return *mCurrent;
  }

  bool operator==(ColumnIterator<T> const& other) const
  {
    return mCurrent == other.mCurrent;
  }

  bool operator!=(ColumnIterator<T> const& other) const
  {
    return mCurrent != other.mCurrent;
  }

  bool operator<(ColumnIterator<T> const& other) const
  {
    if (mCurrentChunk < other.mCurrentChunk) {
      return true;
    }
    return mCurrent < other.mCurrent;
  }

  ColumnIterator<T>& operator++()
  {
    // Notice how end is actually mLast of the last chunk
    if (mCurrent + 1 == mLast && mCurrentChunk < mLastChunk) {
      mCurrentChunk += 1;
      moveToChunk(mCurrentChunk);
    } else {
      mCurrent++;
    }
    return *this;
  }

  ColumnIterator<T> operator++(int)
  {
    ColumnIterator<T> old = *this;
    operator++();
    return old;
  }

  std::shared_ptr<arrow::Column> mColumn;
  T const* mCurrent;
  T const* mLast;
  int mCurrentChunk;
  int mLastChunk;
};

template <typename T, typename INHERIT>
struct Column {
  using storage_t = T;
  using inherited_t = INHERIT;
  Column(ColumnIterator<T> it)
    : mColumnIterator{it}
  {
  }

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
  using storage_t = F;
  using inherited_t = INHERIT;

  DynamicColumn(F callable)
    : mCallable{callable}
  {
  }

  using persistent = std::false_type;
  static constexpr const char* const& label() { return INHERIT::mLabel; }
  F& callable() { return mCallable; }

  F mCallable;
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

  RowView(std::shared_ptr<arrow::Table> const& table)
    : C(table)...
  {
    bindAllDynamicColumns(dynamic_columns_t{});
  }

  RowView<C...>& operator++()
  {
    incrementPersistent(persistent_columns_t{});
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
    return doCompareNotEqual(persistent_columns_t{}, other);
  }

  bool operator==(RowView<C...> const& other) const
  {
    return doCompareEqual(persistent_columns_t{}, other);
  }

  void moveToEnd()
  {
    doMoveToEnd(persistent_columns_t{});
  }

 private:
  /// Helper to only increment the iterators for persistent columns which actually have
  /// it.
  template <typename... PC>
  void incrementPersistent(framework::pack<PC...> pack)
  {
    (PC::mColumnIterator.operator++(), ...);
  }

  /// Helper to move at the end of columns which actually have an iterator.
  template <typename... PC>
  void doMoveToEnd(framework::pack<PC...> pack)
  {
    (PC::mColumnIterator.moveToEnd(), ...);
  }

  /// Helper to move at the end of columns which actually have an iterator.
  template <typename... PC>
  bool doCompareEqual(framework::pack<PC...> pack, RowView<C...> const& other) const
  {
    using first_t = framework::pack_element_t<0, decltype(pack)>;
    return (static_cast<first_t const&>(*this).mColumnIterator.operator==(static_cast<first_t const&>(other).mColumnIterator));
  }

  template <typename... PC>
  bool doCompareNotEqual(framework::pack<PC...> pack, RowView<C...> const& other) const
  {
    using first_t = framework::pack_element_t<0, decltype(pack)>;
    return (static_cast<first_t const&>(*this).mColumnIterator.operator!=(static_cast<first_t const&>(other).mColumnIterator));
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

template <typename... C>
class Table
{
 public:
  using iterator = RowView<C...>;
  using const_iterator = RowView<C...>;
  using columns = framework::pack<C...>;

  Table(std::shared_ptr<arrow::Table> table)
    : mTable(table)
  {
  }

  iterator begin()
  {
    return iterator(mTable);
  }

  iterator end()
  {
    iterator end(mTable);
    end.moveToEnd();
    return end;
  }

  const_iterator begin() const
  {
    return const_iterator(mTable);
  }

  const_iterator end() const
  {
    const_iterator end(mTable);
    end.moveToEnd();
    return end;
  }

  std::shared_ptr<arrow::Table> asArrowTable()
  {
    return mTable;
  }

  std::size_t size() const
  {
    return mTable->num_rows();
  }

 private:
  std::shared_ptr<arrow::Table> mTable;
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

#define DECLARE_SOA_COLUMN(_Name_, _Getter_, _Type_, _Label_)                                                                            \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {                                                                                      \
    static constexpr const char* mLabel = _Label_;                                                                                       \
    using type = _Type_;                                                                                                                 \
    using column_t = _Name_;                                                                                                             \
    _Name_(std::shared_ptr<arrow::Table> table)                                                                                          \
      : o2::soa::Column<_Type_, _Name_>(o2::soa::ColumnIterator<type>(table->column(table->schema()->GetFieldIndex(column_t::label())))) \
    {                                                                                                                                    \
    }                                                                                                                                    \
                                                                                                                                         \
    _Type_ const _Getter_() const                                                                                                        \
    {                                                                                                                                    \
      return *mColumnIterator;                                                                                                           \
    }                                                                                                                                    \
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
#define DECLARE_SOA_DYNAMIC_COLUMN(_Name_, _Getter_, ...)                                                                                  \
  struct _Name_##Callback {                                                                                                                \
    static inline constexpr auto getLambda() { return __VA_ARGS__; }                                                                       \
  };                                                                                                                                       \
                                                                                                                                           \
  struct _Name_##Helper {                                                                                                                  \
    using callable_t = decltype(framework::FunctionMetadata(std::declval<decltype(_Name_##Callback::getLambda())>()));                     \
    using return_type = typename callable_t::return_type;                                                                                  \
  };                                                                                                                                       \
  template <typename... Bindings>                                                                                                          \
  struct _Name_ : o2::soa::DynamicColumn<typename _Name_##Helper::callable_t::type, _Name_<Bindings...>> {                                 \
    using base = o2::soa::DynamicColumn<typename _Name_##Helper::callable_t::type, _Name_<Bindings...>>;                                   \
    using helper = _Name_##Helper;                                                                                                         \
    using callback_holder_t = _Name_##Callback;                                                                                            \
    using callable_t = helper::callable_t;                                                                                                 \
    using callback_t = callable_t::type;                                                                                                   \
                                                                                                                                           \
    _Name_(const std::shared_ptr<arrow::Table> table)                                                                                      \
      : o2::soa::DynamicColumn<typename _Name_##Helper::callable_t::type, _Name_<Bindings...>>(callback_t(callback_holder_t::getLambda())) \
    {                                                                                                                                      \
    }                                                                                                                                      \
    static constexpr const char* mLabel = #_Name_;                                                                                         \
    using type = typename callable_t::return_type;                                                                                         \
                                                                                                                                           \
    template <typename... FreeArgs>                                                                                                        \
    type const _Getter_(FreeArgs... freeArgs) const                                                                                        \
    {                                                                                                                                      \
      return boundGetter(std::make_index_sequence<std::tuple_size_v<decltype(boundIterators)>>{}, freeArgs...);                            \
    }                                                                                                                                      \
                                                                                                                                           \
    template <size_t... Is, typename... FreeArgs>                                                                                          \
    type const boundGetter(std::integer_sequence<size_t, Is...>&& index, FreeArgs... freeArgs) const                                       \
    {                                                                                                                                      \
      return base::mCallable((**std::get<Is>(boundIterators))..., freeArgs...);                                                            \
    }                                                                                                                                      \
                                                                                                                                           \
    using bindings_t = typename framework::pack<Bindings...>;                                                                              \
    std::tuple<soa::ColumnIterator<typename Bindings::type> const*...> boundIterators;                                                     \
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
