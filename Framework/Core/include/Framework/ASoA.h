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
    : mColumn{ column },
      mCurrentChunk{ 0 },
      mLastChunk{ mColumn->data()->num_chunks() - 1 },
      mLast{ nullptr },
      mCurrent{ nullptr }
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
  using type = T;
  static constexpr const char* const& label() { return INHERIT::mLabel; }
  ColumnIterator<T> const& getIterator() const
  {
    return mColumnIterator;
  }
  ColumnIterator<T> mColumnIterator;
};

template <typename... C>
struct RowView : public C... {
 public:
  RowView(std::shared_ptr<arrow::Table> const& table)
    : C{ ColumnIterator<typename C::type>(table->column(table->schema()->GetFieldIndex(C::label()))) }...
  {
  }

  RowView<C...>& operator++()
  {
    (C::mColumnIterator.operator++(), ...);
    return *this;
  }

  RowView<C...> operator++(int)
  {
    RowView<C...> copy = *this;
    operator++();
    return copy;
  }

  RowView<C...> const& operator*()
  {
    return *this;
  }

  bool operator!=(RowView<C...> const& other) const
  {
    return (static_cast<C const&>(*this).mColumnIterator.operator!=(static_cast<C const&>(other).mColumnIterator) || ...);
  }

  bool operator==(RowView<C...> const& other) const
  {
    return (static_cast<C const&>(*this).mColumnIterator.operator==(static_cast<C const&>(other).mColumnIterator) || ...);
  }

  void moveToEnd()
  {
    (C::mColumnIterator.moveToEnd(), ...);
  }
};

template <typename... C>
class Table
{
 public:
  using iterator = RowView<C...>;
  using columns = std::tuple<C...>;

  Table(std::shared_ptr<arrow::Table> table)
    : mTable(table)
  {
  }

  RowView<C...> begin() const
  {
    return RowView<C...>(mTable);
  }

  RowView<C...> end() const
  {
    RowView<C...> end(mTable);
    end.moveToEnd();
    return end;
  }

 private:
  std::shared_ptr<arrow::Table> mTable;
};

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

#define DECLARE_SOA_COLUMN(_Name_, _Getter_, _Type_, _Label_) \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {           \
    static constexpr const char* mLabel = _Label_;            \
    using type = _Type_;                                      \
                                                              \
    _Type_ const _Getter_() const                             \
    {                                                         \
      return *mColumnIterator;                                \
    }                                                         \
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
