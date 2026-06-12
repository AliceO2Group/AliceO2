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

#include "Framework/ConcreteDataMatcher.h"
#include "Framework/Pack.h"                   // IWYU pragma: export
#include "Framework/FunctionalHelpers.h"      // IWYU pragma: export
#include "Headers/DataHeader.h"               // IWYU pragma: export
#include "Headers/DataHeaderHelpers.h"        // IWYU pragma: export
#include "Framework/CompilerBuiltins.h"       // IWYU pragma: export
#include "Framework/Traits.h"                 // IWYU pragma: export
#include "Framework/Expressions.h"            // IWYU pragma: export
#include "Framework/ArrowTypes.h"             // IWYU pragma: export
#include "Framework/ArrowTableSlicingCache.h" // IWYU pragma: export
#include "Framework/SliceCache.h"             // IWYU pragma: export
#include "Framework/VariantHelpers.h"         // IWYU pragma: export
#include <arrow/array/array_binary.h>
#include <arrow/table.h>              // IWYU pragma: export
#include <arrow/array.h>              // IWYU pragma: export
#include <arrow/util/config.h>        // IWYU pragma: export
#include <gandiva/selection_vector.h> // IWYU pragma: export
#include <array>                      // IWYU pragma: export
#include <cassert>
#include <fmt/format.h>
#include <concepts>
#include <cstring>
#include <gsl/span> // IWYU pragma: export

namespace o2::framework
{
using ListVector = std::vector<std::vector<int64_t>>;

std::string cutString(std::string&& str);
std::string strToUpper(std::string&& str);
} // namespace o2::framework

struct TClass;

namespace o2::soa
{
void accessingInvalidIndexFor(const char* getter);
void dereferenceWithWrongType(const char* getter, const char* target);
void missingFilterDeclaration(int hash, int ai);
void notBoundTable(const char* tableName);
void* extractCCDBPayload(char* payload, size_t size, TClass const* cl, const char* what);

template <typename... C>
auto createFieldsFromColumns(framework::pack<C...>)
{
  return std::vector<std::shared_ptr<arrow::Field>>{C::asArrowField()...};
}
} // namespace o2::soa

namespace o2::soa
{
/// Generic identifier for a table type
struct TableRef {
  consteval TableRef()
    : label_hash{0},
      desc_hash{0},
      origin_hash{0},
      version{0}
  {
  }
  consteval TableRef(uint32_t _label, uint32_t _desc, uint32_t _origin, uint32_t _version)
    : label_hash{_label},
      desc_hash{_desc},
      origin_hash{_origin},
      version{_version}
  {
  }
  uint32_t label_hash;
  uint32_t desc_hash;
  uint32_t origin_hash;
  uint32_t version;

  constexpr bool operator==(TableRef const& other) const noexcept
  {
    return (this->label_hash == other.label_hash) &&
           (this->desc_hash == other.desc_hash) &&
           (this->origin_hash == other.origin_hash) &&
           (this->version == other.version);
  }

  constexpr bool descriptionCompatible(TableRef const& other) const noexcept
  {
    return this->desc_hash == other.desc_hash;
  }

  constexpr bool descriptionCompatible(uint32_t _desc_hash) const noexcept
  {
    return this->desc_hash == _desc_hash;
  }

  constexpr TableRef(TableRef const&) = default;
  constexpr TableRef& operator=(TableRef const&) = default;
  constexpr TableRef(TableRef&&) = default;
  constexpr TableRef& operator=(TableRef&&) = default;
};

/// Helpers to manipulate TableRef arrays
template <size_t N1, size_t N2, std::array<TableRef, N1> ar1, std::array<TableRef, N2> ar2>
consteval auto merge()
{
  constexpr const int duplicates = std::ranges::count_if(ar2.begin(), ar2.end(), [&](TableRef const& a) { return std::any_of(ar1.begin(), ar1.end(), [&](TableRef const& e) { return e == a; }); });
  std::array<TableRef, N1 + N2 - duplicates> out;

  auto pos = std::copy(ar1.begin(), ar1.end(), out.begin());
  std::copy_if(ar2.begin(), ar2.end(), pos, [&](TableRef const& a) { return std::none_of(ar1.begin(), ar1.end(), [&](TableRef const& e) { return e == a; }); });
  return out;
}

template <size_t N1, size_t N2, std::array<TableRef, N1> ar1, std::array<TableRef, N2> ar2, typename L>
consteval auto merge_if(L l)
{
  constexpr const int to_remove = std::ranges::count_if(ar1.begin(), ar1.end(), [&](TableRef const& a) { return !l(a); });
  constexpr const int duplicates = std::ranges::count_if(ar2.begin(), ar2.end(), [&](TableRef const& a) { return std::any_of(ar1.begin(), ar1.end(), [&](TableRef const& e) { return e == a; }) || !l(a); });
  std::array<TableRef, N1 + N2 - duplicates - to_remove> out;

  auto pos = std::copy_if(ar1.begin(), ar1.end(), out.begin(), [&](TableRef const& a) { return l(a); });
  std::copy_if(ar2.begin(), ar2.end(), pos, [&](TableRef const& a) { return std::none_of(ar1.begin(), ar1.end(), [&](TableRef const& e) { return e == a; }) && l(a); });
  return out;
}

template <size_t N, std::array<TableRef, N> ar, typename L>
consteval auto remove_if(L l)
{
  constexpr const int to_remove = std::ranges::count_if(ar.begin(), ar.end(), [&l](TableRef const& e) { return l(e); });
  std::array<TableRef, N - to_remove> out;
  std::copy_if(ar.begin(), ar.end(), out.begin(), [&l](TableRef const& e) { return !l(e); });
  return out;
}

template <size_t N1, size_t N2, std::array<TableRef, N1> ar1, std::array<TableRef, N2> ar2>
consteval auto intersect()
{
  constexpr const int duplicates = std::ranges::count_if(ar2.begin(), ar2.end(), [&](TableRef const& a) { return std::any_of(ar1.begin(), ar1.end(), [&](TableRef const& e) { return e == a; }); });
  std::array<TableRef, duplicates> out;
  std::copy_if(ar1.begin(), ar1.end(), out.begin(), [](TableRef const& a) { return std::find(ar2.begin(), ar2.end(), a) != ar2.end(); });
  return out;
}

template <typename T, typename... Ts>
consteval auto mergeOriginals()
  requires(sizeof...(Ts) == 1)
{
  using T1 = framework::pack_head_t<framework::pack<Ts...>>;
  return merge<T::originals.size(), T1::originals.size(), T::originals, T1::originals>();
}

template <typename T, typename... Ts>
consteval auto mergeOriginals()
  requires(sizeof...(Ts) > 1)
{
  constexpr auto tail = mergeOriginals<Ts...>();
  return merge<T::originals.size(), tail.size(), T::originals, tail>();
}

template <typename T, typename... Ts>
  requires(sizeof...(Ts) == 1)
consteval auto intersectOriginals()
{
  using T1 = framework::pack_head_t<framework::pack<Ts...>>;
  return intersect<T::originals.size(), T1::originals.size(), T::originals, T1::originals>();
}

template <typename T, typename... Ts>
  requires(sizeof...(Ts) > 1)
consteval auto intersectOriginals()
{
  constexpr auto tail = intersectOriginals<Ts...>();
  return intersect<T::originals.size(), tail.size(), T::originals, tail>();
}
} // namespace o2::soa

namespace o2::soa
{
struct Binding;

template <typename T>
concept not_void = requires { !std::same_as<T, void>; };

/// column identification concepts
template <typename C>
concept is_persistent_column = requires(C c) { c.mColumnIterator; };

template <typename C>
constexpr bool is_persistent_v = is_persistent_column<C>;

template <typename C>
using is_persistent_column_t = std::conditional_t<is_persistent_column<C>, std::true_type, std::false_type>;

template <typename C>
concept is_self_index_column = not_void<typename C::self_index_t> && std::same_as<typename C::self_index_t, std::true_type>;

template <typename C>
concept is_index_column = !is_self_index_column<C> && requires(C c, o2::soa::Binding b) {
  { c.setCurrentRaw(b) } -> std::same_as<bool>;
  requires std::same_as<decltype(c.mBinding), o2::soa::Binding>;
};

template <typename C>
using is_external_index_t = typename std::conditional_t<is_index_column<C>, std::true_type, std::false_type>;

template <typename C>
using is_self_index_t = typename std::conditional_t<is_self_index_column<C>, std::true_type, std::false_type>;
} // namespace o2::soa

namespace o2::aod
{
namespace
{
template <typename Key, size_t N, std::array<bool, N> map>
static consteval int getIndexPosToKey_impl()
{
  constexpr const auto pos = std::find(map.begin(), map.end(), true);
  if constexpr (pos != map.end()) {
    return std::distance(map.begin(), pos);
  } else {
    return -1;
  }
}
} // namespace

/// Base type for table metadata
template <typename D, typename... Cs>
struct TableMetadata {
  using columns = framework::pack<Cs...>;
  using persistent_columns_t = framework::selected_pack<soa::is_persistent_column_t, Cs...>;
  using external_index_columns_t = framework::selected_pack<soa::is_external_index_t, Cs...>;
  using internal_index_columns_t = framework::selected_pack<soa::is_self_index_t, Cs...>;

  template <typename Key, typename... PCs>
  static consteval std::array<bool, sizeof...(PCs)> getMap(framework::pack<PCs...>)
  {
    return std::array<bool, sizeof...(PCs)>{[]() {
      if constexpr (requires { PCs::index_targets.size(); }) {
        return Key::template isIndexTargetOf<PCs::index_targets.size(), PCs::index_targets>();
      } else {
        return false;
      }
    }()...};
  }

  template <typename Key>
  static consteval int getIndexPosToKey()
  {
    return getIndexPosToKey_impl<Key, framework::pack_size(persistent_columns_t{}), getMap<Key>(persistent_columns_t{})>();
  }

  static std::shared_ptr<arrow::Schema> getSchema()
  {
    return std::make_shared<arrow::Schema>([]<typename... C>(framework::pack<C...>&& p) { return o2::soa::createFieldsFromColumns(p); }(persistent_columns_t{}));
  }
};

template <typename D>
struct MetadataTrait {
  using metadata = void;
};

/// Special struc to map the string hash back to the string and wrap a string literal into the
/// type signature
template <uint32_t H>
struct Hash {
  static constexpr uint32_t hash = H;
  static constexpr char const* const str{""};
};

/// Filter TableRef array for compatibility with Key table
template <size_t N, std::array<soa::TableRef, N> ar, typename Key>
consteval auto filterForKey()
{
  constexpr std::array<bool, N> test = []<size_t... Is>(std::index_sequence<Is...>) {
    return std::array<bool, N>{(Key::template hasOriginal<ar[Is]>() || (o2::aod::MetadataTrait<o2::aod::Hash<ar[Is].desc_hash>>::metadata::template getIndexPosToKey<Key>() >= 0))...};
  }(std::make_index_sequence<N>());
  constexpr int correct = std::ranges::count(test.begin(), test.end(), true);
  std::array<soa::TableRef, correct> out;
  std::ranges::copy_if(ar.begin(), ar.end(), out.begin(), [&test](soa::TableRef const& r) { return test[std::distance(ar.begin(), std::find(ar.begin(), ar.end(), r))]; });
  return out;
}

/// Pre-declare Hash specialization for a generic string
#define O2HASH(_Str_)                              \
  template <>                                      \
  struct Hash<_Str_ ""_h> {                        \
    static constexpr uint32_t hash = _Str_ ""_h;   \
    static constexpr char const* const str{_Str_}; \
  };

/// Pre-declare Hash specialization for an origin string
#define O2ORIGIN(_Str_)                                \
  template <>                                          \
  struct Hash<_Str_ ""_h> {                            \
    static constexpr header::DataOrigin origin{_Str_}; \
    static constexpr uint32_t hash = _Str_ ""_h;       \
    static constexpr char const* const str{_Str_};     \
  };

/// Compile-time function to extract version from table signature string "DESC/#"
static inline constexpr uint32_t version(const char* const str)
{
  if (str[0] == '\0') {
    return 0;
  }
  size_t len = 0;
  uint32_t res = 0;
  while (str[len] != '/' && str[len] != '\0') {
    ++len;
  }
  if (str[len - 1] == '\0') {
    return -1;
  }
  for (auto i = len + 1; str[i] != '\0'; ++i) {
    res = res * 10 + (int)(str[i] - '0');
  }
  return res;
}

/// Compile-time functions to extract description from table signature string "DESC/#"
static inline constexpr std::string_view description_str(const char* const str)
{
  size_t len = 0;
  while (len < 15 && str[len] != '/') {
    ++len;
  }
  return std::string_view{str, len};
}

static inline constexpr header::DataDescription description(const char* const str)
{
  size_t len = 0;
  while (len < 15 && str[len] != '/') {
    ++len;
  }
  char out[16];
  for (auto i = 0; i < 16; ++i) {
    out[i] = 0;
  }
  std::memcpy(out, str, len);
  return {out};
}

// Helpers to get strings from TableRef
template <soa::TableRef R>
consteval const char* label()
{
  return o2::aod::Hash<R.label_hash>::str;
}

template <soa::TableRef R>
consteval const char* origin_str()
{
  return o2::aod::Hash<R.origin_hash>::str;
}

template <soa::TableRef R>
consteval header::DataOrigin origin()
{
  return o2::aod::Hash<R.origin_hash>::origin;
}

template <soa::TableRef R>
consteval const char* signature()
{
  return o2::aod::Hash<R.desc_hash>::str;
}

template <soa::TableRef R>
constexpr framework::ConcreteDataMatcher matcher()
{
  return {origin<R>(), description(signature<R>()), R.version};
}

/// hash identification concepts
template <typename T>
concept is_aod_hash = requires(T t) { t.hash; t.str; };

template <typename T>
concept is_origin_hash = is_aod_hash<T> && requires(T t) { t.origin; };

/// convert TableRef to a DPL source specification
template <soa::TableRef R>
static constexpr auto sourceSpec()
{
  return fmt::format("{}/{}/{}/{}", label<R>(), origin_str<R>(), description_str(signature<R>()), R.version);
}

/// Replace origins in the TableRef array
template <size_t N, std::array<soa::TableRef, N> ar, o2::aod::is_origin_hash O>
consteval auto replaceOrigin()
{
  std::array<soa::TableRef, N> res;
  for (auto i = 0U; i < N; ++i) {
    res[i].label_hash = ar[i].label_hash;
    res[i].desc_hash = ar[i].desc_hash;
    res[i].origin_hash = O::hash;
    res[i].version = ar[i].version;
  }
  return res;
}
} // namespace o2::aod

namespace o2::soa
{
template <aod::is_aod_hash L, aod::is_aod_hash D, aod::is_origin_hash O, typename... Ts>
class Table;

/// Type-checking index column binding
struct Binding {
  void const* ptr = nullptr;
  uint32_t hash = 0;
  // std::span<TableRef const> refs;

  template <typename T>
  void bind(T const* table)
  {
    ptr = table;
    hash = o2::framework::TypeIdHelpers::uniqueId<T>();
    // refs = std::span{T::originals};
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

using SelectionVector = std::vector<int64_t>;

template <typename T>
concept has_parent_t = not_void<typename T::parent_t>;

template <typename T>
concept is_metadata = framework::base_of_template<aod::TableMetadata, T>;

template <typename T>
concept is_metadata_trait = framework::specialization_of_template<aod::MetadataTrait, T>;

template <typename T>
concept has_metadata = is_metadata_trait<T> && not_void<typename T::metadata>;

template <typename T>
concept has_extension = is_metadata<T> && not_void<typename T::extension_table_t>;

template <typename T>
concept has_configurable_extension = has_extension<T> && requires(T t) { typename T::configurable_t; requires std::same_as<std::true_type, typename T::configurable_t>; };

template <typename T>
concept is_spawnable_column = std::same_as<typename T::spawnable_t, std::true_type>;

template <typename B, typename E>
struct EquivalentIndex {
  constexpr static bool value = false;
};

template <aod::is_aod_hash A, aod::is_aod_hash B>
struct EquivalentIndexNG {
  constexpr static bool value = false;
};

template <typename B, typename E>
constexpr bool is_index_equivalent_v = EquivalentIndex<B, E>::value || EquivalentIndex<E, B>::value;

template <aod::is_aod_hash A, aod::is_aod_hash B>
constexpr bool is_ng_index_equivalent_v = EquivalentIndexNG<A, B>::value || EquivalentIndexNG<B, A>::value;

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
  static constexpr char SCALE_FACTOR = std::same_as<std::decay_t<T>, bool> ? 3 : 0;

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

  auto operator*() const
    requires std::same_as<bool, std::decay_t<T>>
  {
    checkSkipChunk();
    return (*(mCurrent - (mOffset >> SCALE_FACTOR) + ((*mCurrentPos + mOffset) >> SCALE_FACTOR)) & (1 << ((*mCurrentPos + mOffset) & 0x7))) != 0;
  }

  auto operator*() const
    requires((!std::same_as<bool, std::decay_t<T>>) && std::same_as<arrow_array_for_t<T>, arrow::ListArray>)
  {
    checkSkipChunk();
    auto list = std::static_pointer_cast<arrow::ListArray>(mColumn->chunk(mCurrentChunk));
    auto offset = list->value_offset(*mCurrentPos - mFirstIndex);
    auto length = list->value_length(*mCurrentPos - mFirstIndex);
    return gsl::span<unwrap_t<T> const>{mCurrent + mFirstIndex + offset, mCurrent + mFirstIndex + (offset + length)};
  }

  decltype(auto) operator*() const
    requires((!std::same_as<bool, std::decay_t<T>>) && std::same_as<arrow_array_for_t<T>, arrow::BinaryViewArray>)
  {
    checkSkipChunk();
    auto array = std::static_pointer_cast<arrow::BinaryViewArray>(mColumn->chunk(mCurrentChunk));
    return array->GetView(*mCurrentPos - mFirstIndex);
  }

  decltype(auto) operator*() const
    requires((!std::same_as<bool, std::decay_t<T>>) && !std::same_as<arrow_array_for_t<T>, arrow::ListArray> && !std::same_as<arrow_array_for_t<T>, arrow::BinaryViewArray>)
  {
    checkSkipChunk();
    return *(mCurrent + (*mCurrentPos >> SCALE_FACTOR));
  }

  // Move to the chunk which containts element pos
  ColumnIterator<T>& moveToPos()
  {
    checkSkipChunk();
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
  void checkSkipChunk() const
    requires((ChunkingPolicy::chunked == true) && std::same_as<arrow_array_for_t<T>, arrow::ListArray>)
  {
    auto list = std::static_pointer_cast<arrow::ListArray>(mColumn->chunk(mCurrentChunk));
    if (O2_BUILTIN_UNLIKELY(*mCurrentPos - mFirstIndex >= list->length())) {
      nextChunk();
    }
  }

  void checkSkipChunk() const
    requires((ChunkingPolicy::chunked == true) && !std::same_as<arrow_array_for_t<T>, arrow::ListArray>)
  {
    if (O2_BUILTIN_UNLIKELY(((mCurrent + (*mCurrentPos >> SCALE_FACTOR)) >= mLast))) {
      nextChunk();
    }
  }

  void checkSkipChunk() const
    requires(ChunkingPolicy::chunked == false)
  {
  }
  /// get pointer to mCurrentChunk chunk
  auto getCurrentArray() const
    requires(std::same_as<arrow_array_for_t<T>, arrow::FixedSizeListArray>)
  {
    std::shared_ptr<arrow::Array> chunkToUse = mColumn->chunk(mCurrentChunk);
    mOffset = chunkToUse->offset();
    chunkToUse = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(chunkToUse)->values();
    return std::static_pointer_cast<arrow_array_for_t<value_for_t<T>>>(chunkToUse);
  }

  auto getCurrentArray() const
    requires(std::same_as<arrow_array_for_t<T>, arrow::ListArray>)
  {
    std::shared_ptr<arrow::Array> chunkToUse = mColumn->chunk(mCurrentChunk);
    mOffset = chunkToUse->offset();
    chunkToUse = std::dynamic_pointer_cast<arrow::ListArray>(chunkToUse)->values();
    mOffset = chunkToUse->offset();
    return std::static_pointer_cast<arrow_array_for_t<value_for_t<T>>>(chunkToUse);
  }

  auto getCurrentArray() const
    requires(!std::same_as<arrow_array_for_t<T>, arrow::FixedSizeListArray> && !std::same_as<arrow_array_for_t<T>, arrow::ListArray>)
  {
    std::shared_ptr<arrow::Array> chunkToUse = mColumn->chunk(mCurrentChunk);
    mOffset = chunkToUse->offset();
    return std::static_pointer_cast<arrow_array_for_t<T>>(chunkToUse);
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

  using type = T;
  static constexpr const char* const& columnLabel() { return INHERIT::mLabel; }
  ColumnIterator<T> const& getIterator() const
  {
    return mColumnIterator;
  }

  static auto asArrowField()
  {
    return std::make_shared<arrow::Field>(inherited_t::mLabel, soa::asArrowDataType<type>());
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

  static constexpr const char* const& columnLabel() { return INHERIT::mLabel; }
};

template <typename INHERIT>
struct IndexColumn {
  using inherited_t = INHERIT;
  static constexpr const uint32_t hash = 0;

  static constexpr const char* const& columnLabel() { return INHERIT::mLabel; }
};

template <typename INHERIT>
struct MarkerColumn {
  using inherited_t = INHERIT;
  static constexpr const uint32_t hash = 0;

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

  std::tuple<int64_t const*, int64_t const*> rowIndices;
  /// The offsets within larger tables. Currently only
  /// one level of nesting is supported.
  std::tuple<uint64_t const*> rowOffsets;
};

template <typename C>
concept is_indexing_column = requires(C& c) {
  c.rowIndices;
  c.rowOffsets;
};

template <typename C>
concept is_dynamic_column = requires(C& c) {
  c.boundIterators;
};

template <typename C>
concept is_marker_column = requires { &C::mark; };

template <typename T>
using is_dynamic_t = std::conditional_t<is_dynamic_column<T>, std::true_type, std::false_type>;

template <typename T>
concept is_column = is_persistent_column<T> || is_dynamic_column<T> || is_indexing_column<T> || is_marker_column<T>;

template <typename T>
using is_indexing_t = std::conditional_t<is_indexing_column<T>, std::true_type, std::false_type>;

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
  FilteredIndexPolicy(std::span<int64_t const> selection, int64_t rows, uint64_t offset = 0)
    : IndexPolicyBase{-1, offset},
      mSelectedRows(selection),
      mMaxSelection(selection.size()),
      nRows{rows}
  {
    this->setCursor(0);
  }

  void resetSelection(std::span<int64_t const> selection)
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
  std::span<int64_t const> mSelectedRows;
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

// template <OriginEnc ORIGIN, typename... C>
// class Table;

template <aod::is_aod_hash L, aod::is_aod_hash D, aod::is_origin_hash O, typename... T>
class Table;

template <typename T>
concept is_table = framework::specialization_of_template<soa::Table, T> || framework::base_of_template<soa::Table, T>;

/// Similar to a pair but not a pair, to avoid
/// exposing the second type everywhere.
template <typename C>
struct ColumnDataHolder {
  C* first;
  arrow::ChunkedArray* second;
};

template <typename T, typename B>
concept can_bind = requires(T&& t) {
  { t.B::mColumnIterator };
};

template <typename... C>
concept has_index = (is_indexing_column<C> || ...);

template <typename D, typename O, typename IP, typename... C>
struct TableIterator : IP, C... {
 public:
  using self_t = TableIterator<D, O, IP, C...>;
  using policy_t = IP;
  using all_columns = framework::pack<C...>;
  using persistent_columns_t = framework::selected_pack<soa::is_persistent_column_t, C...>;
  using external_index_columns_t = framework::selected_pack<soa::is_external_index_t, C...>;
  using internal_index_columns_t = framework::selected_pack<soa::is_self_index_t, C...>;
  using bindings_pack_t = decltype([]<typename... Cs>(framework::pack<Cs...>) -> framework::pack<typename Cs::binding_t...> {}(external_index_columns_t{})); // decltype(extractBindings(external_index_columns_t{}));

  TableIterator(arrow::ChunkedArray* columnData[sizeof...(C)], IP&& policy)
    : IP{policy},
      C(columnData[framework::has_type_at_v<C>(all_columns{})])...
  {
    if (this->size() != 0) {
      bind();
    }
  }

  TableIterator(arrow::ChunkedArray* columnData[sizeof...(C)], IP&& policy)
    requires(has_index<C...>)
    : IP{policy},
      C(columnData[framework::has_type_at_v<C>(all_columns{})])...
  {
    if (this->size() != 0) {
      bind();
    }
    // In case we have an index column might need to constrain the actual
    // number of rows in the view to the range provided by the index.
    // FIXME: we should really understand what happens to an index when we
    // have a RowViewFiltered.
    this->limitRange(this->rangeStart(), this->rangeEnd());
  }

  TableIterator() = default;
  TableIterator(self_t const& other)
    : IP{static_cast<IP const&>(other)},
      C(static_cast<C const&>(other))...
  {
    if (this->size() != 0) {
      bind();
    }
  }

  TableIterator& operator=(TableIterator other)
  {
    IP::operator=(static_cast<IP const&>(other));
    (void(static_cast<C&>(*this) = static_cast<C>(other)), ...);
    if (this->size() != 0) {
      bind();
    }
    return *this;
  }

  TableIterator(TableIterator<D, O, FilteredIndexPolicy, C...> const& other)
    requires std::same_as<IP, DefaultIndexPolicy>
    : IP{static_cast<IP const&>(other)},
      C(static_cast<C const&>(other))...
  {
    if (this->size() != 0) {
      bind();
    }
  }

  TableIterator& operator++()
  {
    this->moveByIndex(1);
    return *this;
  }

  TableIterator operator++(int)
  {
    self_t copy = *this;
    this->operator++();
    return copy;
  }

  TableIterator& operator--()
  {
    this->moveByIndex(-1);
    return *this;
  }

  TableIterator operator--(int)
  {
    self_t copy = *this;
    this->operator--();
    return copy;
  }

  /// Allow incrementing by more than one the iterator
  TableIterator operator+(int64_t inc) const
  {
    TableIterator copy = *this;
    copy.moveByIndex(inc);
    return copy;
  }

  TableIterator operator-(int64_t dec) const
  {
    return operator+(-dec);
  }

  TableIterator const& operator*() const
  {
    return *this;
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
    ([this]<soa::is_index_column... CCs>(TA* cur, framework::pack<CCs...>) {
      (CCs::setCurrent(cur), ...);
    }(current, external_index_columns_t{}),
     ...);
  }

  template <typename TA>
  void bindExternalIndex(TA* current)
  {
    [this]<soa::is_index_column... CCs>(TA* cur, framework::pack<CCs...>) {
      (CCs::setCurrent(cur), ...);
    }(current, external_index_columns_t{});
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
    auto f = framework::overloaded{
      [this]<soa::is_persistent_column T>(T*) -> void { T::mColumnIterator.mCurrentPos = &this->mRowIndex; },
      [this]<soa::is_dynamic_column T>(T*) -> void { bindDynamicColumn<T>(typename T::bindings_t{}); },
      [this]<typename T>(T*) -> void {},
    };
    (f(static_cast<C*>(nullptr)), ...);
    if constexpr (has_index<C...>) {
      this->setIndices(this->getIndices());
      this->setOffsets(this->getOffsets());
    }
  }

  template <typename DC, typename... B>
  auto bindDynamicColumn(framework::pack<B...>)
  {
    DC::boundIterators = std::make_tuple(getDynamicBinding<B>()...);
  }

  // Sometimes dynamic columns are defined for tables in
  // the hope that it will be joined / extended with another one which provides
  // the full set of bindings. This is to avoid a compilation
  // error if constructor for the table or any other thing involving a missing
  // binding is preinstanciated.
  template <typename B>
    requires(can_bind<self_t, B>)
  decltype(auto) getDynamicBinding()
  {
    static_assert(std::same_as<decltype(&(static_cast<B*>(this)->mColumnIterator)), std::decay_t<decltype(B::mColumnIterator)>*>, "foo");
    return &(static_cast<B*>(this)->mColumnIterator);
    // return static_cast<std::decay_t<decltype(B::mColumnIterator)>*>(nullptr);
  }

  template <typename B>
  decltype(auto) getDynamicBinding()
  {
    return static_cast<std::decay_t<decltype(B::mColumnIterator)>*>(nullptr);
  }
};

struct ArrowHelpers {
  static std::shared_ptr<arrow::Table> joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables);
  static std::shared_ptr<arrow::Table> joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables, std::span<const char* const> labels);
  static std::shared_ptr<arrow::Table> joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables, std::span<const std::string> labels);
  static std::shared_ptr<arrow::Table> concatTables(std::vector<std::shared_ptr<arrow::Table>>&& tables);
};

//! Helper to check if a type T is an iterator
template <typename T>
concept is_iterator = framework::base_of_template<TableIterator, T> || framework::specialization_of_template<TableIterator, T>;

template <typename T>
concept is_table_or_iterator = is_table<T> || is_iterator<T>;

template <typename T>
concept with_originals = requires {
  T::originals.size();
};

template <typename T>
concept with_sources = requires {
  T::sources.size();
};

template <typename T>
concept with_sources_generator = requires(T t) {
  t.template generateSources<o2::aod::Hash<"AOD"_h>>();
};

template <typename T>
concept with_ccdb_urls = requires {
  T::ccdb_urls.size();
};

template <typename T>
concept with_base_table = requires {
  typename aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>::metadata::base_table_t;
};

template <typename T>
concept with_expression_pack = requires {
  typename T::expression_pack_t{};
};

template <typename T>
concept with_index_pack = requires {
  typename T::index_pack_t{};
};

template <size_t N1, std::array<TableRef, N1> os1, size_t N2, std::array<TableRef, N2> os2>
consteval bool is_compatible()
{
  return []<size_t... Is>(std::index_sequence<Is...>) {
    return ([]<size_t... Ks>(std::index_sequence<Ks...>) {
      constexpr auto h = os1[Is].desc_hash;
      using H = o2::aod::Hash<h>;
      return (((h == os2[Ks].desc_hash) || is_ng_index_equivalent_v<H, o2::aod::Hash<os2[Ks].desc_hash>>) || ...);
    }(std::make_index_sequence<N2>()) ||
            ...);
  }(std::make_index_sequence<N1>());
}

template <with_originals T, with_originals B>
consteval bool is_binding_compatible_v()
{
  return is_compatible<T::originals.size(), T::originals, B::originals.size(), B::originals>();
}

template <typename T, typename B>
using is_binding_compatible = std::conditional_t<is_binding_compatible_v<T, typename B::binding_t>(), std::true_type, std::false_type>;

template <typename L, typename D, typename O, typename Key, typename H, typename... Ts>
struct IndexTable;

template <typename T>
concept is_index_table = framework::specialization_of_template<o2::soa::IndexTable, T>;

template <soa::is_table T>
static constexpr std::string getLabelForTable()
{
  return std::string{aod::label<std::decay_t<T>::originals[0]>()};
}

template <soa::is_table T>
  requires(!(soa::is_index_table<T> || soa::with_base_table<T>))
static constexpr std::string getLabelFromType()
{
  return getLabelForTable<T>();
}

template <soa::is_iterator T>
static constexpr std::string getLabelFromType()
{
  return getLabelForTable<typename std::decay_t<T>::parent_t>();
}

template <soa::is_index_table T>
static constexpr std::string getLabelFromType()
{
  return getLabelForTable<typename std::decay_t<T>::first_t>();
}
template <soa::with_base_table T>
  requires(!soa::is_iterator<T>)
static constexpr std::string getLabelFromType()
{
  return getLabelForTable<typename aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>::metadata::base_table_t>();
}

template <typename... C>
static constexpr auto hasColumnForKey(framework::pack<C...>, std::string_view key)
{
  auto caseInsensitiveCompare = [](const std::string_view& str1, const std::string_view& str2) {
    return std::ranges::equal(
      str1, str2,
      [](char c1, char c2) {
        return std::tolower(static_cast<unsigned char>(c1)) ==
               std::tolower(static_cast<unsigned char>(c2));
      });
  };
  return (caseInsensitiveCompare(C::inherited_t::mLabel, key) || ...);
}

template <TableRef ref>
static constexpr std::pair<bool, std::string> hasKey(std::string_view key)
{
  return {hasColumnForKey(typename aod::MetadataTrait<o2::aod::Hash<ref.desc_hash>>::metadata::columns{}, key), aod::label<ref>()};
}

template <TableRef ref>
static constexpr std::pair<bool, framework::ConcreteDataMatcher> hasKeyM(std::string_view key)
{
  return {hasColumnForKey(typename aod::MetadataTrait<o2::aod::Hash<ref.desc_hash>>::metadata::columns{}, key), aod::matcher<ref>()};
}

void notFoundColumn(const char* label, const char* key);
void missingOptionalPreslice(const char* label, const char* key);

template <with_originals T, bool OPT = false>
static constexpr std::string getLabelFromTypeForKey(std::string_view key)
{
  auto locate = []<size_t... Is>(std::index_sequence<Is...>, std::string_view key) {
    return std::array{hasKey<T::originals[Is]>(key)...} |
           std::views::filter([](auto const& x) { return x.first; });
  }(std::make_index_sequence<T::originals.size()>{}, key);
  if (!locate.empty()) {
    return locate.front().second;
  }

  if constexpr (!OPT) {
    notFoundColumn(getLabelFromType<std::decay_t<T>>().data(), key.data());
  } else {
    return "[MISSING]";
  }
  O2_BUILTIN_UNREACHABLE();
}

template <with_originals T, bool OPT = false>
static constexpr framework::ConcreteDataMatcher getMatcherFromTypeForKey(std::string_view key)
{
  auto locate = []<size_t... Is>(std::index_sequence<Is...>, std::string_view key) {
    return std::array{hasKeyM<T::originals[Is]>(key)...} |
           std::views::filter([](auto const& x) { return x.first; });
  }(std::make_index_sequence<T::originals.size()>{}, key);
  if (!locate.empty()) {
    return locate.front().second;
  }

  if constexpr (!OPT) {
    notFoundColumn(getLabelFromType<std::decay_t<T>>().data(), key.data());
  } else {
    return framework::ConcreteDataMatcher{header::DataOrigin{"AOD"}, header::DataDescription{"[MISSING]"}, 0};
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
/// tracks origin in bindingKey matcher to handle the correct arguments
struct PreslicePolicyBase {
  const std::string binding;
  Entry bindingKey;

  bool isMissing() const;
  Entry const& getBindingKey() const;
};

struct PreslicePolicySorted : public PreslicePolicyBase {
  void updateSliceInfo(SliceInfoPtr&& si);

  SliceInfoPtr sliceInfo;
  std::shared_ptr<arrow::Table> getSliceFor(int value, std::shared_ptr<arrow::Table> const& input, uint64_t& offset) const;
  // One-slot cache for the empty (0-row) slice, so that empty groups do not
  // slice every column only to produce 0 rows (the common case for sparse
  // grouping, e.g. candidates per collision). Keyed by the input table, which
  // changes with every dataframe.
  mutable std::pair<arrow::Table const*, std::shared_ptr<arrow::Table>> emptySlice{nullptr, nullptr};
};

struct PreslicePolicyGeneral : public PreslicePolicyBase {
  void updateSliceInfo(SliceInfoUnsortedPtr&& si);

  SliceInfoUnsortedPtr sliceInfo;
  std::span<const int64_t> getSliceFor(int value) const;
};

template <typename T>
concept is_preslice_policy = std::derived_from<T, PreslicePolicyBase>;

template <soa::is_table T, is_preslice_policy Policy, bool OPT = false>
struct PresliceBase : public Policy {
  constexpr static bool optional = OPT;
  using target_t = T;
  using policy_t = Policy;
  const std::string binding;

  PresliceBase(expressions::BindingNode index_)
    : Policy{PreslicePolicyBase{{o2::soa::getLabelFromTypeForKey<T, OPT>(std::string{index_.name})}, Entry(o2::soa::getLabelFromTypeForKey<T, OPT>(std::string{index_.name}), o2::soa::getMatcherFromTypeForKey<T, OPT>(std::string{index_.name}), std::string{index_.name})}, {}}
  {
  }

  std::shared_ptr<arrow::Table> getSliceFor(int value, std::shared_ptr<arrow::Table> const& input, uint64_t& offset) const
  {
    if constexpr (OPT) {
      if (Policy::isMissing()) {
        return nullptr;
      }
    }
    return Policy::getSliceFor(value, input, offset);
  }

  std::span<const int64_t> getSliceFor(int value) const
  {
    if constexpr (OPT) {
      if (Policy::isMissing()) {
        return {};
      }
    }
    return Policy::getSliceFor(value);
  }
};

template <soa::is_table T>
using PresliceUnsorted = PresliceBase<T, PreslicePolicyGeneral, false>;
template <soa::is_table T>
using PresliceUnsortedOptional = PresliceBase<T, PreslicePolicyGeneral, true>;
template <soa::is_table T>
using Preslice = PresliceBase<T, PreslicePolicySorted, false>;
template <soa::is_table T>
using PresliceOptional = PresliceBase<T, PreslicePolicySorted, true>;

template <typename T>
concept is_preslice = std::derived_from<T, PreslicePolicyBase>&&
  requires(T)
{
  T::optional;
};

/// Can be user to group together a number of Preslice declaration
/// to avoid the limit of 100 data members per task
///
/// struct MyTask
///   struct : public PresliceGroup {
///     Preslice<aod::Tracks> perCol = aod::track::collisonId;
///     Preslice<aod::McParticles> perMcCol = aod::mcparticle::mcCollisionId;
///   } preslices;
///
/// individual components can be access with
///
/// preslices.perCol;
struct PresliceGroup {
};

template <typename T>
concept is_preslice_group = std::derived_from<T, PresliceGroup>;

} // namespace o2::framework

namespace o2::soa
{
template <soa::is_table T>
class FilteredBase;
template <typename T>
class Filtered;

template <typename T>
concept has_filtered_policy = not_void<typename T::policy_t> && std::same_as<typename T::policy_t, soa::FilteredIndexPolicy>;

template <typename T>
concept is_filtered_iterator = is_iterator<T> && has_filtered_policy<T>;

template <typename T>
concept is_filtered_table = framework::base_of_template<soa::FilteredBase, T>;

// FIXME: compatbility declaration to be removed
template <typename T>
constexpr bool is_soa_filtered_v = is_filtered_table<T>;

template <typename T>
concept is_filtered = is_filtered_table<T> || is_filtered_iterator<T>;

template <typename T>
concept is_not_filtered_table = is_table<T> && !is_filtered_table<T>;

/// Helper function to extract bound indices
template <typename... Is>
static consteval auto extractBindings(framework::pack<Is...>)
{
  return framework::pack<typename Is::binding_t...>{};
}

SelectionVector selectionToVector(gandiva::Selection const& sel);

template <typename T, typename C, typename Policy, bool OPT>
  requires std::same_as<Policy, framework::PreslicePolicySorted> && (o2::soa::is_binding_compatible_v<C, T>())
auto doSliceBy(T const* table, o2::framework::PresliceBase<C, Policy, OPT> const& container, int value)
{
  if constexpr (OPT) {
    if (container.isMissing()) {
      missingOptionalPreslice(getLabelFromType<std::decay_t<T>>().data(), container.bindingKey.key.c_str());
    }
  }
  uint64_t offset = 0;
  auto out = container.getSliceFor(value, table->asArrowTable(), offset);
  auto t = typename T::self_t({out}, offset);
  if (t.tableSize() != 0) {
    table->copyIndexBindings(t);
    t.bindInternalIndicesTo(table);
  }
  return t;
}

template <soa::is_filtered_table T>
auto doSliceByHelper(T const* table, std::span<const int64_t> const& selection)
{
  auto t = soa::Filtered<typename T::base_t>({table->asArrowTable()}, selection);
  if (t.tableSize() != 0) {
    table->copyIndexBindings(t);
    t.bindInternalIndicesTo(table);
    t.intersectWithSelection(table->getSelectedRows()); // intersect filters
  }
  return t;
}

template <soa::is_table T>
  requires(!soa::is_filtered_table<T>)
auto doSliceByHelper(T const* table, std::span<const int64_t> const& selection)
{
  auto t = soa::Filtered<T>({table->asArrowTable()}, selection);
  if (t.tableSize() != 0) {
    table->copyIndexBindings(t);
    t.bindInternalIndicesTo(table);
  }
  return t;
}

template <typename T, typename C, typename Policy, bool OPT>
  requires std::same_as<Policy, framework::PreslicePolicyGeneral> && (o2::soa::is_binding_compatible_v<C, T>())
auto doSliceBy(T const* table, o2::framework::PresliceBase<C, Policy, OPT> const& container, int value)
{
  if constexpr (OPT) {
    if (container.isMissing()) {
      missingOptionalPreslice(getLabelFromType<std::decay_t<T>>().data(), container.bindingKey.key.c_str());
    }
  }
  auto selection = container.getSliceFor(value);
  return doSliceByHelper(table, selection);
}

SelectionVector sliceSelection(std::span<int64_t const> const& mSelectedRows, int64_t nrows, uint64_t offset);

template <soa::is_filtered_table T>
auto prepareFilteredSlice(T const* table, std::shared_ptr<arrow::Table> slice, uint64_t offset)
{
  if (offset >= static_cast<uint64_t>(table->tableSize())) {
    Filtered<typename T::base_t> fresult{{{slice}}, SelectionVector{}, 0};
    if (fresult.tableSize() != 0) {
      table->copyIndexBindings(fresult);
    }
    return fresult;
  }
  auto slicedSelection = sliceSelection(table->getSelectedRows(), slice->num_rows(), offset);
  Filtered<typename T::base_t> fresult{{{slice}}, std::move(slicedSelection), offset};
  if (fresult.tableSize() != 0) {
    table->copyIndexBindings(fresult);
  }
  return fresult;
}

template <soa::is_filtered_table T, typename C, bool OPT>
  requires(o2::soa::is_binding_compatible_v<C, T>())
auto doFilteredSliceBy(T const* table, o2::framework::PresliceBase<C, framework::PreslicePolicySorted, OPT> const& container, int value)
{
  if constexpr (OPT) {
    if (container.isMissing()) {
      missingOptionalPreslice(getLabelFromType<T>().data(), container.bindingKey.key.c_str());
    }
  }
  uint64_t offset = 0;
  auto slice = container.getSliceFor(value, table->asArrowTable(), offset);
  return prepareFilteredSlice(table, slice, offset);
}

std::function<framework::ConcreteDataMatcher(framework::ConcreteDataMatcher&&)> originReplacement(header::DataOrigin newOrigin);

template <soa::is_table T>
auto doSliceByCached(T const* table, framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache)
{
  auto localCache = cache.ptr->getCacheFor({"", originReplacement(cache.ptr->newOrigin)(o2::soa::getMatcherFromTypeForKey<T>(node.name)),
                                            node.name});
  auto [offset, count] = localCache.getSliceFor(value);
  // Empty group: reuse a cached empty (0-row) table instead of slicing every column.
  auto slice = count == 0 ? cache.ptr->getEmptySliceFor(table->asArrowTable())
                          : table->asArrowTable()->Slice(static_cast<uint64_t>(offset), count);
  auto t = typename T::self_t({slice}, static_cast<uint64_t>(offset));
  if (t.tableSize() != 0) {
    table->copyIndexBindings(t);
  }
  return t;
}

template <soa::is_filtered_table T>
auto doFilteredSliceByCached(T const* table, framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache)
{
  auto localCache = cache.ptr->getCacheFor({"", originReplacement(cache.ptr->newOrigin)(o2::soa::getMatcherFromTypeForKey<T>(node.name)),
                                            node.name});
  auto [offset, count] = localCache.getSliceFor(value);
  // Empty group: reuse a cached empty (0-row) table instead of slicing every column.
  auto slice = count == 0 ? cache.ptr->getEmptySliceFor(table->asArrowTable())
                          : table->asArrowTable()->Slice(static_cast<uint64_t>(offset), count);
  return prepareFilteredSlice(table, slice, offset);
}

template <soa::is_table T>
auto doSliceByCachedUnsorted(T const* table, framework::expressions::BindingNode const& node, int value, o2::framework::SliceCache& cache)
{
  auto localCache = cache.ptr->getCacheUnsortedFor({"", originReplacement(cache.ptr->newOrigin)(o2::soa::getMatcherFromTypeForKey<T>(node.name)),
                                                    node.name});
  if constexpr (soa::is_filtered_table<T>) {
    auto t = typename T::self_t({table->asArrowTable()}, localCache.getSliceFor(value));
    if (t.tableSize() != 0) {
      t.intersectWithSelection(table->getSelectedRows());
      table->copyIndexBindings(t);
    }
    return t;
  } else {
    auto t = Filtered<T>({table->asArrowTable()}, localCache.getSliceFor(value));
    if (t.tableSize() != 0) {
      table->copyIndexBindings(t);
    }
    return t;
  }
}

template <with_originals T>
auto select(T const& t, framework::expressions::Filter const& f)
{
  return Filtered<T>({t.asArrowTable()}, selectionToVector(framework::expressions::createSelection(t.asArrowTable(), f)));
}

arrow::ChunkedArray* getIndexFromLabel(arrow::Table* table, std::string_view label);

template <typename D, typename O, typename IP, typename... C>
consteval auto base_iter(framework::pack<C...>&&) -> TableIterator<D, O, IP, C...>
{
}

template <TableRef ref, typename... Ts>
  requires((sizeof...(Ts) > 0) && (soa::is_column<Ts> && ...))
consteval auto getColumns()
{
  return framework::pack<Ts...>{};
}

template <TableRef ref, typename... Ts>
  requires((sizeof...(Ts) > 0) && !(soa::is_column<Ts> || ...) && (ref.origin_hash == "CONC"_h))
consteval auto getColumns()
{
  return framework::full_intersected_pack_t<typename Ts::columns_t...>{};
}

template <TableRef ref, typename... Ts>
  requires((sizeof...(Ts) > 0) && !(soa::is_column<Ts> || ...) && (ref.origin_hash != "CONC"_h))
consteval auto getColumns()
{
  return framework::concatenated_pack_unique_t<typename Ts::columns_t...>{};
}

template <TableRef ref, typename... Ts>
  requires(sizeof...(Ts) == 0 && soa::has_metadata<aod::MetadataTrait<o2::aod::Hash<ref.desc_hash>>>)
consteval auto getColumns()
{
  return typename aod::MetadataTrait<o2::aod::Hash<ref.desc_hash>>::metadata::columns{};
}

template <TableRef ref, typename... Ts>
  requires((sizeof...(Ts) == 0) || (o2::soa::is_column<Ts> && ...))
consteval auto computeOriginals()
{
  return std::array<TableRef, 1>{ref};
}

template <TableRef ref, typename... Ts>
  requires((sizeof...(Ts) > 0) && (!(o2::soa::is_column<Ts> && ...)))
consteval auto computeOriginals()
{
  return o2::soa::mergeOriginals<Ts...>();
}

/// A Table class which observes an arrow::Table and provides
/// It is templated on a set of Column / DynamicColumn types.
template <aod::is_aod_hash L, aod::is_aod_hash D, aod::is_origin_hash O, typename... Ts>
class Table
{
 public:
  static constexpr const auto ref = TableRef{L::hash, D::hash, O::hash, o2::aod::version(D::str)};
  using self_t = Table<L, D, O, Ts...>;
  using table_t = self_t;

  static constexpr const auto originals = computeOriginals<ref, Ts...>();
  static constexpr const auto originalLabels = []<size_t N, std::array<TableRef, N> refs, size_t... Is>(std::index_sequence<Is...>) {
    return std::array<const char*, N>{o2::aod::label<refs[Is]>()...};
  }.template operator()<originals.size(), originals>(std::make_index_sequence<originals.size()>());
  static constexpr const uint32_t binding_origin = originals[0].origin_hash; // commonOrigin<originals.size(), originals>();
  static constexpr header::DataOrigin binding_origin_ = o2::aod::Hash<binding_origin>::origin;

  template <size_t N, std::array<TableRef, N> bindings>
    requires(ref.origin_hash == "CONC"_h)
  static consteval auto isIndexTargetOf()
  {
    return false;
  }

  template <size_t N, std::array<TableRef, N> bindings>
    requires(ref.origin_hash == "JOIN"_h)
  static consteval auto isIndexTargetOf()
  {
    return std::ranges::any_of(self_t::originals,
                               [](TableRef const& r) {
                                 return std::ranges::any_of(bindings, [&r](TableRef const& b) { return b == r; });
                               });
  }

  template <size_t N, std::array<TableRef, N> bindings>
    requires(!(ref.origin_hash == "CONC"_h || ref.origin_hash == "JOIN"_h))
  static consteval auto isIndexTargetOf()
  {
    return std::find(bindings.begin(), bindings.end(), self_t::ref) != bindings.end();
  }

  template <TableRef r>
  static consteval bool hasOriginal()
  {
    return std::ranges::any_of(originals, [](TableRef const& o) { return o.desc_hash == r.desc_hash; });
  }

  using columns_t = decltype(getColumns<ref, Ts...>());

  using persistent_columns_t = decltype([]<typename... C>(framework::pack<C...>&&) -> framework::selected_pack<soa::is_persistent_column_t, C...> {}(columns_t{}));
  using column_types = decltype([]<typename... C>(framework::pack<C...>) -> framework::pack<typename C::type...> {}(persistent_columns_t{}));

  using external_index_columns_t = decltype([]<typename... C>(framework::pack<C...>&&) -> framework::selected_pack<soa::is_external_index_t, C...> {}(columns_t{}));
  using internal_index_columns_t = decltype([]<typename... C>(framework::pack<C...>&&) -> framework::selected_pack<soa::is_self_index_t, C...> {}(columns_t{}));
  template <typename IP>
  using base_iterator = decltype(base_iter<D, O, IP>(columns_t{}));

  template <typename IP, typename Parent, typename... T>
  struct TableIteratorBase : base_iterator<IP> {
    using columns_t = typename Parent::columns_t;
    using external_index_columns_t = typename Parent::external_index_columns_t;
    using bindings_pack_t = decltype([]<typename... C>(framework::pack<C...>) -> framework::pack<typename C::binding_t...> {}(external_index_columns_t{}));
    // static constexpr const std::array<TableRef, sizeof...(T)> originals{T::ref...};
    static constexpr auto originals = Parent::originals;
    using policy_t = IP;
    using parent_t = Parent;

    TableIteratorBase() = default;

    TableIteratorBase(arrow::ChunkedArray* columnData[framework::pack_size(columns_t{})], IP&& policy)
      : base_iterator<IP>(columnData, std::forward<decltype(policy)>(policy))
    {
    }

    template <typename P, typename... Os>
    TableIteratorBase& operator=(TableIteratorBase<IP, P, Os...> other)
      requires(P::ref.desc_hash == Parent::ref.desc_hash)
    {
      static_cast<base_iterator<IP>&>(*this) = static_cast<base_iterator<IP>>(other);
      return *this;
    }

    template <typename P>
    TableIteratorBase& operator=(TableIteratorBase<IP, P, T...> other)
    {
      static_cast<base_iterator<IP>&>(*this) = static_cast<base_iterator<IP>>(other);
      return *this;
    }

    template <typename P>
    TableIteratorBase& operator=(TableIteratorBase<FilteredIndexPolicy, P, T...> other)
      requires std::same_as<IP, DefaultIndexPolicy>
    {
      static_cast<base_iterator<IP>&>(*this) = static_cast<base_iterator<FilteredIndexPolicy>>(other);
      return *this;
    }

    template <typename P, typename O1, typename... Os>
    TableIteratorBase(TableIteratorBase<IP, P, O1, Os...> const& other)
      requires(P::ref.desc_hash == Parent::ref.desc_hash)
    {
      *this = other;
    }

    template <typename P, typename O1, typename... Os>
    TableIteratorBase(TableIteratorBase<IP, P, O1, Os...>&& other) noexcept
      requires(P::ref.desc_hash == Parent::ref.desc_hash)
    {
      *this = other;
    }

    template <typename P>
    TableIteratorBase(TableIteratorBase<IP, P, T...> const& other)
    {
      *this = other;
    }

    template <typename P>
    TableIteratorBase(TableIteratorBase<IP, P, T...>&& other) noexcept
    {
      *this = other;
    }

    template <typename P>
    TableIteratorBase(TableIteratorBase<FilteredIndexPolicy, P, T...> other)
      requires std::same_as<IP, DefaultIndexPolicy>
    {
      *this = other;
    }

    TableIteratorBase& operator=(RowViewSentinel const& other)
    {
      this->mRowIndex = other.index;
      return *this;
    }
    template <typename P>
    void matchTo(TableIteratorBase<IP, P, T...> const& other)
    {
      this->mRowIndex = other.mRowIndex;
    }

    template <typename P, typename... Os>
    void matchTo(TableIteratorBase<IP, P, Os...> const& other)
      requires std::same_as<typename P::table_t, typename Parent::table_t>
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
      } else if constexpr (std::same_as<decayed, Parent>) { // self index
        return this->globalIndex();
      } else if constexpr (is_indexing_column<decayed>) { // soa::Index<>
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
      static_assert(is_dynamic_t<COL>() || soa::is_persistent_column<COL>, "Should be persistent or dynamic column with no argument that has a return type convertable to float");
      return static_cast<B>(static_cast<COL>(*this).get());
    }

    template <typename B, typename... CCs>
    std::array<B, sizeof...(CCs)> getValues() const
    {
      static_assert(std::same_as<B, float> || std::same_as<B, double>, "The common return type should be float or double");
      return {getValue<B, CCs>()...};
    }

    using IP::size;

    using base_iterator<IP>::operator++;

    /// Allow incrementing by more than one the iterator
    TableIteratorBase operator+(int64_t inc) const
    {
      TableIteratorBase copy = *this;
      copy.moveByIndex(inc);
      return copy;
    }

    TableIteratorBase operator-(int64_t dec) const
    {
      return operator+(-dec);
    }

    TableIteratorBase const& operator*() const
    {
      return *this;
    }
  };

  template <typename IP, typename Parent, typename... T>
  using iterator_template = TableIteratorBase<IP, Parent, T...>;

  template <typename IP, typename Parent>
  static consteval auto full_iter()
  {
    if constexpr (sizeof...(Ts) == 0) {
      return iterator_template<IP, Parent>{};
    } else {
      if constexpr ((o2::soa::is_column<Ts> && ...)) {
        return iterator_template<IP, Parent>{};
      } else {
        return iterator_template<IP, Parent, Ts...>{};
      }
    }
  }

  template <typename IP, typename Parent>
  using iterator_template_o = decltype(full_iter<IP, Parent>());

  using iterator = iterator_template_o<DefaultIndexPolicy, table_t>;
  using filtered_iterator = iterator_template_o<FilteredIndexPolicy, table_t>;

  using unfiltered_iterator = iterator;
  using const_iterator = iterator;
  using unfiltered_const_iterator = unfiltered_iterator;

  static constexpr auto hashes()
  {
    return []<typename... C>(framework::pack<C...>) { return std::set{{C::hash...}}; }(columns_t{});
  }

  Table(std::shared_ptr<arrow::Table> table, uint64_t offset = 0)
    : mTable(table),
      mOffset(offset),
      mEnd{table->num_rows()}
  {
    if (mTable->num_rows() == 0) {
      for (size_t ci = 0; ci < framework::pack_size(columns_t{}); ++ci) {
        mColumnChunks[ci] = nullptr;
      }
      mBegin = mEnd;
    } else {
      auto lookups = [this]<typename... C>(framework::pack<C...>) -> std::array<arrow::ChunkedArray*, framework::pack_size(columns_t{})> { return {lookupColumn<C>()...}; }(columns_t{});
      for (size_t ci = 0; ci < framework::pack_size(columns_t{}); ++ci) {
        mColumnChunks[ci] = lookups[ci];
      }
      mBegin = unfiltered_iterator{mColumnChunks, {table->num_rows(), offset}};
      mBegin.bindInternalIndices(this);
    }
  }

  Table(std::vector<std::shared_ptr<arrow::Table>>&& tables, uint64_t offset = 0)
    requires(ref.origin_hash != "CONC"_h)
    : Table(ArrowHelpers::joinTables(std::move(tables), std::span{originalLabels}), offset)
  {
  }

  Table(std::vector<std::shared_ptr<arrow::Table>>&& tables, uint64_t offset = 0)
    requires(ref.origin_hash == "CONC"_h)
    : Table(ArrowHelpers::concatTables(std::move(tables)), offset)
  {
  }

  template <typename Key>
  inline arrow::ChunkedArray* getIndexToKey()
  {
    constexpr auto map = []<typename... Cs>(framework::pack<Cs...>) {
      return std::array<bool, sizeof...(Cs)>{[]() {
        if constexpr (requires { Cs::index_targets.size(); }) {
          return Key::template isIndexTargetOf<Cs::index_targets.size(), Cs::index_targets>();
        } else {
          return false;
        }
      }()...};
    }(persistent_columns_t{});
    constexpr auto pos = std::find(map.begin(), map.end(), true);
    if constexpr (pos != map.end()) {
      return mColumnChunks[std::distance(map.begin(), pos)];
    } else {
      static_assert(framework::always_static_assert_v<Key>, "This table does not have an index to given Key");
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

  filtered_iterator filtered_begin(std::span<int64_t const> selection)
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
    ([this](TA* cur) {
      if constexpr (binding_origin == TA::binding_origin) {
        mBegin.bindExternalIndex(cur);
      }
    }(current),
     ...);
  }

  template <typename TA>
  void bindExternalIndex(TA* current)
  {
    mBegin.bindExternalIndex(current); // unchecked binding for the derived tables
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

  template <typename T1, typename Policy, bool OPT>
  auto sliceBy(o2::framework::PresliceBase<T1, Policy, OPT> const& container, int value) const
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

 private:
  template <typename T>
  arrow::ChunkedArray* lookupColumn()
  {
    if constexpr (soa::is_persistent_column<T>) {
      auto label = T::columnLabel();
      return getIndexFromLabel(mTable.get(), label);
    } else {
      return nullptr;
    }
  }
  std::shared_ptr<arrow::Table> mTable = nullptr;
  uint64_t mOffset = 0;
  // Cached pointers to the ChunkedArray associated to a column
  arrow::ChunkedArray* mColumnChunks[framework::pack_size(columns_t{})];
  RowViewSentinel mEnd;
  iterator mBegin;
};

template <uint32_t D, soa::is_column... C>
using InPlaceTable = Table<o2::aod::Hash<"TEST"_h>, o2::aod::Hash<D>, o2::aod::Hash<"TEST"_h>, C...>;

void getterNotFound(const char* targetColumnLabel);
void emptyColumnLabel();

namespace row_helpers
{
template <typename R, typename T, typename C>
R getColumnValue(const T& rowIterator)
{
  return static_cast<R>(static_cast<C>(rowIterator).get());
}

namespace
{
template <typename R, typename T>
using ColumnGetterFunction = R (*)(const T&);

template <typename T, typename R>
concept dynamic_with_common_getter = is_dynamic_column<T> &&
                                     // lambda is callable without additional free args
                                     framework::pack_size(typename T::bindings_t{}) == framework::pack_size(typename T::callable_t::args{}) &&
                                     requires(T t) {
                                       { t.get() } -> std::convertible_to<R>;
                                     };

template <typename T, typename R>
concept persistent_with_common_getter = is_persistent_v<T> && requires(T t) {
  { t.get() } -> std::convertible_to<R>;
};

template <typename R, typename T, persistent_with_common_getter<R> C>
ColumnGetterFunction<R, T> createGetterPtr(const std::string_view& targetColumnLabel)
{
  return targetColumnLabel == C::columnLabel() ? &getColumnValue<R, T, C> : nullptr;
}

template <typename R, typename T, dynamic_with_common_getter<R> C>
ColumnGetterFunction<R, T> createGetterPtr(const std::string_view& targetColumnLabel)
{
  std::string_view columnLabel(C::columnLabel());

  // allows user to use consistent formatting (with prefix) of all column labels
  // by default there isn't 'f' prefix for dynamic column labels
  if (targetColumnLabel.starts_with("f") && targetColumnLabel.substr(1) == columnLabel) {
    return &getColumnValue<R, T, C>;
  }

  // check also exact match if user is aware of prefix missing
  if (targetColumnLabel == columnLabel) {
    return &getColumnValue<R, T, C>;
  }

  return nullptr;
}

template <typename R, typename T, typename... Cs>
ColumnGetterFunction<R, T> getColumnGetterByLabel(o2::framework::pack<Cs...>, const std::string_view& targetColumnLabel)
{
  ColumnGetterFunction<R, T> func;

  (void)((func = createGetterPtr<R, T, Cs>(targetColumnLabel), func) || ...);

  if (!func) {
    getterNotFound(targetColumnLabel.data());
  }

  return func;
}

template <typename T, typename R>
using with_common_getter_t = typename std::conditional<persistent_with_common_getter<T, R> || dynamic_with_common_getter<T, R>, std::true_type, std::false_type>::type;
} // namespace

template <typename R, typename T>
ColumnGetterFunction<R, typename T::iterator> getColumnGetterByLabel(const std::string_view& targetColumnLabel)
{
  using TypesWithCommonGetter = o2::framework::selected_pack_multicondition<with_common_getter_t, framework::pack<R>, typename T::columns_t>;

  if (targetColumnLabel.size() == 0) {
    emptyColumnLabel();
  }

  return getColumnGetterByLabel<R, typename T::iterator>(TypesWithCommonGetter{}, targetColumnLabel);
}
} // namespace row_helpers
} // namespace o2::soa

namespace o2::aod
{
// If you get an error about not satisfying is_origin_hash, you need to add
// an entry here.
O2ORIGIN("AOD");
O2ORIGIN("AOD1");
O2ORIGIN("AOD2");
// O2ORIGIN("DYN");
// O2ORIGIN("IDX");
// O2ORIGIN("ATIM");
O2ORIGIN("JOIN");
O2HASH("JOIN/0");
O2ORIGIN("CONC");
O2HASH("CONC/0");
O2ORIGIN("TEST");
O2HASH("TEST/0");
} // namespace o2::aod

namespace
{
template <typename T>
consteval static std::string_view namespace_prefix()
{
  constexpr auto name = o2::framework::type_name<T>();
  const auto pos = name.rfind(std::string_view{":"});
  return name.substr(0, pos + 1);
}
} // namespace

#define DECLARE_EQUIVALENT_FOR_INDEX(_Base_, _Equiv_)                                                     \
  template <>                                                                                             \
  struct EquivalentIndexNG<o2::aod::Hash<_Base_::ref.desc_hash>, o2::aod::Hash<_Equiv_::ref.desc_hash>> { \
    constexpr static bool value = true;                                                                   \
  }

#define DECLARE_EQUIVALENT_FOR_INDEX_NG(_Base_, _Equiv_)                              \
  template <>                                                                         \
  struct EquivalentIndexNG<o2::aod::Hash<_Base_ ""_h>, o2::aod::Hash<_Equiv_ ""_h>> { \
    constexpr static bool value = true;                                               \
  }

#define DECLARE_SOA_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_)                                                                                                                \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {                                                                                                                               \
    static constexpr const char* mLabel = _Label_;                                                                                                                                \
    static constexpr const uint32_t hash = compile_time_hash(namespace_prefix<_Name_>(), std::string_view{#_Getter_});                                                            \
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
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_ { _Label_, _Name_::hash, o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_CCDB_COLUMN_FULL(_Name_, _Label_, _Getter_, _ConcreteType_, _CCDBQuery_)                      \
  struct _Name_ : o2::soa::Column<std::span<std::byte>, _Name_> {                                                 \
    static constexpr const char* mLabel = _Label_;                                                                \
    static constexpr const char* query = _CCDBQuery_;                                                             \
    static constexpr const uint32_t hash = crc32(namespace_prefix<_Name_>(), std::string_view{#_Getter_});        \
    using base = o2::soa::Column<std::span<std::byte>, _Name_>;                                                   \
    using type = std::span<std::byte>;                                                                            \
    using column_t = _Name_;                                                                                      \
    _Name_(arrow::ChunkedArray const* column)                                                                     \
      : o2::soa::Column<std::span<std::byte>, _Name_>(o2::soa::ColumnIterator<std::span<std::byte>>(column))      \
    {                                                                                                             \
    }                                                                                                             \
                                                                                                                  \
    _Name_() = default;                                                                                           \
    _Name_(_Name_ const& other) = default;                                                                        \
    _Name_& operator=(_Name_ const& other) = default;                                                             \
                                                                                                                  \
    decltype(auto) _Getter_() const                                                                               \
    {                                                                                                             \
      if constexpr (std::same_as<_ConcreteType_, std::span<std::byte>>) {                                         \
        return *mColumnIterator;                                                                                  \
      } else {                                                                                                    \
        static std::byte* payload = nullptr;                                                                      \
        static _ConcreteType_* deserialised = nullptr;                                                            \
        static TClass* c = TClass::GetClass(#_ConcreteType_);                                                     \
        auto span = *mColumnIterator;                                                                             \
        if (payload != (std::byte*)span.data()) {                                                                 \
          payload = (std::byte*)span.data();                                                                      \
          delete deserialised;                                                                                    \
          TBufferFile f(TBufferFile::EMode::kRead, span.size(), (char*)span.data(), kFALSE);                      \
          deserialised = (_ConcreteType_*)soa::extractCCDBPayload((char*)payload, span.size(), c, "ccdb_object"); \
        }                                                                                                         \
        return *deserialised;                                                                                     \
      }                                                                                                           \
    }                                                                                                             \
                                                                                                                  \
    decltype(auto)                                                                                                \
      get() const                                                                                                 \
    {                                                                                                             \
      return _Getter_();                                                                                          \
    }                                                                                                             \
  };

#define DECLARE_SOA_CCDB_COLUMN(_Name_, _Getter_, _ConcreteType_, _CCDBQuery_) \
  DECLARE_SOA_CCDB_COLUMN_FULL(_Name_, "f" #_Name_, _Getter_, _ConcreteType_, _CCDBQuery_)

#define DECLARE_SOA_COLUMN(_Name_, _Getter_, _Type_) \
  DECLARE_SOA_COLUMN_FULL(_Name_, _Getter_, _Type_, "f" #_Name_)

/// A 'bitmap' column, i.e. a int-based column with custom accessors to check
/// individual bits
#define MAKEINT(_Size_) uint##_Size_##_t

#define DECLARE_SOA_BITMAP_COLUMN_FULL(_Name_, _Getter_, _Size_, _Label_)                                                                                                         \
  struct _Name_ : o2::soa::Column<MAKEINT(_Size_), _Name_> {                                                                                                                      \
    static constexpr const char* mLabel = _Label_;                                                                                                                                \
    static constexpr const uint32_t hash = compile_time_hash(namespace_prefix<_Name_>(), std::string_view{#_Getter_});                                                            \
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
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_ { _Label_, _Name_::hash, o2::framework::expressions::selectArrowType<MAKEINT(_Size_)>() }

#define DECLARE_SOA_BITMAP_COLUMN(_Name_, _Getter_, _Size_) \
  DECLARE_SOA_BITMAP_COLUMN_FULL(_Name_, _Getter_, _Size_, "f" #_Name_)

/// An 'expression' column. i.e. a column that can be calculated from other
/// columns with gandiva based on static C++ expression.
#define DECLARE_SOA_EXPRESSION_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_, _Expression_)                            \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {                                                                    \
    static constexpr const char* mLabel = _Label_;                                                                     \
    static constexpr const uint32_t hash = compile_time_hash(namespace_prefix<_Name_>(), std::string_view{#_Getter_}); \
    using base = o2::soa::Column<_Type_, _Name_>;                                                                      \
    using type = _Type_;                                                                                               \
    using column_t = _Name_;                                                                                           \
    using spawnable_t = std::true_type;                                                                                \
    _Name_(arrow::ChunkedArray const* column)                                                                          \
      : o2::soa::Column<_Type_, _Name_>(o2::soa::ColumnIterator<type>(column))                                         \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    _Name_() = default;                                                                                                \
    _Name_(_Name_ const& other) = default;                                                                             \
    _Name_& operator=(_Name_ const& other) = default;                                                                  \
                                                                                                                       \
    decltype(auto) _Getter_() const                                                                                    \
    {                                                                                                                  \
      return *mColumnIterator;                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    decltype(auto) get() const                                                                                         \
    {                                                                                                                  \
      return _Getter_();                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    static o2::framework::expressions::Projector Projector()                                                           \
    {                                                                                                                  \
      return _Expression_;                                                                                             \
    }                                                                                                                  \
  };                                                                                                                   \
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_ { _Label_, _Name_::hash, o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_EXPRESSION_COLUMN(_Name_, _Getter_, _Type_, _Expression_) \
  DECLARE_SOA_EXPRESSION_COLUMN_FULL(_Name_, _Getter_, _Type_, "f" #_Name_, _Expression_);

/// A configurable 'expression' column. i.e. a column that can be calculated from other
/// columns with gandiva based on dynamically supplied C++ expression or a string definition.
#define DECLARE_SOA_CONFIGURABLE_EXPRESSION_COLUMN(_Name_, _Getter_, _Type_, _Label_)                                  \
  struct _Name_ : o2::soa::Column<_Type_, _Name_> {                                                                    \
    static constexpr const char* mLabel = _Label_;                                                                     \
    static constexpr const uint32_t hash = compile_time_hash(namespace_prefix<_Name_>(), std::string_view{#_Getter_}); \
    static constexpr const int32_t mHash = _Label_ ""_h;                                                               \
    using base = o2::soa::Column<_Type_, _Name_>;                                                                      \
    using type = _Type_;                                                                                               \
    using column_t = _Name_;                                                                                           \
    using spawnable_t = std::true_type;                                                                                \
    _Name_(arrow::ChunkedArray const* column)                                                                          \
      : o2::soa::Column<_Type_, _Name_>(o2::soa::ColumnIterator<type>(column))                                         \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    _Name_() = default;                                                                                                \
    _Name_(_Name_ const& other) = default;                                                                             \
    _Name_& operator=(_Name_ const& other) = default;                                                                  \
                                                                                                                       \
    decltype(auto) _Getter_() const                                                                                    \
    {                                                                                                                  \
      return *mColumnIterator;                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    decltype(auto) get() const                                                                                         \
    {                                                                                                                  \
      return _Getter_();                                                                                               \
    }                                                                                                                  \
  };                                                                                                                   \
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_ { _Label_, _Name_::hash, o2::framework::expressions::selectArrowType<_Type_>() }

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

template <o2::soa::is_table T>
consteval auto getIndexTargets()
{
  return T::originals;
}

#define DECLARE_SOA_SLICE_INDEX_COLUMN_FULL_CUSTOM(_Name_, _Getter_, _Type_, _Table_, _Label_, _Suffix_) \
  struct _Name_##IdSlice : o2::soa::Column<_Type_[2], _Name_##IdSlice> {                                 \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");                            \
    static_assert((*_Suffix_ == '\0') || (*_Suffix_ == '_'), "Suffix has to begin with _");              \
    static constexpr const char* mLabel = "fIndexSlice" _Label_ _Suffix_;                                \
    static constexpr const uint32_t hash = 0;                                                            \
    using base = o2::soa::Column<_Type_[2], _Name_##IdSlice>;                                            \
    using type = _Type_[2];                                                                              \
    using column_t = _Name_##IdSlice;                                                                    \
    using binding_t = _Table_;                                                                           \
    static constexpr auto index_targets = getIndexTargets<_Table_>();                                    \
    _Name_##IdSlice(arrow::ChunkedArray const* column)                                                   \
      : o2::soa::Column<_Type_[2], _Name_##IdSlice>(o2::soa::ColumnIterator<type>(column))               \
    {                                                                                                    \
    }                                                                                                    \
                                                                                                         \
    _Name_##IdSlice() = default;                                                                         \
    _Name_##IdSlice(_Name_##IdSlice const& other) = default;                                             \
    _Name_##IdSlice& operator=(_Name_##IdSlice const& other) = default;                                  \
    std::array<_Type_, 2> inline getIds() const                                                          \
    {                                                                                                    \
      return _Getter_##Ids();                                                                            \
    }                                                                                                    \
                                                                                                         \
    bool has_##_Getter_() const                                                                          \
    {                                                                                                    \
      auto a = *mColumnIterator;                                                                         \
      return a[0] >= 0 && a[1] >= 0;                                                                     \
    }                                                                                                    \
                                                                                                         \
    std::array<_Type_, 2> _Getter_##Ids() const                                                          \
    {                                                                                                    \
      auto a = *mColumnIterator;                                                                         \
      return std::array{a[0], a[1]};                                                                     \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    auto _Getter_##_as() const                                                                           \
    {                                                                                                    \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                                \
        o2::soa::notBoundTable(#_Table_);                                                                \
      }                                                                                                  \
      auto t = mBinding.get<T>();                                                                        \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                           \
        o2::soa::dereferenceWithWrongType(#_Getter_, #_Table_);                                          \
      }                                                                                                  \
      if (O2_BUILTIN_UNLIKELY(!has_##_Getter_())) {                                                      \
        return t->emptySlice();                                                                          \
      }                                                                                                  \
      auto a = *mColumnIterator;                                                                         \
      auto r = t->rawSlice(a[0], a[1]);                                                                  \
      t->copyIndexBindings(r);                                                                           \
      r.bindInternalIndicesTo(t);                                                                        \
      return r;                                                                                          \
    }                                                                                                    \
                                                                                                         \
    auto _Getter_() const                                                                                \
    {                                                                                                    \
      return _Getter_##_as<binding_t>();                                                                 \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    bool setCurrent(T const* current)                                                                    \
    {                                                                                                    \
      if constexpr (o2::soa::is_binding_compatible_v<T, binding_t>()) {                                  \
        assert(current != nullptr);                                                                      \
        this->mBinding.bind(current);                                                                    \
        return true;                                                                                     \
      }                                                                                                  \
      return false;                                                                                      \
    }                                                                                                    \
                                                                                                         \
    bool setCurrentRaw(o2::soa::Binding current)                                                         \
    {                                                                                                    \
      this->mBinding = current;                                                                          \
      return true;                                                                                       \
    }                                                                                                    \
    binding_t const* getCurrent() const { return mBinding.get<binding_t>(); }                            \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                                          \
    o2::soa::Binding mBinding;                                                                           \
  };

#define DECLARE_SOA_SLICE_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Table_, _Suffix_) DECLARE_SOA_SLICE_INDEX_COLUMN_FULL_CUSTOM(_Name_, _Getter_, _Type_, _Table_, #_Table_, _Suffix_)
#define DECLARE_SOA_SLICE_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_SLICE_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, _Name_##s, "")
#define DECLARE_SOA_SLICE_INDEX_COLUMN_CUSTOM(_Name_, _Getter_, _Label_) DECLARE_SOA_SLICE_INDEX_COLUMN_FULL_CUSTOM(_Name_, _Getter_, int32_t, _Name_##s, _Label_, "")

/// ARRAY
#define DECLARE_SOA_ARRAY_INDEX_COLUMN_FULL_CUSTOM(_Name_, _Getter_, _Type_, _Table_, _Label_, _Suffix_) \
  struct _Name_##Ids : o2::soa::Column<std::vector<_Type_>, _Name_##Ids> {                               \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");                            \
    static_assert((*_Suffix_ == '\0') || (*_Suffix_ == '_'), "Suffix has to begin with _");              \
    static constexpr const char* mLabel = "fIndexArray" _Label_ _Suffix_;                                \
    static constexpr const uint32_t hash = 0;                                                            \
    using base = o2::soa::Column<std::vector<_Type_>, _Name_##Ids>;                                      \
    using type = std::vector<_Type_>;                                                                    \
    using column_t = _Name_##Ids;                                                                        \
    using binding_t = _Table_;                                                                           \
    static constexpr auto index_targets = getIndexTargets<_Table_>();                                    \
    _Name_##Ids(arrow::ChunkedArray const* column)                                                       \
      : o2::soa::Column<std::vector<_Type_>, _Name_##Ids>(o2::soa::ColumnIterator<type>(column))         \
    {                                                                                                    \
    }                                                                                                    \
                                                                                                         \
    _Name_##Ids() = default;                                                                             \
    _Name_##Ids(_Name_##Ids const& other) = default;                                                     \
    _Name_##Ids& operator=(_Name_##Ids const& other) = default;                                          \
                                                                                                         \
    gsl::span<const _Type_> inline getIds() const                                                        \
    {                                                                                                    \
      return _Getter_##Ids();                                                                            \
    }                                                                                                    \
                                                                                                         \
    gsl::span<const _Type_> _Getter_##Ids() const                                                        \
    {                                                                                                    \
      return *mColumnIterator;                                                                           \
    }                                                                                                    \
                                                                                                         \
    bool has_##_Getter_() const                                                                          \
    {                                                                                                    \
      return !(*mColumnIterator).empty();                                                                \
    }                                                                                                    \
                                                                                                         \
    template <soa::is_table T>                                                                           \
    auto _Getter_##_as() const                                                                           \
    {                                                                                                    \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                                \
        o2::soa::notBoundTable(#_Table_);                                                                \
      }                                                                                                  \
      auto t = mBinding.get<T>();                                                                        \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                           \
        o2::soa::dereferenceWithWrongType(#_Getter_, #_Table_);                                          \
      }                                                                                                  \
      auto result = std::vector<typename T::unfiltered_iterator>();                                      \
      result.reserve((*mColumnIterator).size());                                                         \
      for (auto& i : *mColumnIterator) {                                                                 \
        result.emplace_back(t->rawIteratorAt(i));                                                        \
      }                                                                                                  \
      return result;                                                                                     \
    }                                                                                                    \
                                                                                                         \
    template <soa::is_filtered_table T>                                                                  \
    auto filtered_##_Getter_##_as() const                                                                \
    {                                                                                                    \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                                \
        o2::soa::notBoundTable(#_Table_);                                                                \
      }                                                                                                  \
      auto t = mBinding.get<T>();                                                                        \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                           \
        o2::soa::dereferenceWithWrongType(#_Getter_, #_Table_);                                          \
      }                                                                                                  \
      auto result = std::vector<typename T::iterator>();                                                 \
      result.reserve((*mColumnIterator).size());                                                         \
      for (auto const& i : *mColumnIterator) {                                                           \
        auto pos = t->isInSelectedRows(i);                                                               \
        if (pos > 0) {                                                                                   \
          result.emplace_back(t->iteratorAt(pos));                                                       \
        }                                                                                                \
      }                                                                                                  \
      return result;                                                                                     \
    }                                                                                                    \
                                                                                                         \
    auto _Getter_() const                                                                                \
    {                                                                                                    \
      return _Getter_##_as<binding_t>();                                                                 \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    auto _Getter_##_first_as() const                                                                     \
    {                                                                                                    \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                                \
        o2::soa::notBoundTable(#_Table_);                                                                \
      }                                                                                                  \
      auto t = mBinding.get<T>();                                                                        \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                           \
        o2::soa::dereferenceWithWrongType(#_Getter_, #_Table_);                                          \
      }                                                                                                  \
      return t->rawIteratorAt((*mColumnIterator)[0]);                                                    \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    auto _Getter_##_last_as() const                                                                      \
    {                                                                                                    \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                                \
        o2::soa::notBoundTable(#_Table_);                                                                \
      }                                                                                                  \
      auto t = mBinding.get<T>();                                                                        \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                           \
        o2::soa::dereferenceWithWrongType(#_Getter_, #_Table_);                                          \
      }                                                                                                  \
      return t->rawIteratorAt((*mColumnIterator).back());                                                \
    }                                                                                                    \
                                                                                                         \
    auto _Getter_first() const                                                                           \
    {                                                                                                    \
      return _Getter_##_first_as<binding_t>();                                                           \
    }                                                                                                    \
                                                                                                         \
    auto _Getter_last() const                                                                            \
    {                                                                                                    \
      return _Getter_##_last_as<binding_t>();                                                            \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    bool setCurrent(T const* current)                                                                    \
    {                                                                                                    \
      if constexpr (o2::soa::is_binding_compatible_v<T, binding_t>()) {                                  \
        assert(current != nullptr);                                                                      \
        this->mBinding.bind(current);                                                                    \
        return true;                                                                                     \
      }                                                                                                  \
      return false;                                                                                      \
    }                                                                                                    \
                                                                                                         \
    bool setCurrentRaw(o2::soa::Binding current)                                                         \
    {                                                                                                    \
      this->mBinding = current;                                                                          \
      return true;                                                                                       \
    }                                                                                                    \
    binding_t const* getCurrent() const { return mBinding.get<binding_t>(); }                            \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                                          \
    o2::soa::Binding mBinding;                                                                           \
  };

#define DECLARE_SOA_ARRAY_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Table_, _Suffix_) DECLARE_SOA_ARRAY_INDEX_COLUMN_FULL_CUSTOM(_Name_, _Getter_, _Type_, _Table_, #_Table_, _Suffix_)
#define DECLARE_SOA_ARRAY_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_ARRAY_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, _Name_##s, "")
#define DECLARE_SOA_ARRAY_INDEX_COLUMN_CUSTOM(_Name_, _Getter_, _Label_) DECLARE_SOA_ARRAY_INDEX_COLUMN_FULL_CUSTOM(_Name_, _Getter_, int32_t, _Name_##s, _Label_, "")

/// NORMAL
#define DECLARE_SOA_INDEX_COLUMN_FULL_CUSTOM(_Name_, _Getter_, _Type_, _Table_, _Label_, _Suffix_)                              \
  struct _Name_##Id : o2::soa::Column<_Type_, _Name_##Id> {                                                                     \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");                                                   \
    static_assert((*_Suffix_ == '\0') || (*_Suffix_ == '_'), "Suffix has to begin with _");                                     \
    static constexpr const char* mLabel = "fIndex" _Label_ _Suffix_;                                                            \
    static constexpr const uint32_t hash = compile_time_hash(namespace_prefix<_Name_##Id>(), std::string_view{#_Getter_ "Id"}); \
    using base = o2::soa::Column<_Type_, _Name_##Id>;                                                                           \
    using type = _Type_;                                                                                                        \
    using column_t = _Name_##Id;                                                                                                \
    using binding_t = _Table_;                                                                                                  \
    static constexpr auto index_targets = getIndexTargets<_Table_>();                                                           \
    _Name_##Id(arrow::ChunkedArray const* column)                                                                               \
      : o2::soa::Column<_Type_, _Name_##Id>(o2::soa::ColumnIterator<type>(column))                                              \
    {                                                                                                                           \
    }                                                                                                                           \
                                                                                                                                \
    _Name_##Id() = default;                                                                                                     \
    _Name_##Id(_Name_##Id const& other) = default;                                                                              \
    _Name_##Id& operator=(_Name_##Id const& other) = default;                                                                   \
    type inline getId() const                                                                                                   \
    {                                                                                                                           \
      return _Getter_##Id();                                                                                                    \
    }                                                                                                                           \
                                                                                                                                \
    type _Getter_##Id() const                                                                                                   \
    {                                                                                                                           \
      return *mColumnIterator;                                                                                                  \
    }                                                                                                                           \
                                                                                                                                \
    bool has_##_Getter_() const                                                                                                 \
    {                                                                                                                           \
      return *mColumnIterator >= 0;                                                                                             \
    }                                                                                                                           \
                                                                                                                                \
    template <typename T>                                                                                                       \
    auto _Getter_##_as() const                                                                                                  \
    {                                                                                                                           \
      if (O2_BUILTIN_UNLIKELY(mBinding.ptr == nullptr)) {                                                                       \
        o2::soa::notBoundTable(#_Table_);                                                                                       \
      }                                                                                                                         \
      if (O2_BUILTIN_UNLIKELY(!has_##_Getter_())) {                                                                             \
        o2::soa::accessingInvalidIndexFor(#_Getter_);                                                                           \
      }                                                                                                                         \
      auto t = mBinding.get<T>();                                                                                               \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                                                  \
        o2::soa::dereferenceWithWrongType(#_Getter_, #_Table_);                                                                 \
      }                                                                                                                         \
      return t->rawIteratorAt(*mColumnIterator);                                                                                \
    }                                                                                                                           \
                                                                                                                                \
    auto _Getter_() const                                                                                                       \
    {                                                                                                                           \
      return _Getter_##_as<binding_t>();                                                                                        \
    }                                                                                                                           \
                                                                                                                                \
    template <typename T>                                                                                                       \
    bool setCurrent(T* current)                                                                                                 \
    {                                                                                                                           \
      if constexpr (o2::soa::is_binding_compatible_v<T, binding_t>()) {                                                         \
        assert(current != nullptr);                                                                                             \
        this->mBinding.bind(current);                                                                                           \
        return true;                                                                                                            \
      }                                                                                                                         \
      return false;                                                                                                             \
    }                                                                                                                           \
                                                                                                                                \
    bool setCurrentRaw(o2::soa::Binding current)                                                                                \
    {                                                                                                                           \
      this->mBinding = current;                                                                                                 \
      return true;                                                                                                              \
    }                                                                                                                           \
    binding_t const* getCurrent() const { return mBinding.get<binding_t>(); }                                                   \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                                                                 \
    o2::soa::Binding mBinding;                                                                                                  \
  };                                                                                                                            \
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_##Id { "fIndex" _Label_ _Suffix_, _Name_##Id::hash, o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Table_, _Suffix_) DECLARE_SOA_INDEX_COLUMN_FULL_CUSTOM(_Name_, _Getter_, _Type_, _Table_, #_Table_, _Suffix_)
#define DECLARE_SOA_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, _Name_##s, "")
#define DECLARE_SOA_INDEX_COLUMN_CUSTOM(_Name_, _Getter_, _Label_) DECLARE_SOA_INDEX_COLUMN_FULL_CUSTOM(_Name_, _Getter_, int32_t, _Name_##s, _Label_, "")

/// SELF
#define DECLARE_SOA_SELF_INDEX_COLUMN_COMPLETE(_Name_, _Getter_, _Type_, _Label_, _IndexTarget_)                                \
  struct _Name_##Id : o2::soa::Column<_Type_, _Name_##Id> {                                                                     \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");                                                   \
    static constexpr const char* mLabel = "fIndex" _Label_;                                                                     \
    static constexpr const uint32_t hash = compile_time_hash(namespace_prefix<_Name_##Id>(), std::string_view{#_Getter_ "Id"}); \
    using base = o2::soa::Column<_Type_, _Name_##Id>;                                                                           \
    using type = _Type_;                                                                                                        \
    using column_t = _Name_##Id;                                                                                                \
    using self_index_t = std::true_type;                                                                                        \
    using compatible_signature = std::conditional<aod::is_aod_hash<_IndexTarget_>, _IndexTarget_, void>;                        \
    _Name_##Id(arrow::ChunkedArray const* column)                                                                               \
      : o2::soa::Column<_Type_, _Name_##Id>(o2::soa::ColumnIterator<type>(column))                                              \
    {                                                                                                                           \
    }                                                                                                                           \
                                                                                                                                \
    _Name_##Id() = default;                                                                                                     \
    _Name_##Id(_Name_##Id const& other) = default;                                                                              \
    _Name_##Id& operator=(_Name_##Id const& other) = default;                                                                   \
    type inline getId() const                                                                                                   \
    {                                                                                                                           \
      return _Getter_##Id();                                                                                                    \
    }                                                                                                                           \
                                                                                                                                \
    type _Getter_##Id() const                                                                                                   \
    {                                                                                                                           \
      return *mColumnIterator;                                                                                                  \
    }                                                                                                                           \
                                                                                                                                \
    bool has_##_Getter_() const                                                                                                 \
    {                                                                                                                           \
      return *mColumnIterator >= 0;                                                                                             \
    }                                                                                                                           \
                                                                                                                                \
    template <typename T>                                                                                                       \
    auto _Getter_##_as() const                                                                                                  \
    {                                                                                                                           \
      if (O2_BUILTIN_UNLIKELY(!has_##_Getter_())) {                                                                             \
        o2::soa::accessingInvalidIndexFor(#_Getter_);                                                                           \
      }                                                                                                                         \
      auto t = mBinding.get<T>();                                                                                               \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                                                  \
        o2::soa::dereferenceWithWrongType(#_Getter_, "self");                                                                   \
      }                                                                                                                         \
      return t->rawIteratorAt(*mColumnIterator);                                                                                \
    }                                                                                                                           \
                                                                                                                                \
    bool setCurrentRaw(o2::soa::Binding current)                                                                                \
    {                                                                                                                           \
      this->mBinding = current;                                                                                                 \
      return true;                                                                                                              \
    }                                                                                                                           \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                                                                 \
    o2::soa::Binding mBinding;                                                                                                  \
  };                                                                                                                            \
  [[maybe_unused]] static constexpr o2::framework::expressions::BindingNode _Getter_##Id { "fIndex" _Label_, _Name_##Id::hash, o2::framework::expressions::selectArrowType<_Type_>() }

#define DECLARE_SOA_SELF_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_) DECLARE_SOA_SELF_INDEX_COLUMN_COMPLETE(_Name_, _Getter_, _Type_, _Label_, void)
#define DECLARE_SOA_SELF_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_SELF_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, #_Name_)
/// SELF SLICE
#define DECLARE_SOA_SELF_SLICE_INDEX_COLUMN_COMPLETE(_Name_, _Getter_, _Type_, _Label_, _IndexTarget_)   \
  struct _Name_##IdSlice : o2::soa::Column<_Type_[2], _Name_##IdSlice> {                                 \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");                            \
    static constexpr const char* mLabel = "fIndexSlice" _Label_;                                         \
    static constexpr const uint32_t hash = 0;                                                            \
    using base = o2::soa::Column<_Type_[2], _Name_##IdSlice>;                                            \
    using type = _Type_[2];                                                                              \
    using column_t = _Name_##IdSlice;                                                                    \
    using self_index_t = std::true_type;                                                                 \
    using compatible_signature = std::conditional<aod::is_aod_hash<_IndexTarget_>, _IndexTarget_, void>; \
    _Name_##IdSlice(arrow::ChunkedArray const* column)                                                   \
      : o2::soa::Column<_Type_[2], _Name_##IdSlice>(o2::soa::ColumnIterator<type>(column))               \
    {                                                                                                    \
    }                                                                                                    \
                                                                                                         \
    _Name_##IdSlice() = default;                                                                         \
    _Name_##IdSlice(_Name_##IdSlice const& other) = default;                                             \
    _Name_##IdSlice& operator=(_Name_##IdSlice const& other) = default;                                  \
    std::array<_Type_, 2> inline getIds() const                                                          \
    {                                                                                                    \
      return _Getter_##Ids();                                                                            \
    }                                                                                                    \
                                                                                                         \
    bool has_##_Getter_() const                                                                          \
    {                                                                                                    \
      auto a = *mColumnIterator;                                                                         \
      return a[0] >= 0 && a[1] >= 0;                                                                     \
    }                                                                                                    \
                                                                                                         \
    std::array<_Type_, 2> _Getter_##Ids() const                                                          \
    {                                                                                                    \
      auto a = *mColumnIterator;                                                                         \
      return std::array{a[0], a[1]};                                                                     \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    auto _Getter_##_as() const                                                                           \
    {                                                                                                    \
      auto t = mBinding.get<T>();                                                                        \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                           \
        o2::soa::dereferenceWithWrongType(#_Getter_, "self");                                            \
      }                                                                                                  \
      if (O2_BUILTIN_UNLIKELY(!has_##_Getter_())) {                                                      \
        return t->emptySlice();                                                                          \
      }                                                                                                  \
      auto a = *mColumnIterator;                                                                         \
      auto r = t->rawSlice(a[0], a[1]);                                                                  \
      t->copyIndexBindings(r);                                                                           \
      r.bindInternalIndicesTo(t);                                                                        \
      return r;                                                                                          \
    }                                                                                                    \
                                                                                                         \
    bool setCurrentRaw(o2::soa::Binding current)                                                         \
    {                                                                                                    \
      this->mBinding = current;                                                                          \
      return true;                                                                                       \
    }                                                                                                    \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                                          \
    o2::soa::Binding mBinding;                                                                           \
  };

#define DECLARE_SOA_SELF_SLICE_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_) DECLARE_SOA_SELF_SLICE_INDEX_COLUMN_COMPLETE(_Name_, _Getter_, _Type_, _Label_, void)
#define DECLARE_SOA_SELF_SLICE_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_SELF_SLICE_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, "_" #_Name_)
/// SELF ARRAY
#define DECLARE_SOA_SELF_ARRAY_INDEX_COLUMN_COMPLETE(_Name_, _Getter_, _Type_, _Label_, _IndexTarget_)   \
  struct _Name_##Ids : o2::soa::Column<std::vector<_Type_>, _Name_##Ids> {                               \
    static_assert(std::is_integral_v<_Type_>, "Index type must be integral");                            \
    static constexpr const char* mLabel = "fIndexArray" _Label_;                                         \
    static constexpr const uint32_t hash = 0;                                                            \
    using base = o2::soa::Column<std::vector<_Type_>, _Name_##Ids>;                                      \
    using type = std::vector<_Type_>;                                                                    \
    using column_t = _Name_##Ids;                                                                        \
    using self_index_t = std::true_type;                                                                 \
    using compatible_signature = std::conditional<aod::is_aod_hash<_IndexTarget_>, _IndexTarget_, void>; \
    _Name_##Ids(arrow::ChunkedArray const* column)                                                       \
      : o2::soa::Column<std::vector<_Type_>, _Name_##Ids>(o2::soa::ColumnIterator<type>(column))         \
    {                                                                                                    \
    }                                                                                                    \
                                                                                                         \
    _Name_##Ids() = default;                                                                             \
    _Name_##Ids(_Name_##Ids const& other) = default;                                                     \
    _Name_##Ids& operator=(_Name_##Ids const& other) = default;                                          \
    gsl::span<const _Type_> inline getIds() const                                                        \
    {                                                                                                    \
      return _Getter_##Ids();                                                                            \
    }                                                                                                    \
                                                                                                         \
    gsl::span<const _Type_> _Getter_##Ids() const                                                        \
    {                                                                                                    \
      return *mColumnIterator;                                                                           \
    }                                                                                                    \
                                                                                                         \
    bool has_##_Getter_() const                                                                          \
    {                                                                                                    \
      return !(*mColumnIterator).empty();                                                                \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    auto _Getter_##_as() const                                                                           \
    {                                                                                                    \
      auto t = mBinding.get<T>();                                                                        \
      if (O2_BUILTIN_UNLIKELY(t == nullptr)) {                                                           \
        o2::soa::dereferenceWithWrongType(#_Getter_, "self");                                            \
      }                                                                                                  \
      auto result = std::vector<typename T::unfiltered_iterator>();                                      \
      for (auto& i : *mColumnIterator) {                                                                 \
        result.push_back(t->rawIteratorAt(i));                                                           \
      }                                                                                                  \
      return result;                                                                                     \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    auto _Getter_##_first_as() const                                                                     \
    {                                                                                                    \
      return mBinding.get<T>()->rawIteratorAt((*mColumnIterator)[0]);                                    \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    auto _Getter_##_last_as() const                                                                      \
    {                                                                                                    \
      return mBinding.get<T>()->rawIteratorAt((*mColumnIterator).back());                                \
    }                                                                                                    \
                                                                                                         \
    bool setCurrentRaw(o2::soa::Binding current)                                                         \
    {                                                                                                    \
      this->mBinding = current;                                                                          \
      return true;                                                                                       \
    }                                                                                                    \
    o2::soa::Binding getCurrentRaw() const { return mBinding; }                                          \
    o2::soa::Binding mBinding;                                                                           \
  };

#define DECLARE_SOA_SELF_ARRAY_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Label_) DECLARE_SOA_SELF_ARRAY_INDEX_COLUMN_COMPLETE(_Name_, _Getter_, _Type_, _Label_, void)
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
    static constexpr const uint32_t hash = 0;                                                                              \
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

#define DECLARE_SOA_TABLE_METADATA(_Name_, _Desc_, _Version_, ...) \
  using _Name_##Metadata = TableMetadata<Hash<_Desc_ "/" #_Version_ ""_h>, __VA_ARGS__>;

#define DECLARE_SOA_TABLE_METADATA_TRAIT(_Name_, _Desc_, _Version_) \
  template <>                                                       \
  struct MetadataTrait<Hash<_Desc_ "/" #_Version_ ""_h>> {          \
    using metadata = _Name_##Metadata;                              \
  };

#define DECLARE_SOA_TABLE_FULL_VERSIONED_(_Name_, _Label_, _Origin_, _Desc_, _Version_)         \
  O2HASH(_Desc_ "/" #_Version_);                                                                \
  template <typename O>                                                                         \
  using _Name_##From = o2::soa::Table<Hash<_Label_ ""_h>, Hash<_Desc_ "/" #_Version_ ""_h>, O>; \
  using _Name_ = _Name_##From<Hash<_Origin_ ""_h>>;                                             \
  template <>                                                                                   \
  struct MetadataTrait<Hash<_Desc_ "/" #_Version_ ""_h>> {                                      \
    using metadata = _Name_##Metadata;                                                          \
  };

#define DECLARE_SOA_STAGE(_Name_, _Origin_, _Desc_, _Version_)                                  \
  template <typename O>                                                                         \
  using _Name_##From = o2::soa::Table<Hash<#_Name_ ""_h>, Hash<_Desc_ "/" #_Version_ ""_h>, O>; \
  using _Name_ = _Name_##From<Hash<_Origin_ ""_h>>;

#define DECLARE_SOA_TABLE_FULL_VERSIONED(_Name_, _Label_, _Origin_, _Desc_, _Version_, ...) \
  DECLARE_SOA_TABLE_METADATA(_Name_, _Desc_, _Version_, __VA_ARGS__);                       \
  DECLARE_SOA_TABLE_FULL_VERSIONED_(_Name_, _Label_, _Origin_, _Desc_, _Version_);

#define DECLARE_SOA_TABLE_FULL(_Name_, _Label_, _Origin_, _Desc_, ...) \
  O2HASH(_Label_);                                                     \
  DECLARE_SOA_TABLE_METADATA(_Name_, _Desc_, 0, __VA_ARGS__);          \
  DECLARE_SOA_TABLE_FULL_VERSIONED_(_Name_, _Label_, _Origin_, _Desc_, 0)

#define DECLARE_SOA_TABLE(_Name_, _Origin_, _Desc_, ...) \
  DECLARE_SOA_TABLE_FULL(_Name_, #_Name_, _Origin_, _Desc_, __VA_ARGS__)

#define DECLARE_SOA_TABLE_VERSIONED(_Name_, _Origin_, _Desc_, _Version_, ...) \
  O2HASH(#_Name_);                                                            \
  DECLARE_SOA_TABLE_METADATA(_Name_, _Desc_, _Version_, __VA_ARGS__);         \
  DECLARE_SOA_TABLE_FULL_VERSIONED_(_Name_, #_Name_, _Origin_, _Desc_, _Version_)

#define DECLARE_SOA_TABLE_STAGED_VERSIONED(_BaseName_, _Desc_, _Version_, ...) \
  O2HASH(_Desc_ "/" #_Version_);                                               \
  O2HASH(#_BaseName_);                                                         \
  O2HASH("Stored" #_BaseName_);                                                \
  DECLARE_SOA_TABLE_METADATA(_BaseName_, _Desc_, _Version_, __VA_ARGS__);      \
  using Stored##_BaseName_##Metadata = _BaseName_##Metadata;                   \
  DECLARE_SOA_TABLE_METADATA_TRAIT(_BaseName_, _Desc_, _Version_);             \
  DECLARE_SOA_STAGE(_BaseName_, "AOD", _Desc_, _Version_);                     \
  DECLARE_SOA_STAGE(Stored##_BaseName_, "AOD1", _Desc_, _Version_);

#define DECLARE_SOA_TABLE_STAGED(_BaseName_, _Desc_, ...) \
  DECLARE_SOA_TABLE_STAGED_VERSIONED(_BaseName_, _Desc_, 0, __VA_ARGS__);

#define DECLARE_SOA_EXTENDED_TABLE_NG(_Name_, _OriginalTable_, _Desc_, _Version_, ...)                                          \
  O2HASH(_Desc_ "/" #_Version_);                                                                                                \
  O2HASH(#_Name_ "Extension");                                                                                                  \
  template <typename O>                                                                                                         \
  using _Name_##ExtensionFrom = soa::Table<o2::aod::Hash<#_Name_ "Extension"_h>, o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>, O>; \
  using _Name_##Extension = _Name_##ExtensionFrom<o2::aod::Hash<"AOD"_h>>;                                                      \
  struct _Name_##ExtensionMetadata : TableMetadata<o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>, __VA_ARGS__> {                    \
    using base_table_t = _OriginalTable_;                                                                                       \
    template <o2::aod::is_origin_hash O>                                                                                        \
    using extension_table_t_from = _Name_##ExtensionFrom<O>;                                                                    \
    using extension_table_t = _Name_##Extension;                                                                                \
    using expression_pack_t = framework::pack<__VA_ARGS__>;                                                                     \
    static constexpr auto N = _OriginalTable_::originals.size();                                                                \
    template <o2::aod::is_origin_hash O = o2::aod::Hash<"AOD"_h>>                                                               \
    static consteval auto generateSources()                                                                                     \
    {                                                                                                                           \
      return _OriginalTable_##From<O>::originals;                                                                               \
    }                                                                                                                           \
  };                                                                                                                            \
  template <>                                                                                                                   \
  struct MetadataTrait<o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>> {                                                             \
    using metadata = _Name_##ExtensionMetadata;                                                                                 \
  };                                                                                                                            \
  template <typename O>                                                                                                         \
  using _Name_##From = o2::soa::Join<_OriginalTable_##From<O>, _Name_##ExtensionFrom<O>>;                                       \
  using _Name_ = _Name_##From<o2::aod::Hash<"AOD"_h>>;

#define DECLARE_SOA_EXTENDED_TABLE(_Name_, _Table_, _Description_, _Version_, ...) \
  DECLARE_SOA_EXTENDED_TABLE_NG(_Name_, _Table_, _Description_, _Version_, __VA_ARGS__)

#define DECLARE_SOA_EXTENDED_TABLE_USER(_Name_, _Table_, _Description_, ...) \
  DECLARE_SOA_EXTENDED_TABLE_NG(_Name_, _Table_, "EX" _Description_, 0, __VA_ARGS__)

#define DECLARE_SOA_CONFIGURABLE_EXTENDED_TABLE_NG(_Name_, _OriginalTable_, _Desc_, _Version_, ...)                                   \
  O2HASH(_Desc_ "/" #_Version_);                                                                                                      \
  O2HASH(#_Name_ "CfgExtension");                                                                                                     \
  template <typename O>                                                                                                               \
  using _Name_##CfgExtensionFrom = soa::Table<o2::aod::Hash<#_Name_ "CfgExtension"_h>, o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>, O>; \
  using _Name_##CfgExtension = _Name_##CfgExtensionFrom<o2::aod::Hash<"AOD"_h>>;                                                      \
  struct _Name_##CfgExtensionMetadata : TableMetadata<o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>, __VA_ARGS__> {                       \
    using base_table_t = _OriginalTable_;                                                                                             \
    template <o2::aod::is_origin_hash O>                                                                                              \
    using extension_table_t_from = _Name_##CfgExtensionFrom<O>;                                                                       \
    using extension_table_t = _Name_##CfgExtension;                                                                                   \
    using placeholders_pack_t = framework::pack<__VA_ARGS__>;                                                                         \
    using configurable_t = std::true_type;                                                                                            \
    static constexpr auto N = _OriginalTable_::originals.size();                                                                      \
    template <o2::aod::is_origin_hash O = o2::aod::Hash<"AOD"_h>>                                                                     \
    static consteval auto generateSources()                                                                                           \
    {                                                                                                                                 \
      return _OriginalTable_##From<O>::originals;                                                                                     \
    }                                                                                                                                 \
  };                                                                                                                                  \
  template <>                                                                                                                         \
  struct MetadataTrait<o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>> {                                                                   \
    using metadata = _Name_##CfgExtensionMetadata;                                                                                    \
  };                                                                                                                                  \
  template <typename O>                                                                                                               \
  using _Name_##From = o2::soa::Join<_OriginalTable_##From<O>, _Name_##CfgExtensionFrom<O>>;                                          \
  using _Name_ = _Name_##From<o2::aod::Hash<"AOD"_h>>;

#define DECLARE_SOA_CONFIGURABLE_EXTENDED_TABLE(_Name_, _OriginalTable_, _Description_, ...) \
  DECLARE_SOA_CONFIGURABLE_EXTENDED_TABLE_NG(_Name_, _OriginalTable_, "EX" _Description_, 0, __VA_ARGS__)

#define DECLARE_SOA_INDEX_TABLE_NG(_Name_, _Key_, _Version_, _Desc_, _Exclusive_, ...)                                                              \
  O2HASH(#_Name_);                                                                                                                                  \
  O2HASH(_Desc_ "/" #_Version_);                                                                                                                    \
  struct _Name_##Metadata : o2::aod::TableMetadata<o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>, soa::Index<>, __VA_ARGS__> {                          \
    static constexpr bool exclusive = _Exclusive_;                                                                                                  \
    template <o2::aod::is_origin_hash O>                                                                                                            \
    using KeyFrom = _Key_##From<O>;                                                                                                                 \
    using Key = _Key_;                                                                                                                              \
    using index_pack_t = framework::pack<__VA_ARGS__>;                                                                                              \
    template <o2::aod::is_origin_hash O = o2::aod::Hash<"AOD"_h>>                                                                                   \
    static consteval auto generateSources()                                                                                                         \
    {                                                                                                                                               \
      return []<soa::is_index_column... Cs>(framework::pack<Cs...>) {                                                                               \
        constexpr auto first = o2::soa::mergeOriginals<typename Cs::binding_t...>();                                                                \
        constexpr auto second = o2::aod::filterForKey<first.size(), first, Key>();                                                                  \
        return o2::aod::replaceOrigin<second.size(), second, O>();                                                                                  \
      }(framework::pack<__VA_ARGS__>{});                                                                                                            \
    }                                                                                                                                               \
    static constexpr auto N = []<typename... Cs>(framework::pack<Cs...>) {                                                                          \
      constexpr auto a = o2::soa::mergeOriginals<typename Cs::binding_t...>();                                                                      \
      return o2::aod::filterForKey<a.size(), a, Key>();                                                                                             \
    }(framework::pack<__VA_ARGS__>{})                                                                                                               \
                                .size();                                                                                                            \
  };                                                                                                                                                \
  template <>                                                                                                                                       \
  struct MetadataTrait<o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>> {                                                                                 \
    using metadata = _Name_##Metadata;                                                                                                              \
  };                                                                                                                                                \
  template <o2::aod::is_origin_hash O>                                                                                                              \
  using _Name_##From = o2::soa::IndexTable<o2::aod::Hash<#_Name_ ""_h>, o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>, O, _Key_##From<O>, __VA_ARGS__>; \
  using _Name_ = _Name_##From<o2::aod::Hash<"AOD"_h>>;

#define DECLARE_SOA_INDEX_TABLE(_Name_, _Key_, _Description_, ...) \
  DECLARE_SOA_INDEX_TABLE_NG(_Name_, _Key_, 0, _Description_, false, __VA_ARGS__)

#define DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(_Name_, _Key_, _Description_, ...) \
  DECLARE_SOA_INDEX_TABLE_NG(_Name_, _Key_, 0, _Description_, true, __VA_ARGS__)

#define DECLARE_SOA_INDEX_TABLE_USER(_Name_, _Key_, _Description_, ...) \
  DECLARE_SOA_INDEX_TABLE_NG(_Name_, _Key_, 0, _Description_, false, __VA_ARGS__)

#define DECLARE_SOA_INDEX_TABLE_EXCLUSIVE_USER(_Name_, _Key_, _Description_, ...) \
  DECLARE_SOA_INDEX_TABLE_NG(_Name_, _Key_, 0, _Description_, true, __VA_ARGS__)

// Declare were each row is associated to a timestamp column of an _TimestampSource_
// table.
//
// The columns of this table have to be CCDB_COLUMNS so that for each timestamp, we get a row
// which points to the specified CCDB objectes described by those columns.
#define DECLARE_SOA_TIMESTAMPED_TABLE_FULL(_Name_, _Label_, _TimestampSource_, _TimestampColumn_, _Version_, _Desc_, ...) \
  O2HASH(_Desc_ "/" #_Version_);                                                                                          \
  template <typename O>                                                                                                   \
  using _Name_##TimestampFrom = soa::Table<o2::aod::Hash<_Label_ ""_h>, o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>, O>;    \
  using _Name_##Timestamp = _Name_##TimestampFrom<o2::aod::Hash<                                                          \
    "AOD"                                                                                                                 \
    ""_h>>;                                                                                                               \
  struct _Name_##TimestampMetadata : TableMetadata<o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>, __VA_ARGS__> {              \
    template <typename O = o2::aod::Hash<"AOD"                                                                            \
                                         ""_h>>                                                                           \
    using base_table_t = _TimestampSource_##From<O>;                                                                      \
    template <typename O = o2::aod::Hash<"AOD"                                                                            \
                                         ""_h>>                                                                           \
    using extension_table_t = _Name_##TimestampFrom<O>;                                                                   \
    static constexpr const auto ccdb_urls = []<typename... Cs>(framework::pack<Cs...>) {                                  \
      return std::array<std::string_view, sizeof...(Cs)>{Cs::query...};                                                   \
    }(framework::pack<__VA_ARGS__>{});                                                                                    \
    static constexpr const auto ccdb_bindings = []<typename... Cs>(framework::pack<Cs...>) {                              \
      return std::array<std::string_view, sizeof...(Cs)>{Cs::mLabel...};                                                  \
    }(framework::pack<__VA_ARGS__>{});                                                                                    \
    static constexpr auto N = _TimestampSource_::originals.size();                                                        \
    template <o2::aod::is_origin_hash O = o2::aod::Hash<"AOD"_h>>                                                         \
    static consteval auto generateSources()                                                                               \
    {                                                                                                                     \
      return _TimestampSource_##From<O>::originals;                                                                       \
    }                                                                                                                     \
    static constexpr auto timestamp_column_label = _TimestampColumn_::mLabel;                                             \
    /*static constexpr auto timestampColumn = _TimestampColumn_;*/                                                        \
  };                                                                                                                      \
  template <>                                                                                                             \
  struct MetadataTrait<o2::aod::Hash<_Desc_ "/" #_Version_ ""_h>> {                                                       \
    using metadata = _Name_##TimestampMetadata;                                                                           \
  };                                                                                                                      \
  template <typename O>                                                                                                   \
  using _Name_##From = o2::soa::Join<_TimestampSource_, _Name_##TimestampFrom<O>>;                                        \
  using _Name_ = _Name_##From<o2::aod::Hash<                                                                              \
    "AOD"                                                                                                                 \
    ""_h>>;

#define DECLARE_SOA_TIMESTAMPED_TABLE(_Name_, _TimestampSource_, _TimestampColumn_, _Version_, _Desc_, ...) \
  O2HASH(#_Name_ "Timestamped");                                                                            \
  DECLARE_SOA_TIMESTAMPED_TABLE_FULL(_Name_, #_Name_ "Timestamped", _TimestampSource_, _TimestampColumn_, _Version_, _Desc_, __VA_ARGS__)

namespace o2::soa
{
template <typename... Ts>
struct Join : Table<o2::aod::Hash<"JOIN"_h>, o2::aod::Hash<"JOIN/0"_h>, o2::aod::Hash<"JOIN"_h>, Ts...> {
  using base = Table<o2::aod::Hash<"JOIN"_h>, o2::aod::Hash<"JOIN/0"_h>, o2::aod::Hash<"JOIN"_h>, Ts...>;

  Join(std::shared_ptr<arrow::Table>&& table, uint64_t offset = 0)
    : base{std::move(table), offset}
  {
    if (this->tableSize() != 0) {
      bindInternalIndicesTo(this);
    }
  }
  Join(std::vector<std::shared_ptr<arrow::Table>>&& tables, uint64_t offset = 0)
    : base{ArrowHelpers::joinTables(std::move(tables), std::span{base::originalLabels}), offset}
  {
    if (this->tableSize() != 0) {
      bindInternalIndicesTo(this);
    }
  }
  using base::bindExternalIndices;
  using base::bindInternalIndicesTo;
  static constexpr const uint32_t binding_origin = base::binding_origin;
  static constexpr const header::DataOrigin binding_origin_ = base::binding_origin_;

  template <typename... TA>
  void bindExternalIndices(TA*... current)
  {
    ([this](TA* cur) {
      if constexpr (binding_origin == TA::binding_origin) {
        this->bindExternalIndex(cur);
      }
    }(current),
     ...);
  }

  using self_t = Join<Ts...>;
  using table_t = base;
  static constexpr const auto originals = base::originals;
  static constexpr const auto originalLabels = base::originalLabels;
  using columns_t = typename table_t::columns_t;
  using persistent_columns_t = typename table_t::persistent_columns_t;
  using iterator = table_t::template iterator_template<DefaultIndexPolicy, self_t, Ts...>;
  using const_iterator = iterator;
  using unfiltered_iterator = iterator;
  using unfiltered_const_iterator = const_iterator;
  using filtered_iterator = table_t::template iterator_template<FilteredIndexPolicy, self_t, Ts...>;
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

  template <typename T1, typename Policy, bool OPT>
  auto sliceBy(o2::framework::PresliceBase<T1, Policy, OPT> const& container, int value) const
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
    return []<size_t... Is>(std::index_sequence<Is...>) {
      return (std::ranges::any_of(originals, [](TableRef const& ref) { return ref.desc_hash == T::originals[Is].desc_hash; }) && ...);
    }(std::make_index_sequence<T::originals.size()>());
  }
};

template <typename... Ts>
constexpr auto join(Ts const&... t)
{
  return Join<Ts...>(ArrowHelpers::joinTables({t.asArrowTable()...}, std::span{Join<Ts...>::base::originalLabels}));
}

template <typename T>
concept is_join = framework::specialization_of_template<Join, T>;

template <typename T>
constexpr bool is_soa_join_v = is_join<T>;

template <typename... Ts>
struct Concat : Table<o2::aod::Hash<"CONC"_h>, o2::aod::Hash<"CONC/0"_h>, o2::aod::Hash<"CONC"_h>, Ts...> {
  using base = Table<o2::aod::Hash<"CONC"_h>, o2::aod::Hash<"CONC/0"_h>, o2::aod::Hash<"CONC"_h>, Ts...>;
  using self_t = Concat<Ts...>;
  Concat(std::vector<std::shared_ptr<arrow::Table>>&& tables, uint64_t offset = 0)
    : base{ArrowHelpers::concatTables(std::move(tables)), offset}
  {
    bindInternalIndicesTo(this);
  }
  Concat(Ts const&... t, uint64_t offset = 0)
    : base{ArrowHelpers::concatTables({t.asArrowTable()...}), offset}
  {
    bindInternalIndicesTo(this);
  }

  using base::originals;

  using base::bindExternalIndices;
  using base::bindInternalIndicesTo;

  using table_t = base;
  using columns_t = typename table_t::columns_t;
  using persistent_columns_t = typename table_t::persistent_columns_t;

  using iterator = table_t::template iterator_template<DefaultIndexPolicy, self_t, Ts...>;
  using const_iterator = iterator;
  using unfiltered_iterator = iterator;
  using unfiltered_const_iterator = const_iterator;
  using filtered_iterator = table_t::template iterator_template<FilteredIndexPolicy, self_t, Ts...>;
  using filtered_const_iterator = filtered_iterator;
};

template <typename... Ts>
constexpr auto concat(Ts const&... t)
{
  return Concat<Ts...>{t...};
}

template <soa::is_table T>
class FilteredBase : public T
{
 public:
  using self_t = FilteredBase<T>;
  using table_t = typename T::table_t;
  using T::originals;
  static constexpr const uint32_t binding_origin = T::binding_origin;
  static constexpr const header::DataOrigin binding_origin_ = T::binding_origin_;
  template <typename... TA>
  void bindExternalIndices(TA*... current)
  {
    ([this](TA* cur) {
      if constexpr (binding_origin == TA::binding_origin) {
        this->bindExternalIndex(cur);
        mFilteredBegin.bindExternalIndex(cur);
      }
    }(current),
     ...);
  }
  using columns_t = typename T::columns_t;
  using persistent_columns_t = typename T::persistent_columns_t;
  using external_index_columns_t = typename T::external_index_columns_t;

  using iterator = T::template iterator_template_o<FilteredIndexPolicy, self_t>;
  using unfiltered_iterator = T::template iterator_template_o<DefaultIndexPolicy, self_t>;
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
    mSelectedRows = std::span{mSelectedRowsCache};
    if (this->tableSize() != 0) {
      mFilteredBegin = table_t::filtered_begin(mSelectedRows);
    }
    resetRanges();
    mFilteredBegin.bindInternalIndices(this);
  }

  FilteredBase(std::vector<std::shared_ptr<arrow::Table>>&& tables, std::span<int64_t const> const& selection, uint64_t offset = 0)
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
      return std::span<int64_t const>{};
    }
    auto array = std::static_pointer_cast<arrow::Int64Array>(sel->ToArray());
    auto start = array->raw_values();
    auto stop = start + array->length();
    return std::span{start, stop};
  }

  /// Bind the columns which refer to other tables
  /// to the associated tables.
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

  template <typename T1, bool OPT>
  auto sliceBy(o2::framework::PresliceBase<T1, framework::PreslicePolicySorted, OPT> const& container, int value) const
  {
    return doFilteredSliceBy(this, container, value);
  }

  template <typename T1, bool OPT>
  auto sliceBy(o2::framework::PresliceBase<T1, framework::PreslicePolicyGeneral, OPT> const& container, int value) const
  {
    return doSliceBy(this, container, value);
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

  void sumWithSelection(std::span<int64_t const> const& selection)
  {
    mCached = true;
    SelectionVector rowsUnion;
    std::set_union(mSelectedRows.begin(), mSelectedRows.end(), selection.begin(), selection.end(), std::back_inserter(rowsUnion));
    mSelectedRowsCache.clear();
    mSelectedRowsCache = rowsUnion;
    resetRanges();
  }

  void intersectWithSelection(std::span<int64_t const> const& selection)
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
      mSelectedRows = std::span{mSelectedRowsCache};
    }
    mFilteredEnd.reset(new RowViewSentinel{static_cast<int64_t>(mSelectedRows.size())});
    if (tableSize() == 0) {
      mFilteredBegin = *mFilteredEnd;
    } else {
      mFilteredBegin.resetSelection(mSelectedRows);
    }
  }

  std::span<int64_t const> mSelectedRows;
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
  using table_t = typename T::table_t;
  using columns_t = typename T::columns_t;

  using iterator = T::template iterator_template_o<FilteredIndexPolicy, self_t>;
  using unfiltered_iterator = T::template iterator_template_o<DefaultIndexPolicy, self_t>;
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

  Filtered(std::vector<std::shared_ptr<arrow::Table>>&& tables, std::span<int64_t const> const& selection, uint64_t offset = 0)
    : FilteredBase<T>(std::move(tables), selection, offset) {}

  Filtered<T> operator+(SelectionVector const& selection)
  {
    Filtered<T> copy(*this);
    copy.sumWithSelection(selection);
    return copy;
  }

  Filtered<T> operator+(std::span<int64_t const> const& selection)
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

  Filtered<T> operator+=(std::span<int64_t const> const& selection)
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

  Filtered<T> operator*(std::span<int64_t const> const& selection)
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

  Filtered<T> operator*=(std::span<int64_t const> const& selection)
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

  template <typename T1, bool OPT>
  auto sliceBy(o2::framework::PresliceBase<T1, framework::PreslicePolicySorted, OPT> const& container, int value) const
  {
    return doFilteredSliceBy(this, container, value);
  }

  template <typename T1, bool OPT>
  auto sliceBy(o2::framework::PresliceBase<T1, framework::PreslicePolicyGeneral, OPT> const& container, int value) const
  {
    return doSliceBy(this, container, value);
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
  using columns_t = typename T::columns_t;

  using iterator = typename T::template iterator_template_o<FilteredIndexPolicy, self_t>;
  using unfiltered_iterator = typename T::template iterator_template_o<DefaultIndexPolicy, self_t>;
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

  Filtered(std::vector<Filtered<T>>&& tables, std::span<int64_t const> const& selection, uint64_t offset = 0)
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

  Filtered<Filtered<T>> operator+(std::span<int64_t const> const& selection)
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

  Filtered<Filtered<T>> operator+=(std::span<int64_t const> const& selection)
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

  Filtered<Filtered<T>> operator*(std::span<int64_t const> const& selection)
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

  Filtered<Filtered<T>> operator*=(std::span<int64_t const> const& selection)
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

  template <typename T1, bool OPT>
  auto sliceBy(o2::framework::PresliceBase<T1, framework::PreslicePolicySorted, OPT> const& container, int value) const
  {
    return doFilteredSliceBy(this, container, value);
  }

  template <typename T1, bool OPT>
  auto sliceBy(o2::framework::PresliceBase<T1, framework::PreslicePolicyGeneral, OPT> const& container, int value) const
  {
    return doSliceBy(this, container, value);
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
template <typename L, typename D, typename O, typename Key, typename H, typename... Ts>
struct IndexTable : Table<L, D, O> {
  using self_t = IndexTable<L, D, O, Key, H, Ts...>;
  using base_t = Table<L, D, O>;
  using table_t = base_t;
  using safe_base_t = Table<L, D, O>;
  using indexing_t = Key;
  using first_t = typename H::binding_t;
  using rest_t = framework::pack<typename Ts::binding_t...>;

  static constexpr const uint32_t binding_origin = Key::binding_origin;
  static constexpr const header::DataOrigin binding_origin_ = Key::binding_origin_;

  template <typename... TA>
  void bindExternalIndices(TA*... current)
  {
    ([this](TA* cur) {
      if constexpr (binding_origin == TA::binding_origin) {
        this->bindExternalIndex(cur);
      }
    }(current),
     ...);
  }

  IndexTable(std::shared_ptr<arrow::Table> table, uint64_t offset = 0)
    : base_t{table, offset}
  {
  }

  IndexTable(std::vector<std::shared_ptr<arrow::Table>> tables, uint64_t offset = 0)
    : base_t{tables[0], offset}
  {
  }

  IndexTable(IndexTable const&) = default;
  IndexTable(IndexTable&&) = default;
  IndexTable& operator=(IndexTable const&) = default;
  IndexTable& operator=(IndexTable&&) = default;

  using iterator = typename base_t::template iterator_template_o<DefaultIndexPolicy, self_t>;
  using const_iterator = iterator;
  using filtered_iterator = typename base_t::template iterator_template_o<FilteredIndexPolicy, self_t>;
  using const_filtered_iterator = filtered_iterator;
};

template <typename T, bool APPLY>
struct SmallGroupsBase : public Filtered<T> {
  static constexpr bool applyFilters = APPLY;
  SmallGroupsBase(std::vector<std::shared_ptr<arrow::Table>>&& tables, gandiva::Selection const& selection, uint64_t offset = 0)
    : Filtered<T>(std::move(tables), selection, offset) {}

  SmallGroupsBase(std::vector<std::shared_ptr<arrow::Table>>&& tables, SelectionVector&& selection, uint64_t offset = 0)
    : Filtered<T>(std::move(tables), std::forward<SelectionVector>(selection), offset) {}

  SmallGroupsBase(std::vector<std::shared_ptr<arrow::Table>>&& tables, std::span<int64_t const> const& selection, uint64_t offset = 0)
    : Filtered<T>(std::move(tables), selection, offset) {}
};

template <typename T>
using SmallGroups = SmallGroupsBase<T, true>;

template <typename T>
using SmallGroupsUnfiltered = SmallGroupsBase<T, false>;

template <typename T>
concept is_smallgroups = requires {
  []<typename B, bool A>(SmallGroupsBase<B, A>*) {}(std::declval<std::decay_t<T>*>());
};
} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOA_H_
