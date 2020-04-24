// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_ASOAHELPERS_H_
#define O2_FRAMEWORK_ASOAHELPERS_H_

#include "Framework/ASoA.h"
#include "Framework/Kernels.h"
#include "Framework/ArrowCompatibility.h"
#include "../src/ExpressionHelpers.h"
#include "Framework/Expressions.h"
#include "Framework/VariantHelpers.h"

#include <arrow/table.h>
#include <arrow/compute/context.h>

#include <iterator>
#include <tuple>
#include <utility>

namespace o2::soa
{

// Functions to enable looping over tuples
template <std::size_t V>
struct Num {
  static const constexpr auto value = V;
};

template <class F, std::size_t... Is>
void for_(F func, std::index_sequence<Is...>)
{
  using expander = int[];
  (void)expander{0, ((void)func(Num<Is>{}), 0)...};
}

template <std::size_t N, typename F>
void for_(F func)
{
  for_(func, std::make_index_sequence<N>());
}

// Creating tuple of given size and type
template <typename T, unsigned N, typename... REST>
struct NTupleType {
  using type = typename NTupleType<T, N - 1, T, REST...>::type;
};

template <typename T, typename... REST>
struct NTupleType<T, 0, REST...> {
  using type = std::tuple<REST...>;
};

// Set constant value for a filter spec from table
template <typename T2, typename ARRAY>
void doSetColumnValue(const std::shared_ptr<arrow::Table>& table, const std::string& columnName, o2::framework::expressions::DatumSpec& datumSpec, uint64_t index)
{
  auto columnIndex = table->schema()->GetFieldIndex(columnName);
  auto dataType = table->column(columnIndex)->type();
  auto chunkedArray = table->column(columnIndex)->data();
  auto chunk = chunkedArray->chunk(0);
  T2 value;

  if (chunk->length() == 0) {
    value = T2();
  }

  uint64_t ind = 0;
  for (uint64_t ci = 0; ci < chunkedArray->num_chunks(); ++ci) {
    auto chunk = chunkedArray->chunk(ci);
    if (index < ind + chunk->length()) {
      value = std::static_pointer_cast<ARRAY>(chunk)->Value(index - ind);
      break;
    }
    ind += chunk->length();
  }
  datumSpec.datum = (o2::framework::expressions::LiteralNode::var_t)value;
  datumSpec.type = dataType->id();
}

template <typename T2>
void setColumnValue(const T2& table, const std::string& columnName, o2::framework::expressions::DatumSpec& datumSpec, uint64_t index)
{
  auto arrowTable = table.asArrowTable();
  auto columnIndex = arrowTable->schema()->GetFieldIndex(columnName);
  auto dataType = arrowTable->column(columnIndex)->type();
  if (dataType->id() == arrow::Type::INT32) {
    doSetColumnValue<int32_t, arrow::Int32Array>(arrowTable, columnName, datumSpec, index);
  } else if (dataType->id() == arrow::Type::FLOAT) {
    doSetColumnValue<float, arrow::FloatArray>(arrowTable, columnName, datumSpec, index);
  } else if (dataType->id() == arrow::Type::DOUBLE) {
    doSetColumnValue<double, arrow::DoubleArray>(arrowTable, columnName, datumSpec, index);
  } else if (dataType->id() == arrow::Type::BOOL) {
    doSetColumnValue<bool, arrow::BooleanArray>(arrowTable, columnName, datumSpec, index);
  } else
    throw std::runtime_error("Combinations: Invalid data type in array!");
}

// Group table (C++ vector of indices)
bool sameCategory(std::pair<uint64_t, uint64_t> const& a, std::pair<uint64_t, uint64_t> const& b)
{
  return a.first < b.first;
}
bool diffCategory(std::pair<uint64_t, uint64_t> const& a, std::pair<uint64_t, uint64_t> const& b)
{
  return a.first >= b.first;
}

template <typename T2, typename ARRAY>
std::vector<std::pair<uint64_t, uint64_t>> doGroupTable(const std::shared_ptr<arrow::Table>& table, const std::string& categoryColumnName, int minCatSize)
{
  auto columnIndex = table->schema()->GetFieldIndex(categoryColumnName);
  auto chunkedArray = framework::getBackendColumnData(table->column(columnIndex));

  uint64_t ind = 0;
  std::vector<std::pair<uint64_t, uint64_t>> groupedIndices;
  groupedIndices.reserve(table->num_rows());
  for (uint64_t ci = 0; ci < chunkedArray->num_chunks(); ++ci) {
    auto chunk = chunkedArray->chunk(ci);
    // Note: assuming that the table is not empty
    T2 const* data = std::static_pointer_cast<ARRAY>(chunk)->raw_values();
    for (uint64_t ai = 0; ai < chunk->length(); ++ai) {
      groupedIndices.emplace_back(data[ai], ind);
      ind++;
    }
  }

  // Do a stable sort so that same categories entries are
  // grouped together.
  std::stable_sort(groupedIndices.begin(), groupedIndices.end());

  // Remove categories of too small size
  if (minCatSize > 1) {
    auto catBegin = groupedIndices.begin();
    while (catBegin != groupedIndices.end()) {
      auto catEnd = std::upper_bound(catBegin, groupedIndices.end(), *catBegin, sameCategory);
      if (std::distance(catBegin, catEnd) < minCatSize) {
        catEnd = groupedIndices.erase(catBegin, catEnd);
      }
      catBegin = catEnd;
    }
  }

  return groupedIndices;
}

template <typename T>
auto groupTable(const T& table, const std::string& categoryColumnName, int minCatSize)
{
  auto arrowTable = table.asArrowTable();
  auto columnIndex = arrowTable->schema()->GetFieldIndex(categoryColumnName);
  auto dataType = arrowTable->column(columnIndex)->type();
  if (dataType->id() == arrow::Type::UINT64) {
    return doGroupTable<uint64_t, arrow::UInt64Array>(arrowTable, categoryColumnName, minCatSize);
  }
  if (dataType->id() == arrow::Type::INT64) {
    return doGroupTable<int64_t, arrow::Int64Array>(arrowTable, categoryColumnName, minCatSize);
  }
  if (dataType->id() == arrow::Type::UINT32) {
    return doGroupTable<uint32_t, arrow::UInt32Array>(arrowTable, categoryColumnName, minCatSize);
  }
  if (dataType->id() == arrow::Type::INT32) {
    return doGroupTable<int32_t, arrow::Int32Array>(arrowTable, categoryColumnName, minCatSize);
  }
  if (dataType->id() == arrow::Type::FLOAT) {
    return doGroupTable<float, arrow::FloatArray>(arrowTable, categoryColumnName, minCatSize);
  }
  // FIXME: Should we support other types as well?
  throw std::runtime_error("Combinations: category column must be of integral type");
}

// Synchronize categories so as groupedIndices contain elements only of categories common to all tables
template <std::size_t K>
void syncCategories(std::array<std::vector<std::pair<uint64_t, uint64_t>>, K>& groupedIndices)
{
  std::vector<std::pair<uint64_t, uint64_t>> firstCategories;
  std::vector<std::pair<uint64_t, uint64_t>> commonCategories;
  std::unique_copy(groupedIndices[0].begin(), groupedIndices[0].end(), std::back_inserter(firstCategories), diffCategory);

  for (auto& cat : firstCategories) {
    bool isCommon = true;
    for (int i = 1; i < K; i++) {
      if (!std::binary_search(groupedIndices[i].begin(), groupedIndices[i].end(), cat, sameCategory)) {
        isCommon = false;
        break;
      }
    }
    if (isCommon) {
      commonCategories.push_back(cat);
    }
  }

  for (int i = 0; i < K; i++) {
    auto nonCatBegin = groupedIndices[i].begin();
    for (auto& cat : commonCategories) {
      auto nonCatEnd = std::lower_bound(nonCatBegin, groupedIndices[i].end(), cat, sameCategory);
      nonCatEnd = groupedIndices[i].erase(nonCatBegin, nonCatEnd);
      nonCatBegin = std::upper_bound(nonCatEnd, groupedIndices[i].end(), cat, sameCategory);
    }
  }
}

template <typename... Ts>
struct CombinationsIndexPolicyBase {
  using CombinationType = std::tuple<typename Ts::iterator...>;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts)>::type;

  CombinationsIndexPolicyBase(const Ts&... tables) : mIsEnd(false),
                                                     mMaxOffset(tables.end().index...),
                                                     mCurrent(tables.begin()...)
  {
    if (((tables.size() == 0) || ...)) {
      mIsEnd = true;
    }
  }

  void moveToEnd() {}
  void addOne() {}

  CombinationType mCurrent;
  IndicesType mMaxOffset; // one position past maximum acceptable position for each element of combination
  bool mIsEnd;            // whether there are any more tuples available
};

template <typename... Ts>
struct CombinationsUpperIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;

  CombinationsUpperIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...) {}

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).moveToEnd();
    });
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrent)++;
        if (*std::get<1>(std::get<curInd>(this->mCurrent).getIndices()) != std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrent).setCursor(*std::get<1>(std::get<curJ - 1>(this->mCurrent).getIndices()));
          });
          modify = false;
        }
      }
    });
    this->mIsEnd = modify;
  }
};

template <typename... Ts>
struct CombinationsStrictlyUpperIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;

  CombinationsStrictlyUpperIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...)
  {
    constexpr auto k = sizeof...(Ts);
    if (((tables.size() < k) || ...)) {
      this->mIsEnd = true;
      return;
    }
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mMaxOffset) += i.value + 1 - k;
      std::get<i.value>(this->mCurrent).moveByIndex(i.value);
    });
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k - 1>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).setCursor(std::get<i.value>(this->mMaxOffset));
    });
    // Hack for Filtered
    std::get<k - 1>(this->mCurrent).moveToEnd();
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrent)++;
        if (*std::get<1>(std::get<curInd>(this->mCurrent).getIndices()) != std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrent).setCursor(*std::get<1>(std::get<curJ - 1>(this->mCurrent).getIndices()) + 1);
          });
          modify = false;
        }
      }
    });
    this->mIsEnd = modify;
  }
};

template <typename... Ts>
struct CombinationsFullIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;

  CombinationsFullIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...) {}

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).moveToEnd();
    });
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrent)++;
        if (*std::get<1>(std::get<curInd>(this->mCurrent).getIndices()) != std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrent).setCursor(0);
          });
          modify = false;
        }
      }
    });
    this->mIsEnd = modify;
  }
};

template <typename... Ts>
struct CombinationsBlockUpperIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts)>::type;

  CombinationsBlockUpperIndexPolicy(const std::string& categoryColumnName, const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...)
  {
    constexpr auto k = sizeof...(Ts);
    if (this->mIsEnd) {
      return;
    }

    int tableIndex = 0;
    ((this->mGroupedIndices[tableIndex++] = groupTable(tables, categoryColumnName, 1)), ...);

    // Synchronize categories across tables
    syncCategories(this->mGroupedIndices);

    for_<k>([this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = 0;
    });

    setRanges();
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      auto catBegin = this->mGroupedIndices[i.value].begin() + std::get<i.value>(this->mCurrentIndices);
      auto range = std::equal_range(catBegin, this->mGroupedIndices[i.value].end(), *catBegin, sameCategory);
      std::get<i.value>(this->mMaxOffset) = std::distance(this->mGroupedIndices[i.value].begin(), range.second);
      std::get<i.value>(this->mCurrent).setCursor(range.first->second);
    });
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).moveToEnd();
    });
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    bool nextCatAvailable = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curInd][curGroupedInd].second);

        // If we remain within the same category
        if (std::get<curInd>(this->mCurrentIndices) < std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices);
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].second);
          });
          modify = false;
        }
      }
    });

    // No more combinations within this category - move to the next category, if possible
    if (modify) {
      for_<k>([&, this](auto m) {
        if (std::get<m.value>(this->mCurrentIndices) == this->mGroupedIndices[m.value].size()) {
          nextCatAvailable = false;
        }
      });
      if (nextCatAvailable) {
        setRanges();
      }
    }

    this->mIsEnd = modify && !nextCatAvailable;
  }

  std::array<std::vector<std::pair<uint64_t, uint64_t>>, sizeof...(Ts)> mGroupedIndices;
  IndicesType mCurrentIndices;
};

template <typename... Ts>
struct CombinationsBlockStrictlyUpperIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts)>::type;

  CombinationsBlockStrictlyUpperIndexPolicy(const std::string& categoryColumnName, const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...)
  {
    constexpr auto k = sizeof...(Ts);
    if (((tables.size() < k) || ...)) {
      this->mIsEnd = true;
      return;
    }

    int tableIndex = 0;
    ((this->mGroupedIndices[tableIndex++] = groupTable(tables, categoryColumnName, k)), ...);

    // Synchronize categories across tables
    syncCategories(this->mGroupedIndices);

    for (int i = 0; i < k; i++) {
      if (this->mGroupedIndices[i].size() == 0) {
        this->mIsEnd = true;
        return;
      }
    }

    for_<k>([this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = i.value;
    });

    setRanges();
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      auto catBegin = this->mGroupedIndices[i.value].begin() + std::get<i.value>(this->mCurrentIndices);
      auto range = std::equal_range(catBegin, this->mGroupedIndices[i.value].end(), *catBegin, sameCategory);
      std::get<i.value>(this->mCurrent).setCursor(range.first->second);
      std::get<i.value>(this->mMaxOffset) = std::distance(this->mGroupedIndices[i.value].begin(), range.second - k + i.value + 1);
    });
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k - 1>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).setCursor(std::get<i.value>(this->mMaxOffset));
    });
    // Hack for Filtered
    std::get<k - 1>(this->mCurrent).moveToEnd();
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    bool nextCatAvailable = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curInd][curGroupedInd].second);

        // If we remain within the same category
        if (std::get<curInd>(this->mCurrentIndices) < std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices) + 1;
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].second);
          });
          modify = false;
        }
      }
    });

    // No more combinations within this category - move to the next category, if possible
    if (modify) {
      for_<k>([&, this](auto m) {
        if (std::get<m.value>(this->mCurrentIndices) == this->mGroupedIndices[m.value].size()) {
          nextCatAvailable = false;
        } else {
          std::get<m.value>(this->mCurrentIndices) = std::get<m.value>(this->mMaxOffset) + k - 1;
        }
      });
      if (nextCatAvailable) {
        setRanges();
      }
    }

    this->mIsEnd = modify && !nextCatAvailable;
  }

  std::array<std::vector<std::pair<uint64_t, uint64_t>>, sizeof...(Ts)> mGroupedIndices;
  IndicesType mCurrentIndices;
};

template <typename... Ts>
struct CombinationsBlockFullIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts)>::type;

  CombinationsBlockFullIndexPolicy(const std::string& categoryColumnName, const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...)
  {
    if (this->mIsEnd) {
      return;
    }

    constexpr auto k = sizeof...(Ts);

    int tableIndex = 0;
    ((this->mGroupedIndices[tableIndex++] = groupTable(tables, categoryColumnName, 1)), ...);

    // Synchronize categories across tables
    syncCategories(this->mGroupedIndices);

    for_<k>([this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = 0;
    });

    setRanges();
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      auto catBegin = this->mGroupedIndices[i.value].begin() + std::get<i.value>(this->mCurrentIndices);
      auto range = std::equal_range(catBegin, this->mGroupedIndices[i.value].end(), *catBegin, sameCategory);
      std::get<i.value>(this->mBeginIndices) = std::distance(this->mGroupedIndices[i.value].begin(), range.first);
      std::get<i.value>(this->mMaxOffset) = std::distance(this->mGroupedIndices[i.value].begin(), range.second);
      std::get<i.value>(this->mCurrent).setCursor(range.first->second);
    });
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).moveToEnd();
    });
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    bool nextCatAvailable = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curInd][curGroupedInd].second);

        // If we remain within the same category
        if (std::get<curInd>(this->mCurrentIndices) < std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrentIndices) = std::get<curJ>(this->mBeginIndices);
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].second);
          });
          modify = false;
        }
      }
    });

    // No more combinations within this category - move to the next category, if possible
    if (modify) {
      for_<k>([&, this](auto m) {
        if (std::get<m.value>(this->mCurrentIndices) == this->mGroupedIndices[m.value].size()) {
          nextCatAvailable = false;
        }
      });
      if (nextCatAvailable) {
        setRanges();
      }
    }

    this->mIsEnd = modify && !nextCatAvailable;
  }

  std::array<std::vector<std::pair<uint64_t, uint64_t>>, sizeof...(Ts)> mGroupedIndices;
  IndicesType mCurrentIndices;
  IndicesType mBeginIndices;
};

template <typename T, typename... Ts>
struct CombinationsBlockUpperSameIndexPolicy : public CombinationsIndexPolicyBase<T, Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<T, Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts) + 1>::type;

  CombinationsBlockUpperSameIndexPolicy(const std::string& categoryColumnName, const T& table, const Ts&... tables) : CombinationsIndexPolicyBase<T, Ts...>(table, tables...)
  {
    constexpr auto k = sizeof...(Ts) + 1;
    if (this->mIsEnd) {
      return;
    }

    this->mGroupedIndices = groupTable(table, categoryColumnName, 1);

    for_<k>([this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = 0;
    });

    setRanges();
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    auto catBegin = this->mGroupedIndices.begin() + std::get<0>(this->mCurrentIndices);
    auto range = std::equal_range(catBegin, this->mGroupedIndices.end(), *catBegin, sameCategory);
    uint64_t offset = std::distance(this->mGroupedIndices.begin(), range.second);

    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mMaxOffset) = offset;
      std::get<i.value>(this->mCurrent).setCursor(range.first->second);
    });
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).moveToEnd();
    });
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].second);

        // If we remain within the same category
        if (std::get<curInd>(this->mCurrentIndices) < std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices);
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].second);
          });
          modify = false;
        }
      }
    });

    // No more combinations within this category - move to the next category, if possible
    if (modify && std::get<0>(this->mCurrentIndices) < this->mGroupedIndices.size()) {
      setRanges();
      return;
    }

    this->mIsEnd = modify;
  }

  std::vector<std::pair<uint64_t, uint64_t>> mGroupedIndices;
  IndicesType mCurrentIndices;
};

template <typename T, typename... Ts>
struct CombinationsBlockStrictlyUpperSameIndexPolicy : public CombinationsIndexPolicyBase<T, Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<T, Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts) + 1>::type;

  CombinationsBlockStrictlyUpperSameIndexPolicy(const std::string& categoryColumnName, const T& table, const Ts&... tables) : CombinationsIndexPolicyBase<T, Ts...>(table, tables...)
  {
    constexpr auto k = sizeof...(Ts) + 1;
    if (table.size() < k) {
      this->mIsEnd = true;
      return;
    }

    this->mGroupedIndices = groupTable(table, categoryColumnName, k);

    if (this->mGroupedIndices.size() == 0) {
      this->mIsEnd = true;
      return;
    }

    for_<k>([this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = i.value;
    });

    setRanges();
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    auto catBegin = this->mGroupedIndices.begin() + std::get<0>(this->mCurrentIndices);
    auto lastIt = std::upper_bound(catBegin, this->mGroupedIndices.end(), *catBegin, sameCategory);
    uint64_t lastOffset = std::distance(this->mGroupedIndices.begin(), lastIt);

    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).setCursor(this->mGroupedIndices[std::get<i.value>(this->mCurrentIndices)].second);
      std::get<i.value>(this->mMaxOffset) = lastOffset - k + i.value + 1;
    });
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    for_<k - 1>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).setCursor(std::get<i.value>(this->mMaxOffset));
    });
    // Hack for Filtered
    std::get<k - 1>(this->mCurrent).moveToEnd();
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].second);

        // If we remain within the same category
        if (std::get<curInd>(this->mCurrentIndices) < std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices) + 1;
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].second);
          });
          modify = false;
        }
      }
    });

    // No more combinations within this category - move to the next category, if possible
    if (modify && std::get<k - 1>(this->mCurrentIndices) < this->mGroupedIndices.size()) {
      for_<k>([&, this](auto m) {
        std::get<m.value>(this->mCurrentIndices) = std::get<m.value>(this->mMaxOffset) + k - 1;
      });
      setRanges();
      return;
    }

    this->mIsEnd = modify;
  }

  std::vector<std::pair<uint64_t, uint64_t>> mGroupedIndices;
  IndicesType mCurrentIndices;
};

template <typename T, typename... Ts>
struct CombinationsBlockFullSameIndexPolicy : public CombinationsIndexPolicyBase<T, Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<T, Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts) + 1>::type;

  CombinationsBlockFullSameIndexPolicy(const std::string& categoryColumnName, const T& table, const Ts&... tables) : CombinationsIndexPolicyBase<T, Ts...>(table, tables...)
  {
    if (this->mIsEnd) {
      return;
    }

    constexpr auto k = sizeof...(Ts) + 1;

    int tableIndex = 0;
    this->mGroupedIndices = groupTable(table, categoryColumnName, 1);
    for_<k>([this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = 0;
    });

    setRanges();
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    auto catBegin = this->mGroupedIndices.begin() + std::get<0>(this->mCurrentIndices);
    auto range = std::equal_range(catBegin, this->mGroupedIndices.end(), *catBegin, sameCategory);
    this->mBeginIndex = std::distance(this->mGroupedIndices.begin(), range.first);
    uint64_t offset = std::distance(this->mGroupedIndices.begin(), range.second);

    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mMaxOffset) = offset;
      std::get<i.value>(this->mCurrent).setCursor(range.first->second);
    });
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).moveToEnd();
    });
    this->mIsEnd = true;
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].second);

        // If we remain within the same category
        if (std::get<curInd>(this->mCurrentIndices) < std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrentIndices) = this->mBeginIndex;
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].second);
          });
          modify = false;
        }
      }
    });

    // No more combinations within this category - move to the next category, if possible
    if (modify && std::get<0>(this->mCurrentIndices) < this->mGroupedIndices.size()) {
      setRanges();
      return;
    }

    this->mIsEnd = modify;
  }

  std::vector<std::pair<uint64_t, uint64_t>> mGroupedIndices;
  IndicesType mCurrentIndices;
  uint64_t mBeginIndex;
};

template <typename T, typename... Ts>
struct CombinationsFilteredIndexPolicyBase {
  using CombinationType = std::tuple<typename Filtered<T>::iterator, typename Ts::iterator...>;
  using IndicesType = typename NTupleType<uint64_t, (sizeof...(Ts) + 1)>::type;

  CombinationsFilteredIndexPolicyBase(o2::framework::expressions::Operations& operationSpecs, std::vector<int>& bindingIndices, std::vector<std::string>& columnNames, const Filtered<T>& filteredTable, const T& table, const Ts&... tables) : mIsEnd(false),
                                                                                                                                                                                                                                                mMaxOffset(filteredTable.end().index, tables.end().index...),
                                                                                                                                                                                                                                                mCurrent(filteredTable.begin(), tables.begin()...),
                                                                                                                                                                                                                                                mTables({table, tables...}),
                                                                                                                                                                                                                                                mBindingIndices(bindingIndices),
                                                                                                                                                                                                                                                mOperationSpecs(operationSpecs),
                                                                                                                                                                                                                                                mColumnNames(columnNames)
  {
    if (table.size() == 0 || ((tables.size() == 0) || ...)) {
      mIsEnd = true;
    }
  }

  void moveToEnd() {}

  // Note: Order of adding reversed!
  void addOne()
  {
    // First iterator checked separately
    std::get<0>(this->mCurrent)++;
    if (*std::get<1>(std::get<0>(this->mCurrent).getIndices()) == std::get<0>(this->mMaxOffset)) {
      // Move other iterators
      addAndUpdate();
    }
  }

  virtual void addAndUpdate() = 0;

  void updateFiltered(const o2::framework::expressions::Selection& selection, int index)
  {
    Filtered<T> filteredTable{{this->mTables[0].asArrowTable()}, selection};
    // For some reason `std::get<0>(this->mCurrent) = filteredTable.begin();` gives segv or dummy values
    auto filteredBeginIt = filteredTable.begin() + index;
    std::get<0>(this->mCurrent) = filteredBeginIt;
    std::get<0>(this->mMaxOffset) = filteredTable.end().index;
  }

  CombinationType mCurrent;
  IndicesType mMaxOffset;                                 // one position past maximum acceptable position for each element of combination
  bool mIsEnd;                                            // whether there are any more tuples available
  std::vector<int> mBindingIndices;                       // indices to former binding nodes in operation specs
  std::vector<std::string> mColumnNames;                  // column names to update values indexed by mBindingIndices
  o2::framework::expressions::Operations mOperationSpecs; // all nodes from filter
  std::vector<T> mTables;                                 // needs to store the original arrays
};

template <typename T, typename... Ts>
struct CombinationsFilteredUpperIndexPolicy : public CombinationsFilteredIndexPolicyBase<T, Ts...> {
  using CombinationType = std::tuple<typename Filtered<T>::iterator, typename Ts::iterator...>;

  CombinationsFilteredUpperIndexPolicy(o2::framework::expressions::Operations& operationSpecs, std::vector<int>& bindingIndices, std::vector<std::string>& columnNames, const Filtered<T>& filteredTable, const T& table, const Ts&... tables) : CombinationsFilteredIndexPolicyBase<T, Ts...>(operationSpecs, bindingIndices, columnNames, filteredTable, table, tables...)
  {
    if (filteredTable.size() == 0) {
      addAndUpdate();
    }
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).moveToEnd();
    });
    this->mIsEnd = true;
  }

  void addAndUpdate() override
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    o2::framework::expressions::Selection selection;
    int beginSelectionIndex = 0;

    do {
      modify = true;
      // Other iterators need to be moved, filter updated
      for_<k>([&, this](auto i) {
        if (modify) {
          constexpr auto curInd = i.value + 1;
          std::get<curInd>(this->mCurrent)++;
          if (*std::get<1>(std::get<curInd>(this->mCurrent).getIndices()) != std::get<curInd>(this->mMaxOffset)) {
            if (curInd - 1 < this->mBindingIndices.size()) {
              setColumnValue(this->mTables[curInd], this->mColumnNames[curInd - 1], this->mOperationSpecs[this->mBindingIndices[curInd - 1]].left, *std::get<1>(std::get<curInd>(this->mCurrent).getIndices()));
            }
            for_<i.value>([&, this](auto j) {
              constexpr auto curJ = i.value - j.value;
              std::get<curJ>(this->mCurrent).setCursor(*std::get<1>(std::get<curJ + 1>(this->mCurrent).getIndices()));
              if (curJ - 1 < this->mBindingIndices.size()) {
                setColumnValue(this->mTables[curJ], this->mColumnNames[curJ - 1], this->mOperationSpecs[this->mBindingIndices[curJ - 1]].left, *std::get<1>(std::get<curJ>(this->mCurrent).getIndices()));
              }
            });
            modify = false;
          }
        }
      });

      selection = o2::framework::expressions::createSelection(this->mTables[0].asArrowTable(), createFilter(this->mTables[0].asArrowTable()->schema(), this->mOperationSpecs));

      // Find first index bigger than the index of 1st iterator
      for (beginSelectionIndex = 0; beginSelectionIndex < selection->GetNumSlots() &&
                                    selection->GetIndex(beginSelectionIndex) < *std::get<1>(std::get<1>(this->mCurrent).getIndices());
           beginSelectionIndex++)
        ;
    } while (!modify && beginSelectionIndex == selection->GetNumSlots());

    this->updateFiltered(selection, beginSelectionIndex);
    this->mIsEnd = modify;
  }
};

template <typename T, typename... Ts>
struct CombinationsFilteredStrictlyUpperIndexPolicy : public CombinationsFilteredIndexPolicyBase<T, Ts...> {
  using CombinationType = std::tuple<typename Filtered<T>::iterator, typename Ts::iterator...>;

  CombinationsFilteredStrictlyUpperIndexPolicy(o2::framework::expressions::Operations& operationSpecs, std::vector<int>& bindingIndices, std::vector<std::string>& columnNames, const Filtered<T>& filteredTable, const T& table, const Ts&... tables) : CombinationsFilteredIndexPolicyBase<T, Ts...>(operationSpecs, bindingIndices, columnNames, filteredTable, table, tables...)
  {
    constexpr auto k = sizeof...(Ts);
    if (table.size() < k || ((tables.size() < k) || ...)) {
      this->mIsEnd = true;
      return;
    }
    for_<k>([&, this](auto i) {
      constexpr auto curInd = i.value + 1;
      std::get<curInd>(this->mMaxOffset) -= curInd;
      std::get<curInd>(this->mCurrent).moveByIndex(k - curInd);
      if (curInd - 1 < this->mBindingIndices.size()) {
        setColumnValue(this->mTables[curInd], this->mColumnNames[curInd - 1], this->mOperationSpecs[this->mBindingIndices[curInd - 1]].left, k - curInd);
      }
    });

    o2::framework::expressions::Selection selection = o2::framework::expressions::createSelection(this->mTables[0].asArrowTable(), createFilter(this->mTables[0].asArrowTable()->schema(), this->mOperationSpecs));
    // Find first index bigger than the index of 1st iterator
    int beginSelectionIndex = 0;
    for (; beginSelectionIndex < selection->GetNumSlots() &&
           selection->GetIndex(beginSelectionIndex) <= *std::get<1>(std::get<1>(this->mCurrent).getIndices());
         beginSelectionIndex++)
      ;
    if (beginSelectionIndex == selection->GetNumSlots()) {
      addAndUpdate();
    } else {
      this->updateFiltered(selection, beginSelectionIndex);
    }
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value + 1>(this->mCurrent).setCursor(std::get<i.value + 1>(this->mMaxOffset));
    });
    std::get<0>(this->mCurrent).moveToEnd();
    this->mIsEnd = true;
  }

  void addAndUpdate() override
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    o2::framework::expressions::Selection selection;
    int beginSelectionIndex = 0;

    do {
      modify = true;
      // Other iterators need to be moved, filter updated
      for_<k>([&, this](auto i) {
        if (modify) {
          constexpr auto curInd = i.value + 1;
          std::get<curInd>(this->mCurrent)++;
          if (*std::get<1>(std::get<curInd>(this->mCurrent).getIndices()) != std::get<curInd>(this->mMaxOffset)) {
            if (curInd - 1 < this->mBindingIndices.size()) {
              setColumnValue(this->mTables[curInd], this->mColumnNames[curInd - 1], this->mOperationSpecs[this->mBindingIndices[curInd - 1]].left, *std::get<1>(std::get<curInd>(this->mCurrent).getIndices()));
            }
            for_<i.value>([&, this](auto j) {
              constexpr auto curJ = i.value - j.value;
              std::get<curJ>(this->mCurrent).setCursor(*std::get<1>(std::get<curJ + 1>(this->mCurrent).getIndices()) + 1);
              if (curJ - 1 < this->mBindingIndices.size()) {
                setColumnValue(this->mTables[curJ], this->mColumnNames[curJ - 1], this->mOperationSpecs[this->mBindingIndices[curJ - 1]].left, *std::get<1>(std::get<curJ>(this->mCurrent).getIndices()));
              }
            });
            modify = false;
          }
        }
      });

      selection = o2::framework::expressions::createSelection(this->mTables[0].asArrowTable(), createFilter(this->mTables[0].asArrowTable()->schema(), this->mOperationSpecs));

      // Find first index bigger than the index of 1st iterator
      for (beginSelectionIndex = 0; beginSelectionIndex < selection->GetNumSlots() &&
                                    selection->GetIndex(beginSelectionIndex) <= *std::get<1>(std::get<1>(this->mCurrent).getIndices());
           beginSelectionIndex++)
        ;
    } while (!modify && beginSelectionIndex == selection->GetNumSlots());

    this->updateFiltered(selection, beginSelectionIndex);
    this->mIsEnd = modify;
  }
};

template <typename T, typename... Ts>
struct CombinationsFilteredFullIndexPolicy : public CombinationsFilteredIndexPolicyBase<T, Ts...> {
  using CombinationType = std::tuple<typename Filtered<T>::iterator, typename Ts::iterator...>;

  CombinationsFilteredFullIndexPolicy(o2::framework::expressions::Operations& operationSpecs, std::vector<int>& bindingIndices, std::vector<std::string>& columnNames, const Filtered<T>& filteredTable, const T& table, const Ts&... tables) : CombinationsFilteredIndexPolicyBase<T, Ts...>(operationSpecs, bindingIndices, columnNames, filteredTable, table, tables...)
  {
    if (filteredTable.size() == 0) {
      addAndUpdate();
    }
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).moveToEnd();
    });
    this->mIsEnd = true;
  }

  void addAndUpdate() override
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    o2::framework::expressions::Selection selection;

    do {
      modify = true;
      // Other iterators need to be moved, filter updated
      for_<k>([&, this](auto i) {
        if (modify) {
          constexpr auto curInd = i.value + 1;
          std::get<curInd>(this->mCurrent)++;
          if (*std::get<1>(std::get<curInd>(this->mCurrent).getIndices()) != std::get<curInd>(this->mMaxOffset)) {
            if (curInd - 1 < this->mBindingIndices.size()) {
              setColumnValue(this->mTables[curInd], this->mColumnNames[curInd - 1], this->mOperationSpecs[this->mBindingIndices[curInd - 1]].left, *std::get<1>(std::get<curInd>(this->mCurrent).getIndices()));
            }
            for_<i.value>([&, this](auto j) {
              constexpr auto curJ = i.value - j.value;
              std::get<curJ>(this->mCurrent).setCursor(0);
              if (curJ - 1 < this->mBindingIndices.size()) {
                setColumnValue(this->mTables[curJ], this->mColumnNames[curJ - 1], this->mOperationSpecs[this->mBindingIndices[curJ - 1]].left, 0);
              }
            });
            modify = false;
          }
        }
      });

      selection = o2::framework::expressions::createSelection(this->mTables[0].asArrowTable(), createFilter(this->mTables[0].asArrowTable()->schema(), this->mOperationSpecs));
    } while (!modify && selection->GetNumSlots() == 0);

    this->updateFiltered(selection, 0);
    this->mIsEnd = modify;
  }
};

/// @return next combination of rows of tables.
/// FIXME: move to coroutines once we have C++20
template <typename P>
struct CombinationsGenerator {
  using CombinationType = typename P::CombinationType;

 public:
  struct CombinationsIterator : public std::iterator<std::forward_iterator_tag, CombinationType>, public P {
   public:
    using reference = CombinationType&;
    using value_type = CombinationType;
    using pointer = CombinationType*;
    using iterator_category = std::forward_iterator_tag;

    CombinationsIterator() = delete;

    CombinationsIterator(const P& policy) : P(policy) {}

    CombinationsIterator(CombinationsIterator const&) = default;
    CombinationsIterator& operator=(CombinationsIterator const&) = default;
    ~CombinationsIterator() = default;

    // prefix increment
    CombinationsIterator& operator++()
    {
      if (!this->mIsEnd) {
        this->addOne();
      }
      return *this;
    }
    // postfix increment
    CombinationsIterator operator++(int /*unused*/)
    {
      CombinationsIterator copy(*this);
      operator++();
      return copy;
    }
    // return reference
    reference operator*()
    {
      return this->mCurrent;
    }
    bool operator==(const CombinationsIterator& rh)
    {
      return (this->mIsEnd && rh.mIsEnd) || (this->mCurrent == rh.mCurrent);
    }
    bool operator!=(const CombinationsIterator& rh)
    {
      return !(*this == rh);
    }
  };

  using iterator = CombinationsIterator;
  using const_iterator = CombinationsIterator;

  inline iterator begin()
  {
    return iterator(mBegin);
  }
  inline iterator end()
  {
    return iterator(mEnd);
  }
  inline const_iterator begin() const
  {
    return iterator(mBegin);
  }
  inline const_iterator end() const
  {
    return iterator(mEnd);
  }

  CombinationsGenerator() = delete;
  CombinationsGenerator(const P& policy) : mBegin(policy), mEnd(policy)
  {
    mEnd.moveToEnd();
  }
  ~CombinationsGenerator() = default;

 private:
  iterator mBegin;
  iterator mEnd;
};

// Provisional version to enable filtering on same-table combinations with current filters API
template <typename T2, typename... T2s>
auto selfCombinations(const o2::framework::expressions::Filter& filter, const T2& table, const T2s&... tables)
{
  return selfCombinations<CombinationsFilteredStrictlyUpperIndexPolicy>(filter, table, tables...);
}

template <template <typename...> typename P2, typename T2, typename... T2s>
auto selfCombinations(const o2::framework::expressions::Filter& filter, const T2& table, const T2s&... tables)
{
  static_assert(std::conjunction_v<std::is_same<T2, T2s>...>, "Input tables must be of the same type!");
  o2::framework::expressions::Operations operationSpecs = createOperations(filter);
  std::vector<int> bindingIndices;
  std::vector<std::string> columnNames;
  // Operations traverse the filter in DFS order, from right to left.
  // So the spec corresponding to the first BindingNode is the last binding spec
  // in a simple tree (a single condition)
  int i = operationSpecs.size() - 1;
  while (i >= 0 && operationSpecs[i].left.datum.index() != 3) {
    i--;
  }
  for (int j = i - 1; j >= 0; j--) {
    if (operationSpecs[j].left.datum.index() == 3) {
      std::string columnName = std::get<std::string>(operationSpecs[j].left.datum);
      bindingIndices.push_back(j);
      columnNames.push_back(columnName);
      setColumnValue(table, columnName, operationSpecs[j].left, 0);
    }
  }
  Filtered<T2> filteredTable{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), o2::framework::expressions::createFilter(table.asArrowTable()->schema(), operationSpecs))};
  return CombinationsGenerator<P2<T2, T2s...>>(P2(operationSpecs, bindingIndices, columnNames, filteredTable, table, tables...));
}

template <typename T2, typename... T2s>
auto selfCombinations(const char* categoryColumnName, const T2& table, const T2s&... tables)
{
  static_assert(std::conjunction_v<std::is_same<T2, T2s>...>, "Tables must have the same type for self combinations");
  return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<T2, T2s...>>(CombinationsBlockStrictlyUpperSameIndexPolicy(categoryColumnName, table, tables...));
}

template <typename T2, typename... T2s>
auto combinations(const char* categoryColumnName, const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsBlockStrictlyUpperIndexPolicy<T2, T2s...>>(CombinationsBlockStrictlyUpperIndexPolicy(categoryColumnName, table, tables...));
  } else {
    return CombinationsGenerator<CombinationsBlockFullIndexPolicy<T2, T2s...>>(CombinationsBlockFullIndexPolicy(categoryColumnName, table, tables...));
  }
}

template <typename T2, typename... T2s>
auto combinations(const char* categoryColumnName, const o2::framework::expressions::Filter& filter, const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsBlockStrictlyUpperIndexPolicy<Filtered<T2>, Filtered<T2s>...>>(CombinationsBlockStrictlyUpperIndexPolicy(categoryColumnName, Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  } else {
    return CombinationsGenerator<CombinationsBlockFullIndexPolicy<Filtered<T2>, Filtered<T2s>...>>(CombinationsBlockFullIndexPolicy(categoryColumnName, Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  }
}

template <typename T2, typename... T2s>
auto combinations(const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<T2, T2s...>>(CombinationsStrictlyUpperIndexPolicy(table, tables...));
  } else {
    return CombinationsGenerator<CombinationsFullIndexPolicy<T2, T2s...>>(CombinationsFullIndexPolicy(table, tables...));
  }
}

template <typename T2, typename... T2s>
auto combinations(const o2::framework::expressions::Filter& filter, const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<Filtered<T2>, Filtered<T2s>...>>(CombinationsStrictlyUpperIndexPolicy(Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  } else {
    return CombinationsGenerator<CombinationsFullIndexPolicy<Filtered<T2>, Filtered<T2s>...>>(CombinationsFullIndexPolicy(Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  }
}

template <template <typename...> typename P2, typename... T2s>
CombinationsGenerator<P2<Filtered<T2s>...>> combinations(const P2<T2s...>& policy, const o2::framework::expressions::Filter& filter, const T2s&... tables)
{
  return CombinationsGenerator<P2<Filtered<T2s>...>>(P2<Filtered<T2s>...>({{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
}

// This shortened version cannot be used for Filtered
// (unless users create filtered tables themselves before policy creation)
template <template <typename...> typename P2, typename... T2s>
CombinationsGenerator<P2<T2s...>> combinations(const P2<T2s...>& policy)
{
  return CombinationsGenerator<P2<T2s...>>(policy);
}

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOAHELPERS_H_
