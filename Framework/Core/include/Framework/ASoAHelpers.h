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
#include <arrow/table.h>
#include "Framework/ArrowCompatibility.h"

#include <arrow/compute/context.h>

// arrow 15.0
//#include <arrow/compute/kernels/sort_to_indices.h>

#include <iterator>
#include <tuple>
#include <utility>

namespace o2::soa
{

// Functions to enable looping over tuples
template <std::size_t V>
struct num {
  static const constexpr auto value = V;
};

template <class F, std::size_t... Is>
void for_(F func, std::index_sequence<Is...>)
{
  using expander = int[];
  (void)expander{0, ((void)func(num<Is>{}), 0)...};
}

template <std::size_t N, typename F>
void for_(F func)
{
  for_(func, std::make_index_sequence<N>());
}

// Creating tuple of given size and type
template <typename T, unsigned N, typename... REST>
struct generate_tuple_type {
  using type = typename generate_tuple_type<T, N - 1, T, REST...>::type;
};

template <typename T, typename... REST>
struct generate_tuple_type<T, 0, REST...> {
  using type = std::tuple<REST...>;
};

// Group table (C++ vector of indices)
bool sameCategory(std::pair<uint64_t, uint64_t> const& a, std::pair<uint64_t, uint64_t> const& b)
{
  return a.first < b.first;
}

template <typename T2, typename ARRAY, typename T>
std::vector<std::pair<uint64_t, uint64_t>> doGroupTable(const T& inTable, const std::string& categoryColumnName, int minCatSize)
{
  auto arrowTable = inTable.asArrowTable();
  auto columnIndex = arrowTable->schema()->GetFieldIndex(categoryColumnName);
  auto chunkedArray = framework::getBackendColumnData(arrowTable->column(columnIndex));

  uint64_t ind = 0;
  std::vector<std::pair<uint64_t, uint64_t>> groupedIndices;
  groupedIndices.reserve(arrowTable->num_rows());
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
auto groupTable(const T& inTable, const std::string& categoryColumnName, int minCatSize)
{
  auto arrowTable = inTable.asArrowTable();
  auto columnIndex = arrowTable->schema()->GetFieldIndex(categoryColumnName);
  auto dataType = arrowTable->column(columnIndex)->type();
  if (dataType->id() == arrow::Type::UINT64) {
    return doGroupTable<uint64_t, arrow::UInt64Array>(inTable, categoryColumnName, minCatSize);
  }
  if (dataType->id() == arrow::Type::INT64) {
    return doGroupTable<int64_t, arrow::Int64Array>(inTable, categoryColumnName, minCatSize);
  }
  if (dataType->id() == arrow::Type::UINT32) {
    return doGroupTable<uint32_t, arrow::UInt32Array>(inTable, categoryColumnName, minCatSize);
  }
  if (dataType->id() == arrow::Type::INT32) {
    return doGroupTable<int32_t, arrow::Int32Array>(inTable, categoryColumnName, minCatSize);
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
  std::unique_copy(groupedIndices[0].begin(), groupedIndices[0].end(), std::back_inserter(firstCategories), sameCategory);

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
  using IndicesType = typename generate_tuple_type<uint64_t, sizeof...(Ts)>::type;

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
  CombinationsStrictlyUpperIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...)
  {
    constexpr auto k = sizeof...(Ts);
    if (((tables.size() < k) || ...)) {
      this->mIsEnd = true;
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
  using IndicesType = typename generate_tuple_type<uint64_t, sizeof...(Ts)>::type;

  CombinationsBlockUpperIndexPolicy(const std::string& categoryColumnName, const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...)
  {
    constexpr auto k = sizeof...(Ts);
    if (this->mIsEnd) {
      return;
    }

    int tableIndex = 0;
    ((this->mGroupedIndices[tableIndex++] = groupTable(tables, categoryColumnName, 1)), ...);

    // Synchronize categories across tables
    // FIXME: This wouldn't be necessary if tables were always the same, are they?
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
  using IndicesType = typename generate_tuple_type<uint64_t, sizeof...(Ts)>::type;

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
    // FIXME: This wouldn't be necessary if tables were always the same, are they?
    syncCategories(this->mGroupedIndices);

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
  using IndicesType = typename generate_tuple_type<uint64_t, sizeof...(Ts)>::type;

  CombinationsBlockFullIndexPolicy(const std::string& categoryColumnName, const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...)
  {
    if (this->mIsEnd) {
      return;
    }

    constexpr auto k = sizeof...(Ts);

    // FIXME: What is preferred from DPL/Framework internals point-of-view?
    // grouping arrow tables directly vs C++ vectors / tuples / ...

    // Grouping with arrow
    // Note: this expects the input table to be already sorted by category
    // Could use arrow's SortToIndices from arrow 15.0 for sorting (or sth based on that)
    //std::vector<arrow::Table> ranges;
    //ranges.reserve(k);
    //auto groupTableArrow = [&] (const auto& inputTable) {
    //  arrow::compute::FunctionContext ctx;
    //  arrow::compute::Datum outRanges;
    //  SortedGroupByKernel groupBy{{categoryColumnName}};
    //  auto result = groupBy.Call(&ctx, arrow::compute::Datum(inputTable.asArrowTable()), &outRanges);
    //  if (result.ok() == false) {
    //    throw std::runtime_error("Combinations: cannot slice table for grouping");
    //  }
    //  ranges.emplace_back(arrow::util::get<std::shared_ptr<arrow::Table>>(outRanges.value));
    //};

    // Grouping with C++ containers - based on RCombinedDS
    int tableIndex = 0;
    ((this->mGroupedIndices[tableIndex++] = groupTable(tables, categoryColumnName, 1)), ...);
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

  // FIXME: Could it be simplified? E.g. by assuming that all input tables are the same table?
  // Or calculating grouped indices on-fly? (How?)
  std::array<std::vector<std::pair<uint64_t, uint64_t>>, sizeof...(Ts)> mGroupedIndices;
  IndicesType mCurrentIndices;
  IndicesType mBeginIndices;
};

/// @return next combination of rows of tables.
/// FIXME: move to coroutines once we have C++20
template <template <typename...> typename P, typename... Ts>
struct CombinationsGenerator {
 public:
  using CombinationType = std::tuple<typename Ts::iterator...>;

  struct CombinationsIterator : public std::iterator<std::forward_iterator_tag, CombinationType>, public P<Ts...> {
   public:
    using reference = CombinationType&;
    using value_type = CombinationType;
    using pointer = CombinationType*;
    using iterator_category = std::forward_iterator_tag;

    CombinationsIterator() = delete;

    CombinationsIterator(const P<Ts...>& policy) : P<Ts...>(policy) {}

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
  CombinationsGenerator(const P<Ts...>& policy) : mBegin(policy), mEnd(policy)
  {
    static_assert(sizeof...(Ts) > 0);
    mEnd.moveToEnd();
  }
  ~CombinationsGenerator() = default;

 private:
  iterator mBegin;
  iterator mEnd;
};

template <typename T2, typename... T2s>
auto combinations(const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, T2, T2s...>(CombinationsStrictlyUpperIndexPolicy(table, tables...));
  } else {
    return CombinationsGenerator<CombinationsFullIndexPolicy, T2, T2s...>(CombinationsFullIndexPolicy(table, tables...));
  }
}

template <typename T2, typename... T2s>
auto combinations(const o2::framework::expressions::Filter& filter, const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy, Filtered<T2>, Filtered<T2s>...>(CombinationsStrictlyUpperIndexPolicy(Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  } else {
    return CombinationsGenerator<CombinationsFullIndexPolicy, Filtered<T2>, Filtered<T2s>...>(CombinationsFullIndexPolicy(Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  }
}

template <template <typename...> typename P2, typename... T2s>
CombinationsGenerator<P2, T2s...> combinations(const P2<T2s...>& policy, const T2s&... tables)
{
  return CombinationsGenerator<P2, T2s...>(P2<T2s...>(tables...));
}

template <template <typename...> typename P2, typename... T2s>
CombinationsGenerator<P2, Filtered<T2s>...> combinations(const P2<T2s...>& policy, const o2::framework::expressions::Filter& filter, const T2s&... tables)
{
  return CombinationsGenerator<P2, Filtered<T2s>...>(P2<Filtered<T2s>...>({{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
}

// This shortened version cannot be used for Filtered
// (unless users create filtered tables themselves before policy creation)
template <template <typename...> typename P2, typename... T2s>
CombinationsGenerator<P2, T2s...> combinations(const P2<T2s...>& policy)
{
  return CombinationsGenerator<P2, T2s...>(policy);
}

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOAHELPERS_H_
