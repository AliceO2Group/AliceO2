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

#ifndef O2_FRAMEWORK_ASOAHELPERS_H_
#define O2_FRAMEWORK_ASOAHELPERS_H_

#include "Framework/ASoA.h"
#include "Framework/BinningPolicy.h"
#include "Framework/RuntimeError.h"
#include <arrow/table.h>

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

struct BinningIndex {
  BinningIndex(int bin_, uint64_t index_) : bin(bin_), index(index_) {}

  bool operator<(const BinningIndex& rhs) const { return std::tie(bin, index) < std::tie(rhs.bin, rhs.index); }

  int bin;
  uint64_t index;
};

// Group table (C++ vector of indices)
inline bool sameCategory(BinningIndex const& a, BinningIndex const& b)
{
  return a.bin < b.bin;
}
inline bool diffCategory(BinningIndex const& a, BinningIndex const& b)
{
  return a.bin >= b.bin;
}

template <template <typename... Cs> typename BP, typename T, typename... Cs>
std::vector<BinningIndex> groupTable(const T& table, const BP<Cs...>& binningPolicy, int minCatSize, int outsider)
{
  arrow::Table* arrowTable = table.asArrowTable().get();
  auto rowIterator = table.begin();

  uint64_t ind = 0;
  uint64_t selInd = 0;
  gsl::span<int64_t const> selectedRows;
  std::vector<BinningIndex> groupedIndices;

  // Separate check to account for Filtered size different from arrow table
  if (table.size() == 0) {
    return groupedIndices;
  }

  if constexpr (soa::is_soa_filtered_v<T>) {
    selectedRows = table.getSelectedRows(); // vector<int64_t>
  }

  auto persistentColumns = typename BP<Cs...>::persistent_columns_t{};
  constexpr auto persistentColumnsCount = pack_size(persistentColumns);
  auto arrowColumns = o2::soa::row_helpers::getArrowColumnsTyped(table, persistentColumns);
  auto chunksCount = arrowColumns[0]->num_chunks();
  for (int i = 1; i < persistentColumnsCount; i++) {
    if (arrowColumns[i]->num_chunks() != chunksCount) {
      throw o2::framework::runtime_error("Combinations: data size varies between selected columns");
    }
  }

  for (uint64_t ci = 0; ci < chunksCount; ++ci) {
    auto chunks = o2::soa::row_helpers::getChunksFromColumns(arrowColumns, ci);
    auto chunkLength = std::get<0>(chunks)->length();
    for_<persistentColumnsCount - 1>([&chunks, &chunkLength](auto i) {
      if (std::get<i.value + 1>(chunks)->length() != chunkLength) {
        throw o2::framework::runtime_error("Combinations: data size varies between selected columns");
      }
    });

    if constexpr (soa::is_soa_filtered_v<T>) {
      if (selectedRows[ind] >= selInd + chunkLength) {
        selInd += chunkLength;
        continue; // Go to the next chunk, no value selected in this chunk
      }
    }

    uint64_t ai = 0;
    while (ai < chunkLength) {
      if constexpr (soa::is_soa_filtered_v<T>) {
        ai += selectedRows[ind] - selInd;
        selInd = selectedRows[ind];
      }

      auto values = binningPolicy.getBinningValues(rowIterator, arrowTable, ci, ai, ind);
      auto val = binningPolicy.getBin(values);
      if (val != outsider) {
        groupedIndices.emplace_back(val, ind);
      }
      ind++;

      if constexpr (soa::is_soa_filtered_v<T>) {
        if (ind >= selectedRows.size()) {
          break;
        }
      } else {
        ai++;
      }
    }

    if constexpr (soa::is_soa_filtered_v<T>) {
      if (ind == selectedRows.size()) {
        break;
      }
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

// Synchronize categories so as groupedIndices contain elements only of categories common to all tables
template <std::size_t K>
void syncCategories(std::array<std::vector<BinningIndex>, K>& groupedIndices)
{
  std::vector<BinningIndex> firstCategories;
  std::vector<BinningIndex> commonCategories;
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

  CombinationsIndexPolicyBase() : mIsEnd(true) {}
  template <typename... Tss>
  CombinationsIndexPolicyBase(const Tss&... tables) : mIsEnd(false),
                                                      mMaxOffset(tables.end().index...),
                                                      mCurrent(tables.begin()...)
  {
    if (((tables.size() == 0) || ...)) {
      this->mIsEnd = true;
    }
  }
  template <typename... Tss>
  CombinationsIndexPolicyBase(Tss&&... tables) : mTables(std::make_shared<std::tuple<Tss...>>(std::make_tuple(std::move(tables)...))),
                                                 mIsEnd(false)
  {
    std::apply([&](auto&&... x) mutable { mMaxOffset = IndicesType{x.end().index...}; mCurrent = CombinationType{x.begin()...}; }, *mTables);
    if (
      std::apply([](auto&&... x) -> bool { return ((x.size() == 0) || ...); }, *mTables)) {
      this->mIsEnd = true;
    }
  }

  void setTables(const Ts&... tables)
  {
    mIsEnd = false;
    mMaxOffset = IndicesType(tables.end().index...);
    mCurrent = CombinationType(tables.begin()...);
    if (((tables.size() == 0) || ...)) {
      this->mIsEnd = true;
    }
  }
  template <typename... Tss>
  void setTables(Tss&&... tables)
  {
    mIsEnd = false;
    mTables = std::make_shared<std::tuple<Tss...>>(std::make_tuple(std::move(tables)...));
    std::apply([&](auto&&... x) mutable { mMaxOffset = IndicesType{x.end().index...}; mCurrent = CombinationType{x.begin()...}; }, *mTables);
    if (
      std::apply([](auto&&... x) -> bool { return ((x.size() == 0) || ...); }, *mTables)) {
      this->mIsEnd = true;
    }
  }

  void moveToEnd()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrent).moveToEnd();
    });
    this->mIsEnd = true;
  }

  void addOne() {}

  std::shared_ptr<std::tuple<Ts...>> mTables;
  CombinationType mCurrent;
  IndicesType mMaxOffset; // one position past maximum acceptable position for each element of combination
  bool mIsEnd;            // whether there are any more tuples available
};

template <typename... Ts>
CombinationsIndexPolicyBase(Ts const&... tables) -> CombinationsIndexPolicyBase<Ts...>;

template <typename... Ts>
CombinationsIndexPolicyBase(Ts&&... tables) -> CombinationsIndexPolicyBase<Ts...>;

template <typename... Ts>
struct CombinationsUpperIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;

  CombinationsUpperIndexPolicy() : CombinationsIndexPolicyBase<Ts...>() {}
  CombinationsUpperIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...) {}
  CombinationsUpperIndexPolicy(Ts&&... tables) : CombinationsIndexPolicyBase<Ts...>(std::forward<Ts>(tables)...) {}

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrent)++;
        if (*std::get<1>(std::get<curInd>(this->mCurrent).getIndices()) != std::get<curInd>(this->mMaxOffset)) {
          modify = false;
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            int64_t nextInd = *std::get<1>(std::get<curJ - 1>(this->mCurrent).getIndices());
            if (nextInd < std::get<curJ>(this->mMaxOffset)) {
              std::get<curJ>(this->mCurrent).setCursor(nextInd);
            } else {
              modify = true;
            }
          });
        }
      }
    });
    this->mIsEnd = modify;
  }
};

template <typename... Ts>
struct CombinationsStrictlyUpperIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;

  CombinationsStrictlyUpperIndexPolicy() : CombinationsIndexPolicyBase<Ts...>() {}
  CombinationsStrictlyUpperIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...)
  {
    if (!this->mIsEnd) {
      setRanges(tables...);
    }
  }
  CombinationsStrictlyUpperIndexPolicy(Ts&&... tables) : CombinationsIndexPolicyBase<Ts...>(std::forward<Ts>(tables)...)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setTables(const Ts&... tables)
  {
    CombinationsIndexPolicyBase<Ts...>::setTables(tables...);
    setRanges(tables...);
  }
  void setTables(Ts&&... tables)
  {
    CombinationsIndexPolicyBase<Ts...>::setTables(std::forward<Ts>(tables)...);
    setRanges(tables...);
  }

  void setRanges(const Ts&... tables)
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
  void setRanges()
  {
    constexpr auto k = sizeof...(Ts);
    if (
      std::apply([](auto&&... x) -> bool { return ((x.size() < k) || ...); }, *this->mTables)) {
      this->mIsEnd = true;
      return;
    }
    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mMaxOffset) += i.value + 1 - k;
      std::get<i.value>(this->mCurrent).moveByIndex(i.value);
    });
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
          modify = false;
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            int64_t nextInd = *std::get<1>(std::get<curJ - 1>(this->mCurrent).getIndices()) + 1;
            if (nextInd < std::get<curJ>(this->mMaxOffset)) {
              std::get<curJ>(this->mCurrent).setCursor(nextInd);
            } else {
              modify = true;
            }
          });
        }
      }
    });
    this->mIsEnd = modify;
  }
};

template <typename... Ts>
struct CombinationsFullIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;

  CombinationsFullIndexPolicy() : CombinationsIndexPolicyBase<Ts...>() {}
  CombinationsFullIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...) {}
  CombinationsFullIndexPolicy(Ts&&... tables) : CombinationsIndexPolicyBase<Ts...>(std::forward<Ts>(tables)...) {}

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

// For upper and full only
template <typename BP, typename T, typename... Ts>
struct CombinationsBlockIndexPolicyBase : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts)>::type;

  CombinationsBlockIndexPolicyBase(const BP& binningPolicy, int categoryNeighbours, const T& outsider) : CombinationsIndexPolicyBase<Ts...>(), mSlidingWindowSize(categoryNeighbours + 1), mBP(binningPolicy), mCategoryNeighbours(categoryNeighbours), mOutsider(outsider), mIsNewWindow(true) {}
  CombinationsBlockIndexPolicyBase(const BP& binningPolicy, int categoryNeighbours, const T& outsider, const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...), mSlidingWindowSize(categoryNeighbours + 1), mBP(binningPolicy), mCategoryNeighbours(categoryNeighbours), mOutsider(outsider), mIsNewWindow(true)
  {
    if (!this->mIsEnd) {
      setRanges(tables...);
    }
  }
  CombinationsBlockIndexPolicyBase(const BP& binningPolicy, int categoryNeighbours, const T& outsider, Ts&&... tables) : CombinationsIndexPolicyBase<Ts...>(std::forward<Ts>(tables)...), mSlidingWindowSize(categoryNeighbours + 1), mBP(binningPolicy), mCategoryNeighbours(categoryNeighbours), mOutsider(outsider), mIsNewWindow(true)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setTables(const Ts&... tables)
  {
    CombinationsIndexPolicyBase<Ts...>::setTables(tables...);
    setRanges(tables...);
  }
  void setTables(Ts&&... tables)
  {
    CombinationsIndexPolicyBase<Ts...>::setTables(std::forward<Ts>(tables)...);
    setRanges();
  }

  void setRanges(const Ts&... tables)
  {
    constexpr auto k = sizeof...(Ts);
    if (mSlidingWindowSize < 1) {
      this->mIsEnd = true;
      return;
    }

    int tableIndex = 0;
    ((this->mGroupedIndices[tableIndex++] = groupTable(tables, this->mBP, 1, this->mOutsider)), ...);

    // Synchronize categories across tables
    syncCategories(this->mGroupedIndices);

    for (int i = 0; i < k; i++) {
      if (this->mGroupedIndices[i].size() == 0) {
        this->mIsEnd = true;
        return;
      }
    }

    for_<k>([this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = 0;
    });
  }
  void setRanges(Ts&&... tables)
  {
    constexpr auto k = sizeof...(Ts);
    if (this->mIsEnd) {
      return;
    }
    if (mSlidingWindowSize < 1) {
      this->mIsEnd = true;
      return;
    }

    int tableIndex = 0;
    std::apply([&, this](auto&&... x) mutable {
      ((this->mGroupedIndices[tableIndex++] = groupTable(x, this->mBP, 1, this->mOutsider)), ...);
    },
               *this->mTables);

    // Synchronize categories across tables
    syncCategories(this->mGroupedIndices);

    for (int i = 0; i < k; i++) {
      if (this->mGroupedIndices[i].size() == 0) {
        this->mIsEnd = true;
        return;
      }
    }

    for_<k>([this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = 0;
    });
  }

  int currentWindowNeighbours()
  {
    // NOTE: The same number of currentWindowNeighbours is returned for all kinds of block combinations.
    // Strictly upper: the first element will is paired with exactly currentWindowNeighbours other elements.
    // Upper: the first element is paired with (currentWindowNeighbours + 1) elements, including itself.
    // Full: (currentWindowNeighbours + 1) pairs with the first element in the first position (c1)
    //       + there are other combinations with the first element at other positions.
    if (this->mIsEnd) {
      return 0;
    }
    uint64_t maxForWindow = std::get<0>(this->mBeginIndices) + this->mSlidingWindowSize - 1;
    uint64_t maxForTable = std::get<0>(this->mMaxOffset);
    uint64_t currentMax = maxForWindow < maxForTable ? maxForWindow : maxForTable;
    return currentMax - std::get<0>(mCurrentIndices);
  }

  bool isNewWindow()
  {
    return mIsNewWindow;
  }

  std::array<std::vector<BinningIndex>, sizeof...(Ts)> mGroupedIndices;
  IndicesType mCurrentIndices;
  IndicesType mBeginIndices;
  uint64_t mSlidingWindowSize;
  const BP mBP;
  const int mCategoryNeighbours;
  const T mOutsider;
  bool mIsNewWindow;
};

template <typename BP, typename T, typename... Ts>
struct CombinationsBlockUpperIndexPolicy : public CombinationsBlockIndexPolicyBase<BP, T, Ts...> {
  using CombinationType = typename CombinationsBlockIndexPolicyBase<BP, T, Ts...>::CombinationType;

  CombinationsBlockUpperIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T& outsider) : CombinationsBlockIndexPolicyBase<BP, T, Ts...>(binningPolicy, categoryNeighbours, outsider) {}
  CombinationsBlockUpperIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T& outsider, const Ts&... tables) : CombinationsBlockIndexPolicyBase<BP, T, Ts...>(binningPolicy, categoryNeighbours, outsider, tables...)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }
  CombinationsBlockUpperIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T& outsider, Ts&&... tables) : CombinationsBlockIndexPolicyBase<BP, T, Ts...>(binningPolicy, categoryNeighbours, outsider, std::forward<Ts>(tables)...)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setTables(const Ts&... tables)
  {
    CombinationsBlockIndexPolicyBase<BP, T, Ts...>::setTables(tables...);
    setRanges();
  }
  void setTables(Ts&&... tables)
  {
    CombinationsBlockIndexPolicyBase<BP, T, Ts...>::setTables(std::forward<Ts>(tables)...);
    setRanges();
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      auto catBegin = this->mGroupedIndices[i.value].begin() + std::get<i.value>(this->mCurrentIndices);
      auto range = std::equal_range(catBegin, this->mGroupedIndices[i.value].end(), *catBegin, sameCategory);
      std::get<i.value>(this->mBeginIndices) = std::distance(this->mGroupedIndices[i.value].begin(), range.first);
      std::get<i.value>(this->mCurrent).setCursor(range.first->index);
      std::get<i.value>(this->mMaxOffset) = std::distance(this->mGroupedIndices[i.value].begin(), range.second);
    });
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    bool nextCatAvailable = true;
    for_<k - 1>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        uint64_t maxForWindow = std::get<curInd>(this->mBeginIndices) + this->mSlidingWindowSize;

        // If we remain within the same sliding window
        if (curGroupedInd < maxForWindow && curGroupedInd < std::get<curInd>(this->mMaxOffset)) {
          std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curInd][curGroupedInd].index);
          modify = false;
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            if (std::get<curJ - 1>(this->mCurrentIndices) < std::get<curJ>(this->mMaxOffset)) {
              std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices);
              uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
              std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].index);
            } else {
              modify = true;
            }
          });
        }
      }
    });

    this->mIsNewWindow = modify;

    // First iterator processed separately
    if (modify) {
      std::get<0>(this->mCurrentIndices)++;
      std::get<0>(this->mBeginIndices)++;
      uint64_t curGroupedInd = std::get<0>(this->mCurrentIndices);

      // If we remain within the same category - slide window
      if (curGroupedInd < std::get<0>(this->mMaxOffset)) {
        std::get<0>(this->mCurrent).setCursor(this->mGroupedIndices[0][curGroupedInd].index);
        modify = false;
        for_<k - 1>([&, this](auto j) {
          constexpr auto curJ = j.value + 1;
          std::get<curJ>(this->mBeginIndices)++;
          if (std::get<curJ>(this->mBeginIndices) < std::get<curJ>(this->mMaxOffset)) {
            std::get<curJ>(this->mCurrentIndices) = std::get<curJ>(this->mBeginIndices);
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].index);
          } else {
            modify = true;
          }
        });
      }
    }

    // No more combinations within this category - move to the next category, if possible
    if (modify) {
      for_<k>([&, this](auto m) {
        std::get<m.value>(this->mCurrentIndices) = std::get<m.value>(this->mMaxOffset);
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
};

template <typename BP, typename T, typename... Ts>
struct CombinationsBlockFullIndexPolicy : public CombinationsBlockIndexPolicyBase<BP, T, Ts...> {
  using CombinationType = typename CombinationsBlockIndexPolicyBase<BP, T, Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts)>::type;

  CombinationsBlockFullIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T& outsider) : CombinationsBlockIndexPolicyBase<BP, T, Ts...>(binningPolicy, categoryNeighbours, outsider), mCurrentlyFixed(0) {}
  CombinationsBlockFullIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T& outsider, const Ts&... tables) : CombinationsBlockIndexPolicyBase<BP, T, Ts...>(binningPolicy, categoryNeighbours, outsider, tables...), mCurrentlyFixed(0)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }
  CombinationsBlockFullIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T& outsider, Ts&&... tables) : CombinationsBlockIndexPolicyBase<BP, T, Ts...>(binningPolicy, categoryNeighbours, outsider, std::forward<Ts>(tables)...), mCurrentlyFixed(0)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setTables(const Ts&... tables)
  {
    CombinationsBlockIndexPolicyBase<BP, T, Ts...>::setTables(tables...);
    setRanges();
  }
  void setTables(Ts&&... tables)
  {
    CombinationsBlockIndexPolicyBase<BP, T, Ts...>::setTables(std::forward<Ts>(tables)...);
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
      std::get<i.value>(this->mCurrent).setCursor(range.first->index);
    });
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
        uint64_t windowOffset = curInd == this->mCurrentlyFixed ? 1 : this->mSlidingWindowSize;
        uint64_t maxForWindow = std::get<curInd>(this->mBeginIndices) + windowOffset;

        // If we remain within the same sliding window and fixed index
        if (curGroupedInd < maxForWindow && curGroupedInd < std::get<curInd>(this->mMaxOffset)) {
          std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curInd][curGroupedInd].index);
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            if (curJ < this->mCurrentlyFixed) { // To assure no repetitions
              std::get<curJ>(this->mCurrentIndices) = std::get<curJ>(this->mBeginIndices) + 1;
            } else {
              std::get<curJ>(this->mCurrentIndices) = std::get<curJ>(this->mBeginIndices);
            }
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].index);
          });
          modify = false;
        }
      }
    });

    this->mIsNewWindow = modify;

    // Currently fixed iterator processed separately
    if (modify) {
      // If we haven't finished with window starting element
      if (this->mCurrentlyFixed < k - 1 && std::get<0>(this->mBeginIndices) < std::get<0>(this->mMaxOffset) - 1) {
        this->mCurrentlyFixed++;
        for_<k>([&, this](auto s) {
          if (s.value < this->mCurrentlyFixed) { // To assure no repetitions
            std::get<s.value>(this->mCurrentIndices) = std::get<s.value>(this->mBeginIndices) + 1;
          } else {
            std::get<s.value>(this->mCurrentIndices) = std::get<s.value>(this->mBeginIndices);
          }
          uint64_t curGroupedI = std::get<s.value>(this->mCurrentIndices);
          std::get<s.value>(this->mCurrent).setCursor(this->mGroupedIndices[s.value][curGroupedI].index);
        });
        modify = false;
      } else {
        this->mCurrentlyFixed = 0;
        std::get<0>(this->mBeginIndices)++;
        std::get<0>(this->mCurrentIndices) = std::get<0>(this->mBeginIndices);

        // If we remain within the same category - slide window
        if (std::get<0>(this->mBeginIndices) < std::get<0>(this->mMaxOffset)) {
          uint64_t curGroupedInd = std::get<0>(this->mCurrentIndices);
          std::get<0>(this->mCurrent).setCursor(this->mGroupedIndices[0][curGroupedInd].index);
          modify = false;
          for_<k - 1>([&, this](auto j) {
            constexpr auto curJ = j.value + 1;
            std::get<curJ>(this->mBeginIndices)++;
            if (std::get<curJ>(this->mBeginIndices) < std::get<curJ>(this->mMaxOffset)) {
              std::get<curJ>(this->mCurrentIndices) = std::get<curJ>(this->mBeginIndices);
              uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
              std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].index);
            } else {
              modify = true;
            }
          });
        }
      }
    }

    // No more combinations within this category - move to the next category, if possible
    if (modify) {
      for_<k>([&, this](auto m) {
        std::get<m.value>(this->mCurrentIndices) = std::get<m.value>(this->mMaxOffset);
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

  uint64_t mCurrentlyFixed;
};

template <typename BP, typename T1, typename T, typename... Ts>
struct CombinationsBlockSameIndexPolicyBase : public CombinationsIndexPolicyBase<T, Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<T, Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts) + 1>::type;

  CombinationsBlockSameIndexPolicyBase(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, int minWindowSize) : CombinationsIndexPolicyBase<T, Ts...>(), mSlidingWindowSize(categoryNeighbours + 1), mBP(binningPolicy), mCategoryNeighbours(categoryNeighbours), mOutsider(outsider), mMinWindowSize(minWindowSize), mIsNewWindow(true) {}
  CombinationsBlockSameIndexPolicyBase(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, int minWindowSize, const T& table, const Ts&... tables) : CombinationsIndexPolicyBase<T, Ts...>(table, tables...), mSlidingWindowSize(categoryNeighbours + 1), mBP(binningPolicy), mCategoryNeighbours(categoryNeighbours), mOutsider(outsider), mMinWindowSize(minWindowSize), mIsNewWindow(true)
  {
    if (!this->mIsEnd) {
      setRanges(table);
    }
  }
  CombinationsBlockSameIndexPolicyBase(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, int minWindowSize, T&& table, Ts&&... tables) : CombinationsIndexPolicyBase<T, Ts...>(std::forward<T>(table), std::forward<Ts>(tables)...), mSlidingWindowSize(categoryNeighbours + 1), mBP(binningPolicy), mCategoryNeighbours(categoryNeighbours), mOutsider(outsider), mMinWindowSize(minWindowSize), mIsNewWindow(true)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setTables(const T& table, const Ts&... tables)
  {
    CombinationsIndexPolicyBase<T, Ts...>::setTables(table, tables...);
    if (!this->mIsEnd) {
      setRanges(table);
    }
  }
  void setTables(T&& table, Ts&&... tables)
  {
    CombinationsIndexPolicyBase<T, Ts...>::setTables(std::forward<T>(table), std::forward<Ts>(tables)...);
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setRanges(const T& table)
  {
    constexpr auto k = sizeof...(Ts) + 1;
    // minWindowSize == 1 for upper and full, and k for strictly upper k-combination
    if (mSlidingWindowSize < mMinWindowSize) {
      this->mIsEnd = true;
      return;
    }

    this->mGroupedIndices = groupTable(table, mBP, mMinWindowSize, mOutsider);

    if (this->mGroupedIndices.size() == 0) {
      this->mIsEnd = true;
      return;
    }

    std::get<0>(this->mCurrentIndices) = 0;
  }
  void setRanges()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    // minWindowSize == 1 for upper and full, and k for strictly upper k-combination
    if (mSlidingWindowSize < mMinWindowSize) {
      this->mIsEnd = true;
      return;
    }

    this->mGroupedIndices = groupTable(std::get<0>(*this->mTables), mBP, mMinWindowSize, mOutsider);

    if (this->mGroupedIndices.size() == 0) {
      this->mIsEnd = true;
      return;
    }

    std::get<0>(this->mCurrentIndices) = 0;
  }

  int currentWindowNeighbours()
  {
    // NOTE: The same number of currentWindowNeighbours is returned for all kinds of block combinations.
    // Strictly upper: the first element will is paired with exactly currentWindowNeighbours other elements.
    // Upper: the first element is paired with (currentWindowNeighbours + 1) elements, including itself.
    // Full: (currentWindowNeighbours + 1) pairs with the first element in the first position (c1)
    //       + there are other combinations with the first element at other positions.
    if (this->mIsEnd) {
      return 0;
    }
    uint64_t maxForWindow = std::get<0>(this->mCurrentIndices) + this->mSlidingWindowSize - 1;
    uint64_t maxForTable = std::get<0>(this->mMaxOffset);
    uint64_t currentMax = maxForWindow < maxForTable ? maxForWindow : maxForTable;
    return currentMax - std::get<0>(mCurrentIndices);
  }

  bool isNewWindow()
  {
    return mIsNewWindow;
  }

  std::vector<BinningIndex> mGroupedIndices;
  IndicesType mCurrentIndices;
  const uint64_t mSlidingWindowSize;
  const int mMinWindowSize;
  const BP mBP;
  const int mCategoryNeighbours;
  const T1 mOutsider;
  bool mIsNewWindow;
};

template <typename BP, typename T1, typename... Ts>
struct CombinationsBlockUpperSameIndexPolicy : public CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...> {
  using CombinationType = typename CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>::CombinationType;

  CombinationsBlockUpperSameIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T1& outsider) : CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>(binningPolicy, categoryNeighbours, outsider, 1) {}
  CombinationsBlockUpperSameIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, const Ts&... tables) : CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>(binningPolicy, categoryNeighbours, outsider, 1, tables...)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }
  CombinationsBlockUpperSameIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, Ts&&... tables) : CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>(binningPolicy, categoryNeighbours, outsider, 1, std::forward<Ts>(tables)...)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setTables(const Ts&... tables)
  {
    CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>::setTables(tables...);
    setRanges();
  }
  void setTables(Ts&&... tables)
  {
    CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>::setTables(std::forward<Ts>(tables)...);
    setRanges();
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts);
    auto catBegin = this->mGroupedIndices.begin() + std::get<0>(this->mCurrentIndices);
    auto range = std::equal_range(catBegin, this->mGroupedIndices.end(), *catBegin, sameCategory);
    uint64_t offset = std::distance(this->mGroupedIndices.begin(), range.second);

    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = std::get<0>(this->mCurrentIndices);
      std::get<i.value>(this->mMaxOffset) = offset;
      std::get<i.value>(this->mCurrent).setCursor(range.first->index);
    });
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    for_<k - 1>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        uint64_t maxForWindow = std::get<0>(this->mCurrentIndices) + this->mSlidingWindowSize;

        // If we remain within the same sliding window
        if (curGroupedInd < maxForWindow && curGroupedInd < std::get<curInd>(this->mMaxOffset)) {
          std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].index);
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices);
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].index);
          });
          modify = false;
        }
      }
    });

    this->mIsNewWindow = modify;

    // First iterator processed separately
    if (modify) {
      std::get<0>(this->mCurrentIndices)++;
      uint64_t curGroupedInd = std::get<0>(this->mCurrentIndices);

      // If we remain within the same category - slide window
      if (curGroupedInd < std::get<0>(this->mMaxOffset)) {
        std::get<0>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].index);
        for_<k - 1>([&, this](auto j) {
          constexpr auto curJ = j.value + 1;
          std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices);
          uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
          std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].index);
        });
        modify = false;
      }
    }

    // No more combinations within this category - move to the next category, if possible
    if (modify && std::get<0>(this->mCurrentIndices) < this->mGroupedIndices.size()) {
      setRanges();
      return;
    }

    this->mIsEnd = modify;
  }
};

template <typename BP, typename T1, typename... Ts>
struct CombinationsBlockStrictlyUpperSameIndexPolicy : public CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...> {
  using CombinationType = typename CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>::CombinationType;

  CombinationsBlockStrictlyUpperSameIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T1& outsider) : CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>(binningPolicy, categoryNeighbours, outsider, sizeof...(Ts)) {}
  CombinationsBlockStrictlyUpperSameIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, const Ts&... tables) : CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>(binningPolicy, categoryNeighbours, outsider, sizeof...(Ts), tables...)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  CombinationsBlockStrictlyUpperSameIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, Ts&&... tables) : CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>(binningPolicy, categoryNeighbours, outsider, sizeof...(Ts), std::forward<Ts>(tables)...)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setTables(const Ts&... tables)
  {
    CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>::setTables(tables...);
    if (!this->mIsEnd) {
      setRanges();
    }
  }
  void setTables(Ts&&... tables)
  {
    CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>::setTables(std::forward<Ts>(tables)...);
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts);
    auto catBegin = this->mGroupedIndices.begin() + std::get<0>(this->mCurrentIndices);
    auto lastIt = std::upper_bound(catBegin, this->mGroupedIndices.end(), *catBegin, sameCategory);
    uint64_t lastOffset = std::distance(this->mGroupedIndices.begin(), lastIt);

    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = std::get<0>(this->mCurrentIndices) + i.value;
      std::get<i.value>(this->mCurrent).setCursor(this->mGroupedIndices[std::get<i.value>(this->mCurrentIndices)].index);
      std::get<i.value>(this->mMaxOffset) = lastOffset - k + i.value + 1;
    });
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    for_<k - 1>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        uint64_t maxForWindow = std::get<0>(this->mCurrentIndices) + this->mSlidingWindowSize - i.value;

        // If we remain within the same sliding window
        if (curGroupedInd < maxForWindow && curGroupedInd < std::get<curInd>(this->mMaxOffset)) {
          std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].index);
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices) + 1;
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].index);
          });
          modify = false;
        }
      }
    });

    this->mIsNewWindow = modify;

    // First iterator processed separately
    if (modify) {
      std::get<0>(this->mCurrentIndices)++;
      uint64_t curGroupedInd = std::get<0>(this->mCurrentIndices);

      // If we remain within the same category - slide window
      if (curGroupedInd < std::get<0>(this->mMaxOffset)) {
        std::get<0>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].index);
        for_<k - 1>([&, this](auto j) {
          constexpr auto curJ = j.value + 1;
          std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices) + 1;
          uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
          std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].index);
        });
        modify = false;
      }
    }

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
};

template <typename BP, typename T1, typename... Ts>
struct CombinationsBlockFullSameIndexPolicy : public CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...> {
  using CombinationType = typename CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>::CombinationType;

  CombinationsBlockFullSameIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T1& outsider) : CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>(binningPolicy, categoryNeighbours, outsider, 1), mCurrentlyFixed(0) {}
  CombinationsBlockFullSameIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, const Ts&... tables) : CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>(binningPolicy, categoryNeighbours, outsider, 1, tables...), mCurrentlyFixed(0)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }
  CombinationsBlockFullSameIndexPolicy(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, Ts&&... tables) : CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>(binningPolicy, categoryNeighbours, outsider, 1, std::forward<Ts>(tables)...), mCurrentlyFixed(0)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setTables(const Ts&... tables)
  {
    CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>::setTables(tables...);
    setRanges();
  }
  void setTables(Ts&&... tables)
  {
    CombinationsBlockSameIndexPolicyBase<BP, T1, Ts...>::setTables(std::forward<Ts>(tables)...);
    setRanges();
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts);
    auto catBegin = this->mGroupedIndices.begin() + std::get<0>(this->mCurrentIndices);
    auto range = std::equal_range(catBegin, this->mGroupedIndices.end(), *catBegin, sameCategory);
    this->mBeginIndex = std::get<0>(this->mCurrentIndices);
    uint64_t offset = std::distance(this->mGroupedIndices.begin(), range.second);

    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mMaxOffset) = offset;
      std::get<i.value>(this->mCurrentIndices) = this->mBeginIndex;
      std::get<i.value>(this->mCurrent).setCursor(range.first->index);
    });
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts);
    bool modify = true;
    for_<k>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        uint64_t windowOffset = curInd == this->mCurrentlyFixed ? 1 : this->mSlidingWindowSize;
        uint64_t maxForWindow = this->mBeginIndex + windowOffset;

        // If we remain within the same sliding window and fixed index
        if (curGroupedInd < maxForWindow && curGroupedInd < std::get<curInd>(this->mMaxOffset)) {
          std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].index);
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            if (curJ < this->mCurrentlyFixed) { // To assure no repetitions
              std::get<curJ>(this->mCurrentIndices) = this->mBeginIndex + 1;
            } else {
              std::get<curJ>(this->mCurrentIndices) = this->mBeginIndex;
            }
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].index);
          });
          modify = false;
        }
      }
    });

    // Currently fixed iterator processed separately
    if (modify) {
      // If we haven't finished with window starting element
      if (this->mCurrentlyFixed < k - 1 && this->mBeginIndex < std::get<0>(this->mMaxOffset) - 1) {
        this->mCurrentlyFixed++;
        for_<k>([&, this](auto s) {
          if (s.value < this->mCurrentlyFixed) { // To assure no repetitions
            std::get<s.value>(this->mCurrentIndices) = this->mBeginIndex + 1;
          } else {
            std::get<s.value>(this->mCurrentIndices) = this->mBeginIndex;
          }
          uint64_t curGroupedI = std::get<s.value>(this->mCurrentIndices);
          std::get<s.value>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedI].index);
        });
        modify = false;
      } else {
        this->mCurrentlyFixed = 0;
        this->mBeginIndex++;
        std::get<0>(this->mCurrentIndices) = this->mBeginIndex;

        // If we remain within the same category - slide window
        if (this->mBeginIndex < std::get<0>(this->mMaxOffset)) {
          uint64_t curGroupedInd = std::get<0>(this->mCurrentIndices);
          std::get<0>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].index);
          for_<k - 1>([&, this](auto j) {
            constexpr auto curJ = j.value + 1;
            std::get<curJ>(this->mCurrentIndices) = this->mBeginIndex;
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].index);
          });
          modify = false;
        } else {
          for_<k>([&, this](auto j) {
            std::get<j.value>(this->mCurrentIndices) = std::get<j.value>(this->mMaxOffset);
          });
        }
      }
    }

    // No more combinations within this category - move to the next category, if possible
    if (modify && std::get<0>(this->mCurrentIndices) < this->mGroupedIndices.size()) {
      setRanges();
      return;
    }

    this->mIsEnd = modify;
  }

  uint64_t mBeginIndex;
  uint64_t mCurrentlyFixed;
};

/// @return next combination of rows of tables.
/// FIXME: move to coroutines once we have C++20
template <typename P>
struct CombinationsGenerator {
  using CombinationType = typename P::CombinationType;

 public:
  struct CombinationsIterator : public P {
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

template <typename T2, typename... T2s>
constexpr bool isSameType()
{
  return std::conjunction_v<std::is_same<T2, T2s>...>;
}

template <typename BP, typename T1, typename... T2s>
auto selfCombinations(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, const T2s&... tables)
{
  static_assert(isSameType<T2s...>(), "Tables must have the same type for self combinations");
  return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2s...>>(CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2s...>(binningPolicy, categoryNeighbours, outsider, tables...));
}

template <typename BP, typename T1, typename T2>
auto selfPairCombinations(const BP& binningPolicy, int categoryNeighbours, const T1& outsider)
{
  return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2, T2>>(CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2, T2>(binningPolicy, categoryNeighbours, outsider));
}

template <typename BP, typename T1, typename T2>
auto selfPairCombinations(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, const T2& table)
{
  return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2, T2>>(CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2, T2>(binningPolicy, categoryNeighbours, outsider, table, table));
}

template <typename BP, typename T1, typename T2>
auto selfTripleCombinations(const BP& binningPolicy, int categoryNeighbours, const T1& outsider)
{
  return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2, T2, T2>>(CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2, T2, T2>(binningPolicy, categoryNeighbours, outsider));
}

template <typename BP, typename T1, typename T2>
auto selfTripleCombinations(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, const T2& table)
{
  return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2, T2, T2>>(CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2, T2, T2>(binningPolicy, categoryNeighbours, outsider, table, table, table));
}

template <typename BP, typename T1, typename... T2s>
auto combinations(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, const T2s&... tables)
{
  if constexpr (isSameType<T2s...>()) {
    return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2s...>>(CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, T2s...>(binningPolicy, categoryNeighbours, outsider, tables...));
  } else {
    return CombinationsGenerator<CombinationsBlockUpperIndexPolicy<BP, T1, T2s...>>(CombinationsBlockUpperIndexPolicy<BP, T1, T2s...>(binningPolicy, categoryNeighbours, outsider, tables...));
  }
}

template <typename BP, typename T1, typename... T2s>
auto combinations(const BP& binningPolicy, int categoryNeighbours, const T1& outsider, const o2::framework::expressions::Filter& filter, const T2s&... tables)
{
  if constexpr (isSameType<T2s...>()) {
    return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<BP, T1, Filtered<T2s>...>>(CombinationsBlockStrictlyUpperSameIndexPolicy(binningPolicy, categoryNeighbours, outsider, tables.select(filter)...));
  } else {
    return CombinationsGenerator<CombinationsBlockUpperIndexPolicy<BP, T1, Filtered<T2s>...>>(CombinationsBlockUpperIndexPolicy(binningPolicy, categoryNeighbours, outsider, tables.select(filter)...));
  }
}

template <typename... T2s>
auto combinations(const o2::framework::expressions::Filter& filter, const T2s&... tables)
{
  if constexpr (isSameType<T2s...>()) {
    return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<Filtered<T2s>...>>(CombinationsStrictlyUpperIndexPolicy(tables.select(filter)...));
  } else {
    return CombinationsGenerator<CombinationsUpperIndexPolicy<Filtered<T2s>...>>(CombinationsUpperIndexPolicy(tables.select(filter)...));
  }
}

// This shortened version cannot be used for Filtered
// (unless users create filtered tables themselves before policy creation)
template <template <typename...> typename P2, typename... T2s>
CombinationsGenerator<P2<T2s...>> combinations(const P2<T2s...>& policy)
{
  return CombinationsGenerator<P2<T2s...>>(policy);
}

template <template <typename...> typename P2, typename... T2s>
CombinationsGenerator<P2<Filtered<T2s>...>> combinations(P2<T2s...>&&, const o2::framework::expressions::Filter& filter, const T2s&... tables)
{
  return CombinationsGenerator<P2<Filtered<T2s>...>>(P2<Filtered<T2s>...>(tables.select(filter)...));
}

template <typename... T2s>
auto combinations(const T2s&... tables)
{
  if constexpr (isSameType<T2s...>()) {
    return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<T2s...>>(CombinationsStrictlyUpperIndexPolicy<T2s...>(tables...));
  } else {
    return CombinationsGenerator<CombinationsUpperIndexPolicy<T2s...>>(CombinationsUpperIndexPolicy<T2s...>(tables...));
  }
}

template <typename T2>
auto pairCombinations()
{
  return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<T2, T2>>(CombinationsStrictlyUpperIndexPolicy<T2, T2>());
}

template <typename T2>
auto pairCombinations(const T2& table)
{
  return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<T2, T2>>(CombinationsStrictlyUpperIndexPolicy(table, table));
}

template <typename T2>
auto tripleCombinations()
{
  return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<T2, T2, T2>>(CombinationsStrictlyUpperIndexPolicy<T2, T2, T2>());
}

template <typename T2>
auto tripleCombinations(const T2& table)
{
  return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<T2, T2, T2>>(CombinationsStrictlyUpperIndexPolicy(table, table, table));
}

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOAHELPERS_H_
