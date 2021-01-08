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
std::vector<std::pair<uint64_t, uint64_t>> doGroupTable(const std::shared_ptr<arrow::Table>& table, const std::string& categoryColumnName, int minCatSize, const T2& outsider)
{
  auto columnIndex = table->schema()->GetFieldIndex(categoryColumnName);
  auto chunkedArray = table->column(columnIndex);

  uint64_t ind = 0;
  std::vector<std::pair<uint64_t, uint64_t>> groupedIndices;
  for (uint64_t ci = 0; ci < chunkedArray->num_chunks(); ++ci) {
    auto chunk = chunkedArray->chunk(ci);
    // Note: assuming that the table is not empty
    T2 const* data = std::static_pointer_cast<ARRAY>(chunk)->raw_values();
    for (uint64_t ai = 0; ai < chunk->length(); ++ai) {
      if (data[ai] != outsider) {
        groupedIndices.emplace_back(data[ai], ind);
      }
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

template <typename T, typename T2>
auto groupTable(const T& table, const std::string& categoryColumnName, int minCatSize, const T2& outsider)
{
  auto arrowTable = table.asArrowTable();
  auto columnIndex = arrowTable->schema()->GetFieldIndex(categoryColumnName);
  auto dataType = arrowTable->column(columnIndex)->type();
  if (dataType->id() == arrow::Type::UINT64) {
    return doGroupTable<uint64_t, arrow::UInt64Array>(arrowTable, categoryColumnName, minCatSize, outsider);
  }
  if (dataType->id() == arrow::Type::INT64) {
    return doGroupTable<int64_t, arrow::Int64Array>(arrowTable, categoryColumnName, minCatSize, outsider);
  }
  if (dataType->id() == arrow::Type::UINT32) {
    return doGroupTable<uint32_t, arrow::UInt32Array>(arrowTable, categoryColumnName, minCatSize, outsider);
  }
  if (dataType->id() == arrow::Type::INT32) {
    return doGroupTable<int32_t, arrow::Int32Array>(arrowTable, categoryColumnName, minCatSize, outsider);
  }
  if (dataType->id() == arrow::Type::FLOAT) {
    return doGroupTable<float, arrow::FloatArray>(arrowTable, categoryColumnName, minCatSize, outsider);
  }
  // FIXME: Should we support other types as well?
  throw o2::framework::runtime_error("Combinations: category column must be of integral type");
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

  CombinationType mCurrent;
  IndicesType mMaxOffset; // one position past maximum acceptable position for each element of combination
  bool mIsEnd;            // whether there are any more tuples available
};

template <typename... Ts>
struct CombinationsUpperIndexPolicy : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;

  CombinationsUpperIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...) {}

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

  CombinationsFullIndexPolicy(const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...) {}

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
template <typename T, typename... Ts>
struct CombinationsBlockIndexPolicyBase : public CombinationsIndexPolicyBase<Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts)>::type;

  CombinationsBlockIndexPolicyBase(const std::string& categoryColumnName, int categoryNeighbours, const T& outsider, const Ts&... tables) : CombinationsIndexPolicyBase<Ts...>(tables...), mSlidingWindowSize(categoryNeighbours + 1)
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
    ((this->mGroupedIndices[tableIndex++] = groupTable(tables, categoryColumnName, 1, outsider)), ...);

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

  std::array<std::vector<std::pair<uint64_t, uint64_t>>, sizeof...(Ts)> mGroupedIndices;
  IndicesType mCurrentIndices;
  IndicesType mBeginIndices;
  uint64_t mSlidingWindowSize;
};

template <typename T, typename... Ts>
struct CombinationsBlockUpperIndexPolicy : public CombinationsBlockIndexPolicyBase<T, Ts...> {
  using CombinationType = typename CombinationsBlockIndexPolicyBase<T, Ts...>::CombinationType;

  CombinationsBlockUpperIndexPolicy(const std::string& categoryColumnName, int categoryNeighbours, const T& outsider, const Ts&... tables) : CombinationsBlockIndexPolicyBase<T, Ts...>(categoryColumnName, categoryNeighbours, outsider, tables...)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts);
    for_<k>([&, this](auto i) {
      auto catBegin = this->mGroupedIndices[i.value].begin() + std::get<i.value>(this->mCurrentIndices);
      auto range = std::equal_range(catBegin, this->mGroupedIndices[i.value].end(), *catBegin, sameCategory);
      std::get<i.value>(this->mBeginIndices) = std::distance(this->mGroupedIndices[i.value].begin(), range.first);
      std::get<i.value>(this->mCurrent).setCursor(range.first->second);
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
        std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curInd][curGroupedInd].second);
        uint64_t maxForWindow = std::get<curInd>(this->mBeginIndices) + this->mSlidingWindowSize;

        // If we remain within the same sliding window
        if (curGroupedInd < maxForWindow && curGroupedInd < std::get<curInd>(this->mMaxOffset)) {
          modify = false;
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            if (std::get<curJ - 1>(this->mCurrentIndices) < std::get<curJ>(this->mMaxOffset)) {
              std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices);
              uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
              std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].second);
            } else {
              modify = true;
            }
          });
        }
      }
    });

    // First iterator processed separately
    if (modify) {
      std::get<0>(this->mCurrentIndices)++;
      std::get<0>(this->mBeginIndices)++;
      uint64_t curGroupedInd = std::get<0>(this->mCurrentIndices);
      std::get<0>(this->mCurrent).setCursor(this->mGroupedIndices[0][curGroupedInd].second);

      // If we remain within the same category - slide window
      if (curGroupedInd < std::get<0>(this->mMaxOffset)) {
        modify = false;
        for_<k - 1>([&, this](auto j) {
          constexpr auto curJ = j.value + 1;
          std::get<curJ>(this->mBeginIndices)++;
          if (std::get<curJ>(this->mBeginIndices) < std::get<curJ>(this->mMaxOffset)) {
            std::get<curJ>(this->mCurrentIndices) = std::get<curJ>(this->mBeginIndices);
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].second);
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

template <typename T, typename... Ts>
struct CombinationsBlockFullIndexPolicy : public CombinationsBlockIndexPolicyBase<T, Ts...> {
  using CombinationType = typename CombinationsBlockIndexPolicyBase<T, Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts)>::type;

  CombinationsBlockFullIndexPolicy(const std::string& categoryColumnName, int categoryNeighbours, const T& outsider, const Ts&... tables) : CombinationsBlockIndexPolicyBase<T, Ts...>(categoryColumnName, categoryNeighbours, outsider, tables...), mCurrentlyFixed(0)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
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
        uint64_t windowOffset = curInd == this->mCurrentlyFixed ? 1 : this->mSlidingWindowSize;
        uint64_t maxForWindow = std::get<curInd>(this->mBeginIndices) + windowOffset;

        // If we remain within the same sliding window and fixed index
        if (curGroupedInd < maxForWindow && curGroupedInd < std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            if (curJ < this->mCurrentlyFixed) { // To assure no repetitions
              std::get<curJ>(this->mCurrentIndices) = std::get<curJ>(this->mBeginIndices) + 1;
            } else {
              std::get<curJ>(this->mCurrentIndices) = std::get<curJ>(this->mBeginIndices);
            }
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].second);
          });
          modify = false;
        }
      }
    });

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
          std::get<s.value>(this->mCurrent).setCursor(this->mGroupedIndices[s.value][curGroupedI].second);
        });
        modify = false;
      } else {
        this->mCurrentlyFixed = 0;
        std::get<0>(this->mBeginIndices)++;
        std::get<0>(this->mCurrentIndices) = std::get<0>(this->mBeginIndices);
        uint64_t curGroupedInd = std::get<0>(this->mCurrentIndices);
        std::get<0>(this->mCurrent).setCursor(this->mGroupedIndices[0][curGroupedInd].second);

        // If we remain within the same category - slide window
        if (std::get<0>(this->mBeginIndices) < std::get<0>(this->mMaxOffset)) {
          modify = false;
          for_<k - 1>([&, this](auto j) {
            constexpr auto curJ = j.value + 1;
            std::get<curJ>(this->mBeginIndices)++;
            if (std::get<curJ>(this->mBeginIndices) < std::get<curJ>(this->mMaxOffset)) {
              std::get<curJ>(this->mCurrentIndices) = std::get<curJ>(this->mBeginIndices);
              uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
              std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curJ][curGroupedJ].second);
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

template <typename T1, typename T, typename... Ts>
struct CombinationsBlockSameIndexPolicyBase : public CombinationsIndexPolicyBase<T, Ts...> {
  using CombinationType = typename CombinationsIndexPolicyBase<T, Ts...>::CombinationType;
  using IndicesType = typename NTupleType<uint64_t, sizeof...(Ts) + 1>::type;

  CombinationsBlockSameIndexPolicyBase(const std::string& categoryColumnName, int categoryNeighbours, const T1& outsider, int minWindowSize, const T& table, const Ts&... tables) : CombinationsIndexPolicyBase<T, Ts...>(table, tables...), mSlidingWindowSize(categoryNeighbours + 1)
  {
    constexpr auto k = sizeof...(Ts) + 1;
    // minWindowSize == 1 for upper and full, and k for strictly upper k-combination
    if (mSlidingWindowSize < minWindowSize) {
      this->mIsEnd = true;
      return;
    }

    this->mGroupedIndices = groupTable(table, categoryColumnName, minWindowSize, outsider);

    if (this->mGroupedIndices.size() == 0) {
      this->mIsEnd = true;
      return;
    }

    std::get<0>(this->mCurrentIndices) = 0;
  }

  std::vector<std::pair<uint64_t, uint64_t>> mGroupedIndices;
  IndicesType mCurrentIndices;
  uint64_t mSlidingWindowSize;
};

template <typename T1, typename T, typename... Ts>
struct CombinationsBlockUpperSameIndexPolicy : public CombinationsBlockSameIndexPolicyBase<T1, T, Ts...> {
  using CombinationType = typename CombinationsBlockSameIndexPolicyBase<T1, T, Ts...>::CombinationType;

  CombinationsBlockUpperSameIndexPolicy(const std::string& categoryColumnName, int categoryNeighbours, const T1& outsider, const T& table, const Ts&... tables) : CombinationsBlockSameIndexPolicyBase<T1, T, Ts...>(categoryColumnName, categoryNeighbours, outsider, 1, table, tables...)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    auto catBegin = this->mGroupedIndices.begin() + std::get<0>(this->mCurrentIndices);
    auto range = std::equal_range(catBegin, this->mGroupedIndices.end(), *catBegin, sameCategory);
    uint64_t offset = std::distance(this->mGroupedIndices.begin(), range.second);

    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = std::get<0>(this->mCurrentIndices);
      std::get<i.value>(this->mMaxOffset) = offset;
      std::get<i.value>(this->mCurrent).setCursor(range.first->second);
    });
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    bool modify = true;
    for_<k - 1>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].second);
        uint64_t maxForWindow = std::get<0>(this->mCurrentIndices) + this->mSlidingWindowSize;

        // If we remain within the same sliding window
        if (curGroupedInd < maxForWindow && curGroupedInd < std::get<curInd>(this->mMaxOffset)) {
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

    // First iterator processed separately
    if (modify) {
      std::get<0>(this->mCurrentIndices)++;
      uint64_t curGroupedInd = std::get<0>(this->mCurrentIndices);
      std::get<0>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].second);

      // If we remain within the same category - slide window
      if (curGroupedInd < std::get<0>(this->mMaxOffset)) {
        for_<k - 1>([&, this](auto j) {
          constexpr auto curJ = j.value + 1;
          std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices);
          uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
          std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].second);
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

template <typename T1, typename T, typename... Ts>
struct CombinationsBlockStrictlyUpperSameIndexPolicy : public CombinationsBlockSameIndexPolicyBase<T1, T, Ts...> {
  using CombinationType = typename CombinationsBlockSameIndexPolicyBase<T1, T, Ts...>::CombinationType;

  CombinationsBlockStrictlyUpperSameIndexPolicy(const std::string& categoryColumnName, int categoryNeighbours, const T1& outsider, const T& table, const Ts&... tables) : CombinationsBlockSameIndexPolicyBase<T1, T, Ts...>(categoryColumnName, categoryNeighbours, outsider, sizeof...(Ts) + 1, table, tables...)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    auto catBegin = this->mGroupedIndices.begin() + std::get<0>(this->mCurrentIndices);
    auto lastIt = std::upper_bound(catBegin, this->mGroupedIndices.end(), *catBegin, sameCategory);
    uint64_t lastOffset = std::distance(this->mGroupedIndices.begin(), lastIt);

    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mCurrentIndices) = std::get<0>(this->mCurrentIndices) + i.value;
      std::get<i.value>(this->mCurrent).setCursor(this->mGroupedIndices[std::get<i.value>(this->mCurrentIndices)].second);
      std::get<i.value>(this->mMaxOffset) = lastOffset - k + i.value + 1;
    });
  }

  void addOne()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    bool modify = true;
    for_<k - 1>([&, this](auto i) {
      if (modify) {
        constexpr auto curInd = k - i.value - 1;
        std::get<curInd>(this->mCurrentIndices)++;
        uint64_t curGroupedInd = std::get<curInd>(this->mCurrentIndices);
        std::get<curInd>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].second);
        uint64_t maxForWindow = std::get<0>(this->mCurrentIndices) + this->mSlidingWindowSize - i.value;

        // If we remain within the same sliding window
        if (curGroupedInd < maxForWindow && curGroupedInd < std::get<curInd>(this->mMaxOffset)) {
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

    // First iterator processed separately
    if (modify) {
      std::get<0>(this->mCurrentIndices)++;
      uint64_t curGroupedInd = std::get<0>(this->mCurrentIndices);
      std::get<0>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].second);

      // If we remain within the same category - slide window
      if (curGroupedInd < std::get<0>(this->mMaxOffset)) {
        for_<k - 1>([&, this](auto j) {
          constexpr auto curJ = j.value + 1;
          std::get<curJ>(this->mCurrentIndices) = std::get<curJ - 1>(this->mCurrentIndices) + 1;
          uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
          std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].second);
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

template <typename T1, typename T, typename... Ts>
struct CombinationsBlockFullSameIndexPolicy : public CombinationsBlockSameIndexPolicyBase<T1, T, Ts...> {
  using CombinationType = typename CombinationsBlockSameIndexPolicyBase<T1, T, Ts...>::CombinationType;

  CombinationsBlockFullSameIndexPolicy(const std::string& categoryColumnName, int categoryNeighbours, const T1& outsider, const T& table, const Ts&... tables) : CombinationsBlockSameIndexPolicyBase<T1, T, Ts...>(categoryColumnName, categoryNeighbours, outsider, 1, table, tables...), mCurrentlyFixed(0)
  {
    if (!this->mIsEnd) {
      setRanges();
    }
  }

  void setRanges()
  {
    constexpr auto k = sizeof...(Ts) + 1;
    auto catBegin = this->mGroupedIndices.begin() + std::get<0>(this->mCurrentIndices);
    auto range = std::equal_range(catBegin, this->mGroupedIndices.end(), *catBegin, sameCategory);
    this->mBeginIndex = std::get<0>(this->mCurrentIndices);
    uint64_t offset = std::distance(this->mGroupedIndices.begin(), range.second);

    for_<k>([&, this](auto i) {
      std::get<i.value>(this->mMaxOffset) = offset;
      std::get<i.value>(this->mCurrentIndices) = this->mBeginIndex;
      std::get<i.value>(this->mCurrent).setCursor(range.first->second);
    });
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
        uint64_t windowOffset = curInd == this->mCurrentlyFixed ? 1 : this->mSlidingWindowSize;
        uint64_t maxForWindow = this->mBeginIndex + windowOffset;

        // If we remain within the same sliding window and fixed index
        if (curGroupedInd < maxForWindow && curGroupedInd < std::get<curInd>(this->mMaxOffset)) {
          for_<i.value>([&, this](auto j) {
            constexpr auto curJ = k - i.value + j.value;
            if (curJ < this->mCurrentlyFixed) { // To assure no repetitions
              std::get<curJ>(this->mCurrentIndices) = this->mBeginIndex + 1;
            } else {
              std::get<curJ>(this->mCurrentIndices) = this->mBeginIndex;
            }
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].second);
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
          std::get<s.value>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedI].second);
        });
        modify = false;
      } else {
        this->mCurrentlyFixed = 0;
        this->mBeginIndex++;
        std::get<0>(this->mCurrentIndices) = this->mBeginIndex;
        uint64_t curGroupedInd = std::get<0>(this->mCurrentIndices);
        std::get<0>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedInd].second);

        // If we remain within the same category - slide window
        if (this->mBeginIndex < std::get<0>(this->mMaxOffset)) {
          for_<k - 1>([&, this](auto j) {
            constexpr auto curJ = j.value + 1;
            std::get<curJ>(this->mCurrentIndices) = this->mBeginIndex;
            uint64_t curGroupedJ = std::get<curJ>(this->mCurrentIndices);
            std::get<curJ>(this->mCurrent).setCursor(this->mGroupedIndices[curGroupedJ].second);
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

template <typename T1, typename T2, typename... T2s>
auto selfCombinations(const char* categoryColumnName, int categoryNeighbours, const T1& outsider, const T2& table, const T2s&... tables)
{
  static_assert(std::conjunction_v<std::is_same<T2, T2s>...>, "Tables must have the same type for self combinations");
  return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<T1, T2, T2s...>>(CombinationsBlockStrictlyUpperSameIndexPolicy(categoryColumnName, categoryNeighbours, outsider, table, tables...));
}

template <typename T1, typename T2>
auto selfPairCombinations(const char* categoryColumnName, int categoryNeighbours, const T1& outsider, const T2& table)
{
  return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<T1, T2, T2>>(CombinationsBlockStrictlyUpperSameIndexPolicy(categoryColumnName, categoryNeighbours, outsider, table, table));
}

template <typename T1, typename T2>
auto selfTripleCombinations(const char* categoryColumnName, int categoryNeighbours, const T1& outsider, const T2& table)
{
  return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<T1, T2, T2, T2>>(CombinationsBlockStrictlyUpperSameIndexPolicy(categoryColumnName, categoryNeighbours, outsider, table, table, table));
}

template <typename T1, typename T2, typename... T2s>
auto combinations(const char* categoryColumnName, int categoryNeighbours, const T1& outsider, const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<T1, T2, T2s...>>(CombinationsBlockStrictlyUpperSameIndexPolicy(categoryColumnName, categoryNeighbours, outsider, table, tables...));
  } else {
    return CombinationsGenerator<CombinationsBlockUpperIndexPolicy<T1, T2, T2s...>>(CombinationsBlockUpperIndexPolicy(categoryColumnName, categoryNeighbours, outsider, table, tables...));
  }
}

template <typename T1, typename T2, typename... T2s>
auto combinations(const char* categoryColumnName, int categoryNeighbours, const T1& outsider, const o2::framework::expressions::Filter& filter, const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsBlockStrictlyUpperSameIndexPolicy<T1, Filtered<T2>, Filtered<T2s>...>>(CombinationsBlockStrictlyUpperSameIndexPolicy(categoryColumnName, categoryNeighbours, outsider, Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  } else {
    return CombinationsGenerator<CombinationsBlockUpperIndexPolicy<T1, Filtered<T2>, Filtered<T2s>...>>(CombinationsBlockUpperIndexPolicy(categoryColumnName, categoryNeighbours, outsider, Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  }
}

template <typename T2, typename... T2s>
auto combinations(const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<T2, T2s...>>(CombinationsStrictlyUpperIndexPolicy(table, tables...));
  } else {
    return CombinationsGenerator<CombinationsUpperIndexPolicy<T2, T2s...>>(CombinationsUpperIndexPolicy(table, tables...));
  }
}

template <typename T2>
auto pairCombinations(const T2& table)
{
  return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<T2, T2>>(CombinationsStrictlyUpperIndexPolicy(table, table));
}

template <typename T2>
auto tripleCombinations(const T2& table)
{
  return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<T2, T2, T2>>(CombinationsStrictlyUpperIndexPolicy(table, table, table));
}

template <typename T2, typename... T2s>
auto combinations(const o2::framework::expressions::Filter& filter, const T2& table, const T2s&... tables)
{
  if constexpr (std::conjunction_v<std::is_same<T2, T2s>...>) {
    return CombinationsGenerator<CombinationsStrictlyUpperIndexPolicy<Filtered<T2>, Filtered<T2s>...>>(CombinationsStrictlyUpperIndexPolicy(Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
  } else {
    return CombinationsGenerator<CombinationsUpperIndexPolicy<Filtered<T2>, Filtered<T2s>...>>(CombinationsUpperIndexPolicy(Filtered<T2>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)}, Filtered<T2s>{{tables.asArrowTable()}, o2::framework::expressions::createSelection(tables.asArrowTable(), filter)}...));
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
