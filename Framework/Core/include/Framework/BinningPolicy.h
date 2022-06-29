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

#ifndef FRAMEWORK_BINNINGPOLICY_H
#define FRAMEWORK_BINNINGPOLICY_H

#include "Framework/HistogramSpec.h" // only for VARIABLE_WIDTH
#include "Framework/Pack.h"
#include "Framework/ArrowTypes.h"
#include <optional>

namespace o2::framework
{

namespace binning_helpers
{
void expandConstantBinning(std::vector<double> const& bins, std::vector<double>& expanded)
{
  if (bins[0] != VARIABLE_WIDTH) {
    int nBins = static_cast<int>(bins[0]);
    expanded.clear();
    expanded.resize(nBins + 2);
    expanded[0] = VARIABLE_WIDTH;
    for (int i = 0; i <= nBins; i++) {
      expanded[i + 1] = bins[1] + i * (bins[2] - bins[1]) / nBins;
    }
  }
}
} // namespace binning_helpers

template <std::size_t N>
struct BinningPolicyBase {
  BinningPolicyBase(std::array<std::vector<double>, N> bins, bool ignoreOverflows = true) : mBins(bins), mIgnoreOverflows(ignoreOverflows)
  {
    static_assert(N <= 3, "No default binning for more than 3 columns, you need to implement a binning class yourself");
    for (int i = 0; i < N; i++) {
      binning_helpers::expandConstantBinning(bins[i], mBins[i]);
    }
  }

  // Note: Overflow / underflow bin -1 is not included
  int getAllBinsCount() const
  {
    if constexpr (N == 1) {
      return getBinsCount(mBins[0]);
    }
    if constexpr (N == 2) {
      return getBinsCount(mBins[0]) * getBinsCount(mBins[1]);
    }
    if constexpr (N == 2) {
      return getBinsCount(mBins[0]) * getBinsCount(mBins[1]) * getBinsCount(mBins[2]);
    }
    return -1;
  }

  // Note: Overflow / underflow bin -1 is not included
  int getXBinsCount() const
  {
    return getBinsCount(mBins[0]);
  }

  // Note: Overflow / underflow bin -1 is not included
  int getYBinsCount() const
  {
    if constexpr (N == 1) {
      return 0;
    }
    return getBinsCount(mBins[1]);
  }

  // Note: Overflow / underflow bin -1 is not included
  int getZBinsCount() const
  {
    if constexpr (N < 3) {
      return 0;
    }
    return getBinsCount(mBins[2]);
  }

  template <typename... Ts>
  int getBin(std::tuple<Ts...> const& data) const
  {
    static_assert(sizeof...(Ts) == N, "There must be the same number of binning axes and data values/columns");

    unsigned int i = 2, j = 2, k = 2;
    if (mIgnoreOverflows) {
      // underflow
      if (std::get<0>(data) < mBins[0][1]) { // mBins[0][0] is a dummy VARIABLE_WIDTH
        return -1;
      }
      if constexpr (N > 1) {
        if (std::get<1>(data) < mBins[1][1]) { // mBins[1][0] is a dummy VARIABLE_WIDTH
          return -1;
        }
      }
      if constexpr (N > 2) {
        if (std::get<2>(data) < mBins[2][1]) { // mBins[2][0] is a dummy VARIABLE_WIDTH
          return -1;
        }
      }
    } else {
      i = 1;
      j = 1;
      k = 1;
    }

    for (; i < mBins[0].size(); i++) {
      if (std::get<0>(data) < mBins[0][i]) {

        if constexpr (N > 1) {
          for (; j < mBins[1].size(); j++) {
            if (std::get<1>(data) < mBins[1][j]) {

              if constexpr (N > 2) {
                for (; k < mBins[2].size(); k++) {
                  if (std::get<2>(data) < mBins[2][k]) {
                    return getBinAt(i, j, k);
                  }
                }
                if (mIgnoreOverflows) {
                  return -1;
                }
              }

              // overflow for mBins[2] only
              return getBinAt(i, j, k);
            }
          }

          if (mIgnoreOverflows) {
            return -1;
          }

          // overflow for mBins[1] only
          if constexpr (N > 2) {
            for (k = 2; k < mBins[2].size(); k++) {
              if (std::get<2>(data) < mBins[2][k]) {
                return getBinAt(i, j, k);
              }
            }
          }
        }

        // overflow for mBins[2] and mBins[1]
        return getBinAt(i, j, k);
      }
    }

    if (mIgnoreOverflows) {
      // overflow
      return -1;
    }

    // overflow for mBins[0] only
    if constexpr (N > 1) {
      for (j = 2; j < mBins[1].size(); j++) {
        if (std::get<1>(data) < mBins[1][j]) {

          if constexpr (N > 2) {
            for (k = 2; k < mBins[2].size(); k++) {
              if (std::get<2>(data) < mBins[2][k]) {
                return getBinAt(i, j, k);
              }
            }
          }

          // overflow for mBins[0] and mBins[2]
          return getBinAt(i, j, k);
        }
      }
    }

    // overflow for mBins[0] and mBins[1]
    if constexpr (N > 2) {
      for (k = 2; k < mBins[2].size(); k++) {
        if (std::get<2>(data) < mBins[2][k]) {
          return getBinAt(i, j, k);
        }
      }
    }

    // overflow for all bins
    return getBinAt(i, j, k);
  }

  std::array<std::vector<double>, N> mBins;
  bool mIgnoreOverflows;

 private:
  // We substract 1 to account for VARIABLE_WIDTH in the bins vector
  // We substract second 1 if we omit values below minima (underflow, mapped to -1)
  // Otherwise we add 1 and we get the number of bins including those below and over the outer edges
  int getBinAt(unsigned int iRaw, unsigned int jRaw, unsigned int kRaw) const
  {
    int shiftBinsWithoutOverflow = getOverflowShift();
    unsigned int i = iRaw - 1 + shiftBinsWithoutOverflow;
    unsigned int j = jRaw - 1 + shiftBinsWithoutOverflow;
    unsigned int k = kRaw - 1 + shiftBinsWithoutOverflow;
    auto xBinsCount = getXBinsCount();
    if constexpr (N == 1) {
      return i;
    } else if constexpr (N == 2) {
      return i + j * xBinsCount;
    } else if constexpr (N == 3) {
      return i + j * xBinsCount + k * xBinsCount * getYBinsCount();
    } else {
      return -1;
    }
  }

  int getOverflowShift() const
  {
    return mIgnoreOverflows ? -1 : 1;
  }

  // Note: Overflow / underflow bin -1 is not included
  int getBinsCount(std::vector<double> const& bins) const
  {
    return bins.size() - 1 + getOverflowShift();
  }
};

template <typename, typename...>
struct BinningPolicy;

template <typename... Ts, typename... Ls>
struct BinningPolicy<std::tuple<Ls...>, Ts...> : BinningPolicyBase<sizeof...(Ts)> {
  BinningPolicy(std::tuple<Ls...> const& lambdaPtrs, std::array<std::vector<double>, sizeof...(Ts)> bins, bool ignoreOverflows = true) : BinningPolicyBase<sizeof...(Ts)>(bins, ignoreOverflows), mBinningFunctions{lambdaPtrs}
  {
  }

  template <typename T, typename T2>
  auto getBinningValue(T& rowIterator, arrow::Table* table, uint64_t globalIndex = -1, uint64_t ci = -1, uint64_t ai = -1) const
  {
    using decayed = std::decay_t<T2>;
    if (globalIndex == -1) {
      globalIndex = *(std::get<0>(rowIterator.getIndices()));
    }

    if constexpr (has_type_v<T2, pack<Ls...>>) {
      rowIterator.setCursor(globalIndex);
      return std::get<T2>(mBinningFunctions)(rowIterator);
    } else {
      if (ci == -1 && ai == -1) {
        auto colIterator = static_cast<decayed>(rowIterator).mColumnIterator;
        ci = colIterator.mCurrentChunk;
        ai = *(colIterator.mCurrentPos) - colIterator.mFirstIndex;
      }
      return soa::row_helpers::getSingleRowData<T, T2>(table, rowIterator, ci, ai, globalIndex);
    }
  }

  template <typename T>
  auto getBinningValues(T& rowIterator, arrow::Table* table, uint64_t globalIndex = -1, uint64_t ci = -1, uint64_t ai = -1) const
  {
    return std::make_tuple(getBinningValue<T, Ts>(rowIterator, table, globalIndex, ci, ai)...);
  }

  template <typename T>
  auto getBinningValues(typename T::iterator rowIterator, T& table, uint64_t globalIndex = -1, uint64_t ci = -1, uint64_t ai = -1) const
  {
    return getBinningValues(rowIterator, table.asArrowTable().get(), globalIndex, ci, ai);
  }

  template <typename... T2s>
  int getBin(std::tuple<T2s...> const& data) const
  {
    return BinningPolicyBase<sizeof...(Ts)>::template getBin<T2s...>(data);
  }

  using persistent_columns_t = framework::selected_pack<o2::soa::is_persistent_t, Ts...>;

 private:
  std::tuple<Ls...> mBinningFunctions;
};

template <typename... Ts>
struct ColumnBinningPolicy : BinningPolicyBase<sizeof...(Ts)> {
  ColumnBinningPolicy(std::array<std::vector<double>, sizeof...(Ts)> bins, bool ignoreOverflows = true) : BinningPolicyBase<sizeof...(Ts)>(bins, ignoreOverflows)
  {
  }

  template <typename T>
  auto getBinningValues(T& rowIterator, arrow::Table* table, uint64_t ci, uint64_t ai, uint64_t globalIndex) const
  {
    return std::make_tuple(soa::row_helpers::getSingleRowData<T, Ts>(table, rowIterator, ci, ai, globalIndex)...);
  }

  int getBin(std::tuple<typename Ts::type...> const& data) const
  {
    return BinningPolicyBase<sizeof...(Ts)>::template getBin<typename Ts::type...>(data);
  }

  using persistent_columns_t = framework::selected_pack<o2::soa::is_persistent_t, Ts...>;
};

template <typename C>
struct NoBinningPolicy {
  // Just take the bin number from the column data
  NoBinningPolicy() = default;

  template <typename T>
  auto getBinningValues(T& rowIterator, arrow::Table* table, uint64_t ci, uint64_t ai, uint64_t globalIndex) const
  {
    return std::make_tuple(soa::row_helpers::getSingleRowData<T, C>(table, rowIterator, ci, ai, globalIndex));
  }

  int getBin(std::tuple<typename C::type> const& data) const
  {
    return std::get<0>(data);
  }

  using persistent_columns_t = framework::selected_pack<o2::soa::is_persistent_t, C>;
};

} // namespace o2::framework
#endif // FRAMEWORK_BINNINGPOLICY_H_
