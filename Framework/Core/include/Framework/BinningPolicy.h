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

#include "Framework/ASoA.h"
#include "Framework/HistogramSpec.h" // only for VARIABLE_WIDTH

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

namespace o2::framework
{

namespace binning_helpers
{
inline void expandConstantBinning(std::vector<double> const& bins, std::vector<double>& expanded)
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
  /// Values outside the outermost edges of any axis are dropped: getBin() maps them
  /// to -1, which groupTable() treats as the outsider category. Giving them bins of
  /// their own used to be selectable per instance, but no analysis ever did, and for
  /// event mixing it is the wrong default anyway -- it pairs collisions that the
  /// vertex or centrality cut deliberately excluded. If it is ever genuinely wanted,
  /// add a BinningPolicyWithOverflow rather than a flag, so the two numberings cannot
  /// be confused at a call site.
  BinningPolicyBase(std::array<std::vector<double>, N> bins) : mBins(bins)
  {
    static_assert(N <= 3, "No default binning for more than 3 columns, you need to implement a binning class yourself");
    for (int i = 0; i < N; i++) {
      binning_helpers::expandConstantBinning(bins[i], mBins[i]);
    }
  }

  template <typename... Ts>
  int getBin(std::tuple<Ts...> const& data) const
  {
    static_assert(sizeof...(Ts) == N, "There must be the same number of binning axes and data values/columns");

    // mBins[d][0] is a dummy VARIABLE_WIDTH marker and mBins[d][1] is the lower edge,
    // so the first candidate edge is 2. A value below the lower edge, or above the
    // last one, puts the row outside the binning altogether.
    unsigned int i = 2, j = 2, k = 2;

    if (std::get<0>(data) < this->mBins[0][1]) {
      return -1;
    }
    if constexpr (N > 1) {
      if (std::get<1>(data) < this->mBins[1][1]) {
        return -1;
      }
    }
    if constexpr (N > 2) {
      if (std::get<2>(data) < this->mBins[2][1]) {
        return -1;
      }
    }

    for (; i < this->mBins[0].size(); i++) {
      if (std::get<0>(data) < this->mBins[0][i]) {
        break;
      }
    }
    if (i == this->mBins[0].size()) {
      return -1;
    }
    if constexpr (N > 1) {
      for (; j < this->mBins[1].size(); j++) {
        if (std::get<1>(data) < this->mBins[1][j]) {
          break;
        }
      }
      if (j == this->mBins[1].size()) {
        return -1;
      }
    }
    if constexpr (N > 2) {
      for (; k < this->mBins[2].size(); k++) {
        if (std::get<2>(data) < this->mBins[2][k]) {
          break;
        }
      }
      if (k == this->mBins[2].size()) {
        return -1;
      }
    }

    return getBinAt(i, j, k);
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

  // Note: Overflow / underflow bin -1 is not included
  int getAllBinsCount() const
  {
    if constexpr (N == 1) {
      return getXBinsCount();
    }
    if constexpr (N == 2) {
      return getXBinsCount() * getYBinsCount();
    }
    if constexpr (N == 3) {
      return getXBinsCount() * getYBinsCount() * getZBinsCount();
    }
    return -1;
  }

  std::array<std::vector<double>, N> mBins;

 private:
  // Two are subtracted: one for the dummy VARIABLE_WIDTH at mBins[d][0], one because
  // values below the first edge are dropped rather than given a bin of their own.
  int getBinAt(unsigned int iRaw, unsigned int jRaw, unsigned int kRaw) const
  {
    unsigned int i = iRaw - 2;
    unsigned int j = jRaw - 2;
    unsigned int k = kRaw - 2;
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

  // Note: Overflow / underflow bin -1 is not included
  int getBinsCount(std::vector<double> const& bins) const
  {
    return bins.size() - 2;
  }
};

template <typename, typename...>
struct FlexibleBinningPolicy;

template <typename... Ts, typename... Ls>
struct FlexibleBinningPolicy<std::tuple<Ls...>, Ts...> : BinningPolicyBase<sizeof...(Ts)> {
  FlexibleBinningPolicy(std::tuple<Ls...> const& lambdaPtrs, std::array<std::vector<double>, sizeof...(Ts)> bins) : BinningPolicyBase<sizeof...(Ts)>(bins), mBinningFunctions{lambdaPtrs}
  {
  }

  template <typename T, typename T2>
  auto getBinningValue(T& rowIterator, uint64_t globalIndex = -1) const
  {
    if (globalIndex != -1) {
      rowIterator.setCursor(globalIndex);
    }
    if constexpr (has_type<T2>(pack<Ls...>{})) {
      return std::get<T2>(mBinningFunctions)(rowIterator);
    } else {
      return soa::row_helpers::getColumnValue<typename T2::type, T, T2>(rowIterator);
    }
  }

  template <typename T>
  auto getBinningValues(T& rowIterator, uint64_t globalIndex = -1) const
  {
    return std::make_tuple(getBinningValue<T, Ts>(rowIterator, globalIndex)...);
  }

  template <typename T>
  auto getBinningValues(typename T::iterator rowIterator, T& table, uint64_t globalIndex = -1) const
  {
    return getBinningValues(rowIterator, globalIndex);
  }

  template <typename... T2s>
  int getBin(std::tuple<T2s...> const& data) const
  {
    return BinningPolicyBase<sizeof...(Ts)>::template getBin<T2s...>(data);
  }

  using persistent_columns_t = framework::selected_pack<o2::soa::is_persistent_column_t, Ts...>;

 private:
  std::tuple<Ls...> mBinningFunctions;
};

template <typename... Ts>
struct ColumnBinningPolicy : BinningPolicyBase<sizeof...(Ts)> {
  ColumnBinningPolicy(std::array<std::vector<double>, sizeof...(Ts)> bins) : BinningPolicyBase<sizeof...(Ts)>(bins)
  {
  }

  template <typename T>
  auto getBinningValues(T& rowIterator, uint64_t globalIndex = -1) const
  {
    if (globalIndex != -1) {
      rowIterator.setCursor(globalIndex);
    }
    return std::make_tuple(soa::row_helpers::getColumnValue<typename Ts::type, T, Ts>(rowIterator)...);
  }

  template <typename T>
  auto getBinningValues(typename T::iterator rowIterator, T& table, uint64_t globalIndex = -1) const
  {
    return getBinningValues(rowIterator, globalIndex);
  }

  int getBin(std::tuple<typename Ts::type...> const& data) const
  {
    return BinningPolicyBase<sizeof...(Ts)>::template getBin<typename Ts::type...>(data);
  }

  using persistent_columns_t = framework::selected_pack<o2::soa::is_persistent_column_t, Ts...>;
};

template <typename C>
struct NoBinningPolicy {
  // Just take the bin number from the column data
  NoBinningPolicy() = default;

  template <typename T>
  auto getBinningValues(T& rowIterator, uint64_t globalIndex = -1) const
  {
    if (globalIndex != -1) {
      rowIterator.setCursor(globalIndex);
    }
    return std::make_tuple(soa::row_helpers::getColumnValue<typename C::type, T, C>(rowIterator));
  }

  template <typename T>
  auto getBinningValues(typename T::iterator rowIterator, T& table, uint64_t globalIndex = -1) const
  {
    return getBinningValues(rowIterator, globalIndex);
  }

  int getBin(std::tuple<typename C::type> const& data) const
  {
    return std::get<0>(data);
  }

  using persistent_columns_t = framework::selected_pack<o2::soa::is_persistent_column_t, C>;
};

} // namespace o2::framework
#endif // FRAMEWORK_BINNINGPOLICY_H_
