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

#include <array>
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
  BinningPolicyBase(std::array<std::vector<double>, N> bins, bool ignoreOverflows = true) : mBins(bins), mIgnoreOverflows(ignoreOverflows)
  {
    static_assert(N >= 1, "A binning policy needs at least one axis");
    for (std::size_t i = 0; i < N; i++) {
      binning_helpers::expandConstantBinning(bins[i], mBins[i]);
    }
  }

  /// Bins are numbered in row major order, the first axis varying fastest:
  ///   bin = sum_d binOnAxis(d) * prod_{e < d} binsCount(e)
  /// For one, two and three axes this is the numbering this class has always
  /// produced. Folding over the axes only removes the ceiling that came from
  /// writing the nest out by hand; it is not a new numbering scheme.
  ///
  /// The index is used by groupTable() as an equality key and as a total order, so
  /// all that is required of it is injectivity. Bins that are adjacent in parameter
  /// space are not adjacent in index, and nothing relies on them being so.
  template <typename... Ts>
  int getBin(std::tuple<Ts...> const& data) const
  {
    static_assert(sizeof...(Ts) == N, "There must be the same number of binning axes and data values/columns");

    // Fold the heterogeneous values into one array, so that the search below is a
    // plain loop over the axes instead of a nest whose depth has to be spelled out.
    // Every value was already compared against a double bin edge before, so going
    // through double here changes nothing.
    std::array<double, N> const values = std::apply(
      [](auto const&... value) { return std::array<double, N>{static_cast<double>(value)...}; }, data);

    int bin = 0;
    int stride = 1;
    for (std::size_t axis = 0; axis < N; axis++) {
      unsigned int edge = 0;
      if (!findEdge(axis, values[axis], edge)) {
        return -1; // outside this axis, and outside values are dropped
      }
      bin += (static_cast<int>(edge) - 1 - getOverflowShift()) * stride;
      stride *= getBinsCount(mBins[axis]);
    }
    return bin;
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
  int getBinsCountForAxis(std::size_t axis) const
  {
    return getBinsCount(mBins[axis]);
  }

  // Note: Overflow / underflow bin -1 is not included
  int getAllBinsCount() const
  {
    int count = 1;
    for (std::size_t axis = 0; axis < N; axis++) {
      count *= getBinsCount(mBins[axis]);
    }
    return count;
  }

  std::array<std::vector<double>, N> mBins;
  bool mIgnoreOverflows;

 private:
  /// Index of the first edge strictly above the value. mBins[axis][0] is a dummy
  /// VARIABLE_WIDTH marker and mBins[axis][1] is the lower edge, so the first
  /// candidate is 2 when under- and overflows are dropped, and 1 when they are kept
  /// and get bins of their own. Returns false when the value falls outside the axis
  /// and outside values are being dropped.
  bool findEdge(std::size_t axis, double value, unsigned int& edge) const
  {
    auto const& edges = mBins[axis];
    if (mIgnoreOverflows && value < edges[1]) {
      return false; // underflow
    }
    for (unsigned int i = mIgnoreOverflows ? 2 : 1; i < edges.size(); i++) {
      if (value < edges[i]) {
        edge = i;
        return true;
      }
    }
    if (mIgnoreOverflows) {
      return false; // overflow
    }
    edge = static_cast<unsigned int>(edges.size());
    return true;
  }

  // We substract 1 to account for VARIABLE_WIDTH in the bins vector
  // We substract second 1 if we omit values below minima (underflow, mapped to -1)
  // Otherwise we add 1 and we get the number of bins including those below and over the outer edges
  int getOverflowShift() const
  {
    return mIgnoreOverflows ? 1 : -1;
  }

  // Note: Overflow / underflow bin -1 is not included
  int getBinsCount(std::vector<double> const& bins) const
  {
    return bins.size() - 1 - getOverflowShift();
  }
};

template <typename, typename...>
struct FlexibleBinningPolicy;

template <typename... Ts, typename... Ls>
struct FlexibleBinningPolicy<std::tuple<Ls...>, Ts...> : BinningPolicyBase<sizeof...(Ts)> {
  FlexibleBinningPolicy(std::tuple<Ls...> const& lambdaPtrs, std::array<std::vector<double>, sizeof...(Ts)> bins, bool ignoreOverflows = true) : BinningPolicyBase<sizeof...(Ts)>(bins, ignoreOverflows), mBinningFunctions{lambdaPtrs}
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
  ColumnBinningPolicy(std::array<std::vector<double>, sizeof...(Ts)> bins, bool ignoreOverflows = true) : BinningPolicyBase<sizeof...(Ts)>(bins, ignoreOverflows)
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
