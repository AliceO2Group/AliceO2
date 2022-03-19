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
#include "Framework/ASoAHelpers.h"
#include "Framework/Pack.h"
#include "Framework/ArrowTypes.h"
#include <optional>

namespace o2::framework
{

namespace binning_helpers
{
template <typename... Cs>
std::array<arrow::ChunkedArray*, sizeof...(Cs)> getArrowColumns(arrow::Table* table, pack<Cs...>)
{
  static_assert(std::conjunction_v<typename Cs::persistent...>, "BinningPolicy: only persistent columns accepted (not dynamic and not index ones");
  return std::array<arrow::ChunkedArray*, sizeof...(Cs)>{o2::soa::getIndexFromLabel(table, Cs::columnLabel())...};
}

template <typename... Cs>
std::array<std::shared_ptr<arrow::Array>, sizeof...(Cs)> getChunks(arrow::Table* table, pack<Cs...>, uint64_t ci)
{
  static_assert(std::conjunction_v<typename Cs::persistent...>, "BinningPolicy: only persistent columns accepted (not dynamic and not index ones");
  return std::array<std::shared_ptr<arrow::Array>, sizeof...(Cs)>{o2::soa::getIndexFromLabel(table, Cs::columnLabel())->chunk(ci)...};
}

template <typename... Cs>
std::tuple<typename Cs::type...> getRowData(arrow::Table* table, pack<Cs...>, uint64_t ci, uint64_t ai)
{
  static_assert(std::conjunction_v<typename Cs::persistent...>, "BinningPolicy: only persistent columns accepted (not dynamic and not index ones");
  return std::make_tuple(std::static_pointer_cast<o2::soa::arrow_array_for_t<typename Cs::type>>(o2::soa::getIndexFromLabel(table, Cs::columnLabel())->chunk(ci))->raw_values()[ai]...);
}
} // namespace binning_helpers

template <typename C, typename... Cs>
struct BinningPolicy {
  BinningPolicy(std::array<std::vector<double>, sizeof...(Cs) + 1> bins, bool ignoreOverflows = true) : mBins(bins), mIgnoreOverflows(ignoreOverflows)
  {
    static_assert(sizeof...(Cs) < 3, "No default binning for more than 3 columns, you need to implement a binning class yourself");
    for (int i = 0; i < sizeof...(Cs) + 1; i++) {
      expandConstantBinning(bins[i], i);
    }
  }

  int getBin(std::tuple<typename C::type, typename Cs::type...> const& data) const
  {
    if (this->mIgnoreOverflows) {
      // underflow
      if (std::get<0>(data) < this->mBins[0][1]) { // xBins[0] is a dummy VARIABLE_WIDTH
        return -1;
      }
      if constexpr (sizeof...(Cs) > 0) {
        if (std::get<1>(data) < this->mBins[1][1]) { // this->mBins[1][0] is a dummy VARIABLE_WIDTH
          return -1;
        }
      }
      if constexpr (sizeof...(Cs) > 1) {
        if (std::get<2>(data) < this->mBins[2][1]) { // this->mBins[2][0] is a dummy VARIABLE_WIDTH
          return -1;
        }
      }
    }

    unsigned int i = 2, j = 2, k = 2;
    for (; i < this->mBins[0].size(); i++) {
      if (std::get<0>(data) < this->mBins[0][i]) {

        if constexpr (sizeof...(Cs) > 0) {
          for (; j < this->mBins[1].size(); j++) {
            if (std::get<1>(data) < this->mBins[1][j]) {

              if constexpr (sizeof...(Cs) > 1) {
                for (; k < this->mBins[2].size(); k++) {
                  if (std::get<2>(data) < this->mBins[2][k]) {
                    return getBinAt(i, j, k);
                  }
                }
                if (this->mIgnoreOverflows) {
                  return -1;
                }
              }

              // overflow for this->mBins[2] only
              return getBinAt(i, j, k);
            }
          }

          if (this->mIgnoreOverflows) {
            return -1;
          }

          // overflow for this->mBins[1] only
          if constexpr (sizeof...(Cs) > 1) {
            for (k = 2; k < this->mBins[2].size(); k++) {
              if (std::get<2>(data) < this->mBins[2][k]) {
                return getBinAt(i, j, k);
              }
            }
          }
        }

        // overflow for this->mBins[2] and this->mBins[1]
        return getBinAt(i, j, k);
      }
    }

    if (this->mIgnoreOverflows) {
      // overflow
      return -1;
    }

    // overflow for this->mBins[0] only
    if constexpr (sizeof...(Cs) > 0) {
      for (j = 2; j < this->mBins[1].size(); j++) {
        if (std::get<1>(data) < this->mBins[1][j]) {

          if constexpr (sizeof...(Cs) > 1) {
            for (k = 2; k < this->mBins[2].size(); k++) {
              if (std::get<2>(data) < this->mBins[2][k]) {
                return getBinAt(i, j, k);
              }
            }
          }

          // overflow for this->mBins[0] and this->mBins[2]
          return getBinAt(i, j, k);
        }
      }
    }

    // overflow for this->mBins[0] and this->mBins[1]
    if constexpr (sizeof...(Cs) > 1) {
      for (k = 2; k < this->mBins[2].size(); k++) {
        if (std::get<2>(data) < this->mBins[2][k]) {
          return getBinAt(i, j, k);
        }
      }
    }

    // overflow for all bins
    return getBinAt(i, j, k);
  }

  pack<C, Cs...> getColumns() const { return pack<C, Cs...>{}; }

 private:
  int getBinAt(unsigned int i, unsigned int j, unsigned int k) const
  {
    if constexpr (sizeof...(Cs) == 0) {
      return i - 1;
    } else if constexpr (sizeof...(Cs) == 1) {
      return (i - 1) + (j - 1) * this->mBins[0].size();
    } else if constexpr (sizeof...(Cs) == 2) {
      return (i - 1) + (j - 1) * this->mBins[0].size() + (k - 1) * (this->mBins[0].size() + this->mBins[1].size());
    } else {
      return -1;
    }
  }

  void expandConstantBinning(std::vector<double> const& bins, int ind)
  {
    if (bins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(bins[0]);
      this->mBins[ind].clear();
      this->mBins[ind].resize(nBins + 2);
      this->mBins[ind][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[ind]) + 1, std::end(this->mBins[ind]), bins[2] - bins[1] / nBins);
    }
  }

  std::array<std::vector<double>, sizeof...(Cs) + 1> mBins;
  bool mIgnoreOverflows;
};

template <typename C>
struct NoBinningPolicy {
  // Just take the bin number from the column data
  NoBinningPolicy() = default;

  int getBin(std::tuple<typename C::type> const& data) const
  {
    return std::get<0>(data);
  }

  pack<C> getColumns() const { return pack<C>{}; }
};

} // namespace o2::framework
#endif // FRAMEWORK_BINNINGPOLICY_H_
