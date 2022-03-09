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

template <typename... Cs>
struct BinningPolicyBase {
  BinningPolicyBase() = default;
  BinningPolicyBase(std::array<std::vector<double>, sizeof...(Cs)> bins, bool ignoreOverflows = true) : mBins(bins), mIgnoreOverflows(ignoreOverflows) {}

  int getBin(std::tuple<typename Cs::type...> const&) const
  {
    return -1;
  }

  std::array<arrow::ChunkedArray*, sizeof...(Cs)> getArrowColumns(arrow::Table* table) const
  {
    // TODO: Do the columns need to be persistent (i.e., not dynamic, not index)?
    return std::array<arrow::ChunkedArray*, sizeof...(Cs)>{o2::soa::getIndexFromLabel(table, Cs::columnLabel())...};
  }

  std::array<std::shared_ptr<arrow::Array>, sizeof...(Cs)> getChunks(arrow::Table* table, uint64_t ci) const
  {
    return std::array<std::shared_ptr<arrow::Array>, sizeof...(Cs)>{o2::soa::getIndexFromLabel(table, Cs::columnLabel())->chunk(ci)...};
  }

  // FIXME: Rather not needed
  std::tuple<typename Cs::type const*...> getChunkData(arrow::Table* table, uint64_t ci) const
  {
    return std::make_tuple(std::static_pointer_cast<o2::soa::arrow_array_for_t<typename Cs::type>>(o2::soa::getIndexFromLabel(table, Cs::columnLabel())->chunk(ci))->raw_values()...);
  }

  std::tuple<typename Cs::type...> getRowData(arrow::Table* table, uint64_t ci, uint64_t ai) const
  {
    return std::make_tuple(std::static_pointer_cast<o2::soa::arrow_array_for_t<typename Cs::type>>(o2::soa::getIndexFromLabel(table, Cs::columnLabel())->chunk(ci))->raw_values()[ai]...);
  }

  std::array<std::vector<double>, sizeof...(Cs)> mBins;
  bool mIgnoreOverflows;
  static constexpr int mColumnsCount = sizeof...(Cs);
};

template <typename C>
struct NoBinningPolicy : public BinningPolicyBase<C> {
  // Just take the bin number from the column data
  NoBinningPolicy() : BinningPolicyBase<C>() {}

  int getBin(std::tuple<typename C::type> const& data) const
  {
    return std::get<0>(data);
  }

  using BinningPolicyBase<C>::getArrowColumns;
  using BinningPolicyBase<C>::getChunks;
  using BinningPolicyBase<C>::getChunkData;
  using BinningPolicyBase<C>::getRowData;
};

template <typename C>
struct SingleBinningPolicy : public BinningPolicyBase<C> {
  SingleBinningPolicy(std::vector<double>& xBins, bool ignoreOverflows = true) : BinningPolicyBase<C>({xBins}, ignoreOverflows)
  {
    expandConstantBinning(xBins);
  }
  SingleBinningPolicy(int xNBins, typename C::type xBinMin, typename C::type xBinMax, bool ignoreOverflows = true) : SingleBinningPolicy<C>({xNBins, xBinMin, xBinMax}, ignoreOverflows)
  {
  }

  int getBin(std::tuple<typename C::type> const& data) const
  {
    auto val1 = std::get<0>(data);
    auto xBins = this->mBins[0];
    if (this->mIgnoreOverflows) {
      // underflow
      if (val1 < xBins[1]) { // xBins[0] is a dummy VARIABLE_WIDTH
        return -1;
      }
    }

    unsigned int i = 2;
    for (i; i < xBins.size(); i++) {
      if (val1 < xBins[i]) {
        return i - 1;
      }
    }
    // overflow
    return this->mIgnoreOverflows ? -1 : i - 1;
  }

  using BinningPolicyBase<C>::getArrowColumns;
  using BinningPolicyBase<C>::getChunks;
  using BinningPolicyBase<C>::getChunkData;
  using BinningPolicyBase<C>::getRowData;

 private:
  void expandConstantBinning(std::vector<double> const& xBins)
  {
    if (xBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(xBins[0]);
      this->mBins[0].clear();
      this->mBins[0].resize(nBins + 2);
      this->mBins[0][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[0]) + 1, std::end(this->mBins[0]), xBins[2] - xBins[1] / nBins);
    }
  }
};

template <typename C1, typename C2>
struct PairBinningPolicy : public BinningPolicyBase<C1, C2> {
  PairBinningPolicy(std::vector<double>& xBins, std::vector<double>& yBins, bool ignoreOverflows = true) : BinningPolicyBase<C1, C2>({xBins, yBins}, ignoreOverflows)
  {
    expandConstantBinning(xBins, yBins);
  }
  PairBinningPolicy(int xNBins, typename C1::type xBinMin, typename C1::type xBinMax, int yNBins, typename C2::type yBinMin, typename C2::type yBinMax, bool ignoreOverflows = true) : PairBinningPolicy<C1, C2>({xNBins, xBinMin, xBinMax}, {yNBins, yBinMin, yBinMax}, ignoreOverflows)
  {
  }

  int getBin(std::tuple<typename C1::type, typename C2::type> const& data) const
  {
    auto val1 = std::get<0>(data);
    auto val2 = std::get<1>(data);
    auto xBins = this->mBins[0];
    auto yBins = this->mBins[1];

    if (this->mIgnoreOverflows) {
      // underflow
      if (val1 < xBins[1]) { // xBins[0] is a dummy VARIABLE_WIDTH
        return -1;
      }
      if (val2 < yBins[1]) { // yBins[0] is a dummy VARIABLE_WIDTH
        return -1;
      }
    }

    unsigned int i = 2, j = 2;
    for (i; i < xBins.size(); i++) {
      if (val1 < xBins[i]) {
        for (j; j < yBins.size(); j++) {
          if (val2 < yBins[j]) {
            return getBinAt(i, j);
          }
        }
        // overflow for yBins only
        return this->mIgnoreOverflows ? -1 : return getBinAt(i, j);
      }
    }

    if (this->mIgnoreOverflows) {
      // overflow
      return -1;
    }

    // overflow for xBins only
    for (j = 2; j < yBins.size(); j++) {
      if (val2 < yBins[j]) {
        return getBinAt(i, j);
      }
    }

    // overflow for both bins
    return getBinAt(i, j);
  }

  using BinningPolicyBase<C1, C2>::getArrowColumns;
  using BinningPolicyBase<C1, C2>::getChunks;
  using BinningPolicyBase<C1, C2>::getChunkData;
  using BinningPolicyBase<C1, C2>::getRowData;

 private:
  int getBinAt(unsigned int i, unsigned int j)
  {
    return (i - 1) + (j - 1) * this->mBins[0].size();
  }

  void expandConstantBinning(std::vector<double> const& xBins, std::vector<double> const& yBins)
  {
    if (xBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(xBins[0]);
      this->mBins[0].clear();
      this->mBins[0].resize(nBins + 2);
      this->mBins[0][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[0]) + 1, std::end(this->mBins[0]), xBins[2] - xBins[1] / nBins);
    }
    if (yBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(yBins[0]);
      this->mBins[1].clear();
      this->mBins[1].resize(nBins + 2);
      this->mBins[1][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[1]) + 1, std::end(this->mBins[1]), xBins[2] - xBins[1] / nBins);
    }
  }
};

template <typename C1, typename C2, typename C3>
struct TripleBinningPolicy : public BinningPolicyBase<C1, C2, C3> {
  TripleBinningPolicy(std::vector<double>& xBins, std::vector<double>& yBins, std::vector<double>& zBins, bool ignoreOverflows = true) : BinningPolicyBase<C1, C2, C3>({xBins, yBins, zBins}, ignoreOverflows)
  {
    expandConstantBinning(xBins, yBins, zBins);
  }
  TripleBinningPolicy(int xNBins, typename C1::type xBinMin, typename C1::type xBinMax, int yNBins, typename C2::type yBinMin, typename C2::type yBinMax, int zNBins, typename C2::type zBinMin, typename C3::type zBinMax, bool ignoreOverflows = true) : TripleBinningPolicy<C1, C2, C3>({xNBins, xBinMin, xBinMax}, {yNBins, yBinMin, yBinMax}, {zNBins, zBinMin, zBinMax}, ignoreOverflows)
  {
  }

  int getBin(std::tuple<typename C1::type, typename C2::type, typename C3::type> const& data) const
  {
    auto val1 = std::get<0>(data);
    auto val2 = std::get<1>(data);
    auto val3 = std::get<2>(data);
    auto xBins = this->mBins[0];
    auto yBins = this->mBins[1];
    auto zBins = this->mBins[2];
    if (this->mIgnoreOverflows) {
      // underflow
      if (val1 < xBins[1]) { // xBins[0] is a dummy VARIABLE_WIDTH
        return -1;
      }
      if (val2 < yBins[1]) { // yBins[0] is a dummy VARIABLE_WIDTH
        return -1;
      }
      if (val3 < zBins[1]) { // zBins[0] is a dummy VARIABLE_WIDTH
        return -1;
      }
    }

    unsigned int i = 2, j = 2, k = 2;
    for (i; i < xBins.size(); i++) {
      if (val1 < xBins[i]) {
        for (j; j < yBins.size(); j++) {
          if (val2 < yBins[j]) {
            for (k; k < zBins.size(); k++) {
              if (val3 < zBins[k]) {
                return getBinAt(i, j, k);
              }
            }
            // overflow for zBins only
            return this->mIgnoreOverflows ? -1 : getBinAt(i, j, k);
          }
        }
        if (this->mIgnoreOverflows) {
          return -1;
        }
        // overflow for yBins only
        for (k = 2; k < zBins.size(); k++) {
          if (val3 < zBins[k]) {
            return getBinAt(i, j, k);
          }
        }
        // overflow for zBins and yBins
        return getBinAt(i, j, k);
      }
    }

    if (this->mIgnoreOverflows) {
      // overflow
      return -1;
    }

    // overflow for xBins only
    for (j = 2; j < yBins.size(); j++) {
      if (val2 < yBins[j]) {
        for (k = 2; k < zBins.size(); k++) {
          if (val3 < zBins[k]) {
            return getBinAt(i, j, k);
          }
        }

        // overflow for xBins and zBins
        return getBinAt(i, j, k);
      }
    }

    // overflow for xBins and yBins
    for (k = 2; k < zBins.size(); k++) {
      if (val3 < zBins[k]) {
        return getBinAt(i, j, k);
      }
    }

    // overflow for all bins
    return getBinAt(i, j, k);
  }

  using BinningPolicyBase<C1, C2, C3>::getArrowColumns;
  using BinningPolicyBase<C1, C2, C3>::getChunks;
  using BinningPolicyBase<C1, C2, C3>::getChunkData;
  using BinningPolicyBase<C1, C2, C3>::getRowData;

 private:
  int getBinAt(unsigned int i, unsigned int j, unsigned int k)
  {
    return (i - 1) + (j - 1) * this->mBins[0].size() + (k - 1) * (this->mBins[0].size() + this->mBins[1].size());
  }

  void expandConstantBinning(std::vector<double> const& xBins, std::vector<double> const& yBins, std::vector<double> const& zBins)
  {
    if (xBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(xBins[0]);
      this->mBins[0].clear();
      this->mBins[0].resize(nBins + 2);
      this->mBins[0][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[0]) + 1, std::end(this->mBins[0]), xBins[2] - xBins[1] / nBins);
    }
    if (yBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(yBins[0]);
      this->mBins[1].clear();
      this->mBins[1].resize(nBins + 2);
      this->mBins[1][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[1]) + 1, std::end(this->mBins[1]), yBins[2] - yBins[1] / nBins);
    }
    if (zBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(zBins[0]);
      this->mBins[2].clear();
      this->mBins[2].resize(nBins + 2);
      this->mBins[2][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[2]) + 1, std::end(this->mBins[2]), zBins[2] - zBins[1] / nBins);
    }
  }
};

} // namespace o2::framework
#endif // FRAMEWORK_BINNINGPOLICY_H_
