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

#include "Framework/ASoAHelpers.h"
#include "Framework/Pack.h"
#include <optional>

namespace o2::framework
{

struct BinningPolicyBase {
  BinningPolicyBase(std::vector<AxisSpec>& bins, bool ignoreOverflows = true) : mBins(bins), mIgnoreOverflows(ignoreOverflows) {}

  template <typename... Bs>
  int getBin(Bs...)
  {
    return -1;
  }

  std::vector<AxisSpec> mBins;
  bool mIgnoreOverflows;
};

struct NoBinningPolicy {
  // Just take the bin number from the column data
  NoBinningPolicy(std::string const& xColName) : BinningPolicyBase(AxisSpec{{}, nullopt, xColName}) {}

  template <typename B1>
  int getBin(B1 val1)
  {
    return val1;
  }
}

struct SingleBinningPolicy {
  template <typename B1>
  SingleBinningPolicy(std::string const& xColName, std::vector<B1> xBins, bool ignoreOverflows = true) : BinningPolicyBase({AxisSpec{xBins, std::nullopt, xColName}}, ignoreOverflows)
  {
    expandConstantBinning(xBins);
  }
  template <typename B1>
  SingleBinningPolicy(std::string const& xColName, int xNBins, B1 xBinMin, B1 xBinMax, bool ignoreOverflows = true) : SingleBinningPolicy(xColName, {xNBins, xBinMin, xBinMax}, ignoreOverflows)
  {
  }

  template <typename B1>
  int getBin(B1 val1)
  {
    auto xBins = this->mBins[0].binEdges;
    if (mIgnoreOverflows) {
      // underflow
      if (val1 < xBins[1]) { // xBins[0] is a dummy VARIABLE_WIDTH
        return -1;
      }
    }

    for (unsigned int i = 2; i < xBins.size(); i++) {
      if (val1 < xBins[i]) {
        return i - 1;
      }
    }
    // overflow
    return ignoreOverflows ? -1 : xBins.size() - 1;
  }

 private:
  template <typename B1, typename B2>
  void expandConstantBinning(std::vector<B1> const& xBins, std::vector<B2> const& yBins)
  {
    if (xBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(xBins[0]);
      this->mBins[0].binEdges.clear;
      this->mBins[0].resize(nBins + 2);
      this->mBins[0][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[0].binEdges) + 1, std::end(this->mBins[0].binEdges), xBins[2] - xBins[1] / nBins);
    }
  }
};

struct PairBinningPolicy {
  PairBinningPolicy(std::string const& xColName, ConfigurableAxis& xBins, std::string const& yColName, ConfigurableAxis& yBins, bool ignoreOverflows = true) : BinningPolicyBase({AxisSpec{xBins, std::nullopt, xColName}, AxisSpec{yBins, std::nullopt, yColName}}, ignoreOverflows)
  {
    expandConstantBinning(xBins, yBins);
  }
  template <typename B1, typename B2>
  PairBinningPolicy(std::string const& xColName, std::vector<B1> const& xBins, std::string const& yColName, std::vector<B2> const& yBins, bool ignoreOverflows = true) : BinningPolicyBase({AxisSpec{xBins, std::nullopt, xColName}, AxisSpec{yBins, std::nullopt, yColName}}, ignoreOverflows)
  {
    expandConstantBinning(xBins, yBins);
  }
  template <typename B1, typename B2>
  PairBinningPolicy(std::string const& xColName, int xNBins, B1 xBinMin, B1 xBinMax, std::string const& yColName, int yNBins, B2 yBinMin, B2 yBinMax, bool ignoreOverflows = true) : PairBinningPolicy(xColName, {xNBins, xBinMin, xBinMax}, yColName, {yNBins, yBinMin, yBinMax}, zColName, {zNBins, zBinMin, zBinMax}, ignoreOverflows)
  {
  }

  template <typename B1, typename B2>
  int getBin(B1 val1, B2 val2)
  {
    auto xBins = this->mBins[0].binEdges;
    auto yBins = this->mBins[1].binEdges;
    if (mIgnoreOverflows) {
      // underflow
      if (val1 < xBins[1]) { // xBins[0] is a dummy VARIABLE_WIDTH
        return -1;
      }
      if (val2 < yBins[1]) { // yBins[0] is a dummy VARIABLE_WIDTH
        return -1;
      }
    }

    for (unsigned int i = 2; i < xBins.size(); i++) {
      if (val1 < xBins[i]) {
        for (unsigned int j = 2; j < yBins.size(); j++) {
          if (val2 < yBins[j]) {
            return (i - 1) + (j - 1) * xBins.size();
          }
        }
        // overflow for yBins only
        return mIgnoreOverflows ? -1 : (i - 1) + (yBins.size() - 1) * xBins.size();
      }
    }

    if (mIgnoreOverflows) {
      // overflow
      return -1;
    }

    // overflow for xBins only
    for (int j = 1; j < yBins.size(); j++) {
      if (val2 < yBins[j]) {
        return (xBins.size() - 1) + (j - 1) * xBins.size();
      }
    }

    // overflow for both bins
    return xBins.size() * yBins.size() - 1;
  }

 private:
  template <typename B1, typename B2>
  void expandConstantBinning(std::vector<B1> const& xBins, std::vector<B2> const& yBins)
  {
    if (xBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(xBins[0]);
      this->mBins[0].binEdges.clear;
      this->mBins[0].resize(nBins + 2);
      this->mBins[0][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[0].binEdges) + 1, std::end(this->mBins[0].binEdges), xBins[2] - xBins[1] / nBins);
    }
    if (yBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(yBins[0]);
      this->mBins[1].binEdges.clear;
      this->mBins[1].resize(nBins + 2);
      this->mBins[1][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[1].binEdges) + 1, std::end(this->mBins[1].binEdges), xBins[2] - xBins[1] / nBins);
    }
  }
};

struct TripleBinningPolicy {
  template <typename B1, typename B2, typename B3>
  TripleBinningPolicy(std::string const& xColName, std::vector<B1> const& xBins, std::string const& yColName, std::vector<B2> const& yBins, std::string const& zColName, std::vector<B2> const& zBins, bool ignoreOverflows = true) : BinningPolicyBase({AxisSpec{xBins, std::nullopt, xColName}, AxisSpec{yBins, std::nullopt, yColName}, AxisSpec{zBins, std::nullopt, zColName}}, ignoreOverflows)
  {
    expandConstantBinning(xBins, yBins, zBins);
  }
  template <typename B1, typename B2, typename B3>
  TripleBinningPolicy(std::string const& xColName, int xNBins, B1 xBinMin, B1 xBinMax, std::string const& yColName, int yNBins, B2 yBinMin, B2 yBinMax, std::string const& zColName, int zNBins, B3 zBinMin, B3 zBinMax, bool ignoreOverflows = true) : TripleBinningPolicy(xColName, {xNBins, xBinMin, xBinMax}, yColName, {yNBins, yBinMin, yBinMax}, zColName, {zNBins, zBinMin, zBinMax}, ignoreOverflows)
  {
  }

  template <typename B1, typename B2, typename B3>
  int getBin(B1 val1, B2 val2, B3 val3)
  {
    auto xBins = std::get<0>(this->mBins);
    auto yBins = std::get<1>(this->mBins);
    auto zBins = std::get<2>(this->mBins);
    if (mIgnoreOverflows) {
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
                return (i - 1) + (j - 1) * xBins.size() + (k - 1) * (xBins.size() + yBins.size());
              }
            }
            // overflow for zBins only
            return mIgnoreOverflows ? -1 : (i - 1) + (j - 1) * xBins.size() + (k - 1) * (xBins.size() + yBins.size);
          }
        }
        if (mIgnoreOverflows) {
          return -1;
        }
        // overflow for yBins only
        for (k = 2; k < zBins.size(); k++) {
          if (val3 < zBins[k]) {
            return (i - 1) + (j - 1) * xBins.size() + (k - 1) * (xBins.size + yBins.size());
          }
        }
        // overflow for zBins and yBins
        return (i - 1) + (j - 1) * xBins.size() + (k - 1) * (xBins.size() + yBins.size());
      }
    }

    if (mIgnoreOverflows) {
      // overflow
      return -1;
    }

    // overflow for xBins only
    for (j = 2; j < yBins.size(); j++) {
      if (val2 < yBins[j]) {
        for (k = 2; k < zBins.size(); k++) {
          if (val3 < zBins[k]) {
            return (i - 1) + (j - 1) * xBins.size() + (k - 1) * (xBins.size() + yBins.size());
          }
        }

        // overflow for xBins and zBins
        return (i - 1) + (j - 1) * xBins.size() + (k - 1) * (xBins.size() + yBins.size());
      }
    }

    // overflow for xBins and yBins
    for (k = 2; k < zBins.size(); k++) {
      if (val3 < zBins[k]) {
        return (i - 1) + (j - 1) * xBins.size() + (k - 1) * (xBins.size() + yBins.size());
      }
    }

    // overflow for all bins
    return (i - 1) + (j - 1) * xBins.size() + (k - 1) * (xBins.size() + yBins.size());
  }

 private:
  template <typename B1, typename B2, typename B3>
  void expandConstantBinning(std::vector<B1> const& xBins, std::vector<B2> const& yBins)
  {
    if (xBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(xBins[0]);
      this->mBins[0].binEdges.clear;
      this->mBins[0].resize(nBins + 2);
      this->mBins[0][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[0].binEdges) + 1, std::end(this->mBins[0].binEdges), xBins[2] - xBins[1] / nBins);
    }
    if (yBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(yBins[0]);
      this->mBins[1].binEdges.clear;
      this->mBins[1].resize(nBins + 2);
      this->mBins[1][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[1].binEdges) + 1, std::end(this->mBins[1].binEdges), yBins[2] - yBins[1] / nBins);
    }
    if (zBins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(zBins[0]);
      this->mBins[2].binEdges.clear;
      this->mBins[2].resize(nBins + 2);
      this->mBins[2][0] = VARIABLE_WIDTH;
      std::iota(std::begin(this->mBins[2].binEdges) + 1, std::end(this->mBins[2].binEdges), zBins[2] - zBins[1] / nBins);
    }
  }
};

} // namespace o2::framework
#endif // FRAMEWORK_BINNINGPOLICY_H_
