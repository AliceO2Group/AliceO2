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
    unsigned int i = 2, j = 2, k = 2;
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
    } else {
      i = 1;
      j = 1;
      k = 1;
    }

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

  // Note: Overflow / underflow bin -1 is not included
  int getXBinsCount() const
  {
    return this->mBins[0].size() - 1 - getOverflowShift();
  }

  // Note: Overflow / underflow bin -1 is not included
  int getYBinsCount() const
  {
    if constexpr (sizeof...(Cs) == 0) {
      return 0;
    }
    return this->mBins[1].size() - 1 - getOverflowShift();
  }

  // Note: Overflow / underflow bin -1 is not included
  int getZBinsCount() const
  {
    if constexpr (sizeof...(Cs) < 2) {
      return 0;
    }
    return this->mBins[2].size() - 1 - getOverflowShift();
  }

  // Note: Overflow / underflow bin -1 is not included
  int getAllBinsCount() const
  {
    if constexpr (sizeof...(Cs) == 0) {
      return getXBinsCount();
    }
    if constexpr (sizeof...(Cs) == 1) {
      return getXBinsCount() * getYBinsCount();
    }
    if constexpr (sizeof...(Cs) == 2) {
      return getXBinsCount() * getYBinsCount() * getZBinsCount();
    }
    return -1;
  }

  using persistent_columns_t = framework::selected_pack<o2::soa::is_persistent_t, C, Cs...>;

 private:
  // We substract 1 to account for VARIABLE_WIDTH in the bins vector
  // We substract second 1 if we omit values below minima (underflow, mapped to -1)
  // Otherwise we add 1 and we get the number of bins including those below and over the outer edges
  int getBinAt(unsigned int iRaw, unsigned int jRaw, unsigned int kRaw) const
  {
    int shiftBinsWithoutOverflow = getOverflowShift();
    unsigned int i = iRaw - 1 - shiftBinsWithoutOverflow;
    unsigned int j = jRaw - 1 - shiftBinsWithoutOverflow;
    unsigned int k = kRaw - 1 - shiftBinsWithoutOverflow;
    auto xBinsCount = getXBinsCount();
    if constexpr (sizeof...(Cs) == 0) {
      return i;
    } else if constexpr (sizeof...(Cs) == 1) {
      return i + j * xBinsCount;
    } else if constexpr (sizeof...(Cs) == 2) {
      return i + j * xBinsCount + k * xBinsCount * (this->mBins[1].size() - 1 - shiftBinsWithoutOverflow);
    } else {
      return -1;
    }
  }

  int getOverflowShift() const
  {
    return mIgnoreOverflows ? 1 : -1;
  }

  void expandConstantBinning(std::vector<double> const& bins, int ind)
  {
    if (bins[0] != VARIABLE_WIDTH) {
      int nBins = static_cast<int>(bins[0]);
      this->mBins[ind].clear();
      this->mBins[ind].resize(nBins + 2);
      this->mBins[ind][0] = VARIABLE_WIDTH;
      for (int i = 0; i <= nBins; i++) {
        this->mBins[ind][i + 1] = bins[1] + i * (bins[2] - bins[1]) / nBins;
      }
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

  using persistent_columns_t = framework::selected_pack<o2::soa::is_persistent_t, C>;
};

} // namespace o2::framework
#endif // FRAMEWORK_BINNINGPOLICY_H_
