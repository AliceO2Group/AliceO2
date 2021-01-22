// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUQAHelper.h
/// \author David Rohr

#ifndef GPUQAHELPER_H
#define GPUQAHELPER_H

#include <utility>
#include <vector>
#include <type_traits>

#ifdef GPUCA_STANDALONE
#include "AliHLTTPCClusterMCData.h"
#endif

struct AliHLTTPCClusterMCWeight;
struct AliHLTTPCClusterMCLabel;

namespace o2
{
class MCCompLabel;
namespace gpu
{
namespace internal
{

template <bool WEIGHT, class T, class S, class U = T>
class GPUTPCTrkLbl
{
 public:
  GPUTPCTrkLbl(const S* v, float maxFake = 0.1f) : mClusterLabels(v), mTrackMCMaxFake(maxFake) { mLabels.reserve(5); };
  GPUTPCTrkLbl(const GPUTPCTrkLbl&) = default;
  inline void reset()
  {
    mLabels.clear();
    mNCl = 0;
    mTotalWeight = 0.f;
  }
  inline void addLabel(unsigned int elementId)
  {
    if constexpr (std::is_same<T, AliHLTTPCClusterMCWeight>::value) {
      for (unsigned int i = 0; i < sizeof(mClusterLabels[elementId]) / sizeof(mClusterLabels[elementId].fClusterID[0]); i++) {
        const auto& element = mClusterLabels[elementId].fClusterID[i];
        if (element.fMCID >= 0) {
          if constexpr (WEIGHT) {
            mTotalWeight += element.fWeight;
          }
          bool found = false;
          for (unsigned int l = 0; l < mLabels.size(); l++) {
            if (mLabels[l].first.fMCID == element.fMCID) {
              mLabels[l].second++;
              if constexpr (WEIGHT) {
                mLabels[l].first.fWeight += element.fWeight;
              }
              found = true;
              break;
            }
          }
          if (!found) {
            mLabels.emplace_back(element, 1);
          }
        }
      }
    } else {
      for (const auto& element : mClusterLabels->getLabels(elementId)) {
        bool found = false;
        for (unsigned int l = 0; l < mLabels.size(); l++) {
          if (mLabels[l].first == element) {
            mLabels[l].second++;
            found = true;
            break;
          }
        }
        if (!found) {
          mLabels.emplace_back(element, 1);
        }
      }
    }
    mNCl++;
  }
  inline U computeLabel(float* labelWeight = nullptr, float* totalWeight = nullptr, int* maxCount = nullptr)
  {
    if (mLabels.size() == 0) {
      return U(); //default constructor creates NotSet label
    } else {
      unsigned int bestLabelNum = 0, bestLabelCount = 0;
      for (unsigned int j = 0; j < mLabels.size(); j++) {
        if (mLabels[j].second > bestLabelCount) {
          bestLabelNum = j;
          bestLabelCount = mLabels[j].second;
        }
      }
      auto& bestLabel = mLabels[bestLabelNum].first;
      if constexpr (std::is_same<T, AliHLTTPCClusterMCWeight>::value && WEIGHT) {
        *labelWeight = bestLabel.fWeight;
        *totalWeight = mTotalWeight;
        *maxCount = bestLabelCount;
      } else {
        (void)labelWeight;
        (void)totalWeight;
        (void)maxCount;
      }
      U retVal = bestLabel;
      if (bestLabelCount < (1.f - mTrackMCMaxFake) * mNCl) {
        retVal.setFakeFlag();
      }
      return retVal;
    }
  }

 private:
  const S* mClusterLabels;
  std::vector<std::pair<T, unsigned int>> mLabels;
  const float mTrackMCMaxFake;
  unsigned int mNCl = 0;
  float mTotalWeight = 0.f;
};
} // namespace internal

struct GPUTPCTrkLbl_ret {
  long int id = -1;
  GPUTPCTrkLbl_ret() = default;
  template <class T>
  GPUTPCTrkLbl_ret(T){};
#ifdef GPUCA_TPC_GEOMETRY_O2
  GPUTPCTrkLbl_ret(const MCCompLabel& a) : id(a.getTrackEventSourceID()){};
#endif
#ifdef GPUCA_STANDALONE
  GPUTPCTrkLbl_ret(const AliHLTTPCClusterMCWeight& a) : id(a.fMCID){};
#endif
  void setFakeFlag()
  {
    id = -1;
  }
};

template <bool WEIGHT = false, class U = void, class T, template <class> class S, typename... Args>
static inline auto GPUTPCTrkLbl(const S<T>* x, Args... args)
{
  if constexpr (std::is_same<U, void>::value) {
    return internal::GPUTPCTrkLbl<WEIGHT, T, S<T>>(x, args...);
  } else {
    return internal::GPUTPCTrkLbl<WEIGHT, T, S<T>, U>(x, args...);
  }
}

template <bool WEIGHT = false, class U = void, typename... Args>
static inline auto GPUTPCTrkLbl(const AliHLTTPCClusterMCLabel* x, Args... args)
{
  using S = AliHLTTPCClusterMCLabel;
  using T = AliHLTTPCClusterMCWeight;
  if constexpr (std::is_same<U, void>::value) {
    return internal::GPUTPCTrkLbl<WEIGHT, T, S>(x, args...);
  } else {
    return internal::GPUTPCTrkLbl<WEIGHT, T, S, U>(x, args...);
  }
}

} // namespace gpu
} // namespace o2

#endif
