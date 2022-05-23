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
#ifndef ALICEO2_BCRANGE_H
#define ALICEO2_BCRANGE_H

#include "CommonDataFormat/Pair.h"

namespace o2
{
namespace dataformats
{
// .............................................................................
struct bcRanges {

  using limits = o2::dataformats::Pair<uint64_t, uint64_t>;

  // members
  const char* mlistName;
  std::vector<limits> mbcRangesList;
  bool isSorted;
  bool isMerged;
  bool isExtended;

 public:
  // constructor
  bcRanges(const char* label)
  {
    mlistName = label;
    reset();
  }

  // reset list
  void reset()
  {
    isSorted = false;
    isMerged = false;
    isExtended = false;
    mbcRangesList.clear();
  }

  char status()
  {
    return isSorted * (1 << 0) + isMerged * (1 << 1) + isExtended * (1 << 2);
  }

  // return number of BC ranges in list
  auto size()
  {
    return mbcRangesList.size();
  }

  // add BC range
  void add(uint64_t first, uint64_t last)
  {
    mbcRangesList.push_back(limits(first, last));
    isSorted = false;
    isMerged = false;
    isExtended = false;
  }

  // sort mbcRangesList according to first entries
  void sort()
  {
    std::sort(mbcRangesList.begin(), mbcRangesList.end(), [](limits a, limits b) {
      return a.first < b.first;
    });
    isSorted = true;
  }

  // get number of BCs not included in ranges
  template <typename BCs>
  uint64_t getnNotCompBCs(BCs bcs)
  {
    // needs to be merged
    if (!isMerged) {
      merge();
    }

    // loop over ranges and count number of BCs not contained in a range
    uint64_t nNotCompBCs = 0;
    uint64_t ilast = 1, inext;
    for (auto iter = mbcRangesList.begin(); iter != mbcRangesList.end(); ++iter) {
      inext = iter->first;
      if (iter == mbcRangesList.begin()) {
        nNotCompBCs += (inext - ilast);
      } else {
        nNotCompBCs += (inext - ilast - 1);
      }
      ilast = iter->second;
    }
    auto bclast = bcs.rawIteratorAt(bcs.size());
    nNotCompBCs += (bclast.globalIndex() - ilast);
    LOGF(debug, "Number of BCs not in range of compatible BCs: %i", nNotCompBCs);

    return nNotCompBCs;
  }

  // merge overlaping ranges
  void merge(bool toForce = false)
  {
    if (!isMerged || toForce) {
      std::vector<limits> tmpList;
      uint64_t ifirst = 0, ilast;

      // apply sorting of the ranges
      if (!isSorted) {
        sort();
      }

      // run over elements of mbcRangesList and merge lines where possible
      for (auto iter = mbcRangesList.begin(); iter != mbcRangesList.end(); ++iter) {
        if (iter == mbcRangesList.begin()) {
          ifirst = iter->first;
          ilast = iter->second;
          continue;
        }

        if (iter->first > (ilast + 1)) {
          // update tmpList
          tmpList.push_back(limits(ifirst, ilast));
          ifirst = iter->first;
          ilast = iter->second;
        } else {
          if (iter->second > ilast) {
            ilast = iter->second;
          }
        }
      }
      tmpList.push_back(limits(ifirst, ilast));

      mbcRangesList.clear();
      mbcRangesList = tmpList;
      isMerged = true;
    }
  }

  // add a factor fillFac of BCs not yet included in the BC ranges
  template <typename BCs>
  void compact(BCs bcs, Double_t fillFac, bool toForce = false)
  {
    if (!isExtended || toForce) {
      // apply merging of the ranges
      if (!isMerged || toForce) {
        merge(toForce);
      }

      // find out number of BCs not in a compatible range
      auto nBCs = bcs.size();
      auto nNotCompBCs = getnNotCompBCs(bcs);

      // keep adding BCs until the required number has been added
      auto nToAdd = (uint64_t)(nNotCompBCs * fillFac);
      int cnt = 0;
      while (nToAdd > 0) {
        // add BC at the beginning
        if (mbcRangesList[0].first > 1) {
          mbcRangesList[0].first--;
          nToAdd--;
        }

        // number of BCs to add in this round
        auto nr = mbcRangesList.size();
        if (nr > nToAdd) {
          nr = nToAdd;
        }

        // add BC after each range
        for (auto ii = 0; ii < nr; ii++) {
          if (mbcRangesList[ii].second < nBCs) {
            mbcRangesList[ii].second++;
            nToAdd--;
          }
        }
        merge(true);
      }
      isExtended = true;
    }
  }

  // check if the value index is in a range
  // and return true if this is the case
  bool isInRange(uint64_t index)
  {
    // make sure that the list is merged
    merge(false);

    // find the range in which the value index falls
    auto range = std::find_if(mbcRangesList.begin(), mbcRangesList.end(), [index](limits a) {
      return (index >= a.first) && (index <= a.second);
    });
    return (range != mbcRangesList.end());
  }

  // get BC range
  auto operator[](int index)
  {
    return mbcRangesList[index];
  }
  auto begin()
  {
    return mbcRangesList.begin();
  }
  auto end()
  {
    return mbcRangesList.end();
  }

  // return list name
  auto name()
  {
    return mlistName;
  }

  // return the list
  auto list()
  {
    return mbcRangesList;
  }
};

// .............................................................................
} // namespace dataformats

} // namespace o2

#endif // ALICEO2__BCRANGE_H
