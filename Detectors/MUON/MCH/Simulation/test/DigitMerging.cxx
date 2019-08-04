// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DigitMerging.h"
#include <map>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <set>

using o2::mch::Digit;

void dumpDigits(const std::vector<Digit>& digits)
{
  int i{0};
  std::cout << "dumpDigits" << std::string(40, '-') << "\n";
  for (auto& d : digits) {
    std::cout << "i=" << i << ":[" << d.getPadID() << "," << d.getADC() << " ] ";
    i++;
  }
  std::cout << "\n";
}

std::vector<Digit> mergeDigits_sortnosizeadjust(const std::vector<Digit>& inputDigits, const std::vector<o2::MCCompLabel>& inputLabels)
{
  std::vector<int> indices(inputDigits.size());
  std::iota(begin(indices), end(indices), 0);

  std::sort(indices.begin(), indices.end(), [&inputDigits](int a, int b) {
    return inputDigits[a].getPadID() < inputDigits[b].getPadID();
  });

  auto sortedDigits = [&inputDigits, &indices](int i) {
    return inputDigits[indices[i]];
  };

  auto sortedLabels = [&inputLabels, &indices](int i) {
    return inputLabels[indices[i]];
  };

  std::vector<Digit> digits;

  std::vector<o2::MCCompLabel> labels;

  int i = 0;
  while (i < indices.size()) {
    int j = i + 1;
    while (j < indices.size() && (sortedDigits(i).getPadID() == sortedDigits(j).getPadID())) {
      j++;
    }
    float adc{0};
    for (int k = i; k < j; k++) {
      adc += sortedDigits(k).getADC();
    }
    digits.emplace_back(sortedDigits(i).getTimeStamp(), sortedDigits(i).getDetID(), sortedDigits(i).getPadID(), adc);
    labels.emplace_back(sortedLabels(i).getTrackID(), sortedLabels(i).getEventID(), sortedLabels(i).getSourceID(), false);
    i = j;
  }
  return digits;
}

std::vector<Digit> mergeDigits_sortsizeadjust(const std::vector<Digit>& inputDigits, const std::vector<o2::MCCompLabel>& inputLabels)
{
  std::vector<int> indices(inputDigits.size());
  std::iota(begin(indices), end(indices), 0);

  std::sort(indices.begin(), indices.end(), [&inputDigits](int a, int b) {
    return inputDigits[a].getPadID() < inputDigits[b].getPadID();
  });

  auto sortedDigits = [&inputDigits, &indices](int i) {
    return inputDigits[indices[i]];
  };

  auto sortedLabels = [&inputLabels, &indices](int i) {
    return inputLabels[indices[i]];
  };

  std::vector<Digit> digits;
  digits.reserve(inputDigits.size());

  std::vector<o2::MCCompLabel> labels;
  labels.reserve(inputLabels.size());

  int i = 0;
  while (i < indices.size()) {
    int j = i + 1;
    while (j < indices.size() && (sortedDigits(i).getPadID() == sortedDigits(j).getPadID())) {
      j++;
    }
    float adc{0};
    for (int k = i; k < j; k++) {
      adc += sortedDigits(k).getADC();
    }
    digits.emplace_back(sortedDigits(i).getTimeStamp(), sortedDigits(i).getDetID(), sortedDigits(i).getPadID(), adc);
    labels.emplace_back(sortedLabels(i).getTrackID(), sortedLabels(i).getEventID(), sortedLabels(i).getSourceID(), false);
    i = j;
  }
  digits.resize(digits.size());
  labels.resize(labels.size());
  return digits;
}

std::vector<Digit> mergeDigits_map(const std::vector<Digit>& inputDigits, const std::vector<o2::MCCompLabel>& inputLabels)
{
  int iter = 0;
  int index = 0;
  std::map<int, int> padidmap;
  std::set<int> forRemoval;
  std::vector<Digit> digits{inputDigits};
  std::vector<o2::MCCompLabel> labels{inputLabels};

  for (auto& digit : digits) {
    int count = 0;
    int padid = digit.getPadID();
    count = padidmap.count(padid);
    if (count) {
      std::pair<std::map<int, int>::iterator, std::map<int, int>::iterator> ret;

      ret = padidmap.equal_range(padid);
      index = ret.first->second;
      (digits.at(index)).setADC(digits.at(index).getADC() + digit.getADC());
      forRemoval.emplace(iter);
    } else {
      padidmap.emplace(padid, iter);
    }
    ++iter;
  }

  int rmcounts = 0;
  for (auto& rmindex : forRemoval) {
    digits.erase(digits.begin() + rmindex - rmcounts);
    labels.erase(labels.begin() + rmindex - rmcounts);
    ++rmcounts;
  }
  return digits;
}

std::vector<MergingFunctionType> mergingFunctions()
{
  return std::vector<MergingFunctionType>{mergeDigits_sortnosizeadjust, mergeDigits_sortsizeadjust, mergeDigits_map};
}
