// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHTimeClustering/TimeClusterFinder.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>
#include <map>
#include <algorithm>
#include <fmt/format.h>

#include <fairmq/Tools.h>
#include <FairMQLogger.h>

#include "MCHMappingInterface/Segmentation.h"

namespace o2
{
namespace mch
{

using namespace std;

static bool sPrint = false;

ostream& operator<<(ostream& ostr, const Digit& d)
{
  ostr << fmt::format("PAD ({:04d} {:04d})\tADC {:04d}\tTIME ({} {} {})",
                      d.getDetID(), d.getPadID(), d.getADC(), d.getTime().orbit, d.getTime().bunchCrossing, d.getTime().sampaTime);
  return ostr;
}

struct CompareDigitTime final {
  bool operator()(const Digit& left, const Digit& right) const
  {
    return (left.getTime().getBXTime() < right.getTime().getBXTime());
  }
};

struct TimeClusterFinder::DetectionElement {
  using DigitsMap = std::multimap<Digit, bool, CompareDigitTime>;
  DigitsMap mDigits[2];
  int orbit[2] = {-1, -1};
  int currentBufId = {1};
  int previousBufId = {0};

  std::function<void(std::vector<const Digit*>&)> sendTimeCluster;

  DetectionElement(std::function<void(std::vector<const Digit*>&)> f) : sendTimeCluster(f)
  {
  }

  int getCurrentOrbit() { return orbit[currentBufId]; }

  void mergeAndSendTimeClusters();

  void storeDigit(const Digit& d);
};

void TimeClusterFinder::DetectionElement::storeDigit(const Digit& d)
{
  if (d.getTime().orbit != getCurrentOrbit()) {
    mergeAndSendTimeClusters();

    currentBufId = 1 - currentBufId;
    previousBufId = 1 - previousBufId;
    mDigits[currentBufId].clear();
    orbit[currentBufId] = d.getTime().orbit;
  }

  mDigits[currentBufId].insert(std::make_pair(d, false));
}

static bool compareDigits(const Digit& lhs, const Digit& rhs)
{
  return (lhs.getTime().getBXTime() < rhs.getTime().getBXTime());
}

static int64_t getTimeDiff(Digit& d1, Digit& d2)
{
  return (static_cast<int64_t>(d2.getTime().getBXTime()) - static_cast<int64_t>(d1.getTime().getBXTime()));
}

//_________________________________________________________________________________________________
void TimeClusterFinder::DetectionElement::mergeAndSendTimeClusters()
{
  if (sPrint) {
    std::cout << "\n[mergeAndSendTimeClusters] list of digits:\n";
    for (auto& d : mDigits[previousBufId])
      std::cout << d.first << " [prev]" << std::endl;
    for (auto& d : mDigits[currentBufId])
      std::cout << d.first << " [cur]" << std::endl;
  }

  // collect the time clusters
  // step 1: find index of first non-merged digit in previous orbit
  DigitsMap::iterator iStart, iEnd, iDigit;
  uint64_t tStart = 0, tEnd = 0;
  for (iStart = mDigits[previousBufId].begin(); iStart != mDigits[previousBufId].end(); iStart++) {
    if (iStart->second == false) {
      break;
    }
  }
  if (iStart == mDigits[previousBufId].end())
    return;

  // the first non-merged digit is used as seed to search for the next time cluster
  while (iStart != mDigits[previousBufId].end()) {

    iEnd = iStart;

    if (sPrint) {
      std::cout << "[mergeAndSendTimeClusters] previousBufId: " << previousBufId << "  iStart: " << iStart->first << std::endl;
    }
    const Digit* d1 = &(iStart->first);

    // we add the seed digit to the next time cluster
    if (sPrint) {
      std::cout << "[mergeAndSendTimeClusters] adding digit to cluster [0]: " << *d1 << std::endl;
    }
    std::vector<const Digit*> timeClusterDigits;
    timeClusterDigits.push_back(d1);

    // the added digit is marked as merged
    iStart->second = true;

    // initialize the time bounds of the current cluster
    tStart = d1->getTime().getBXTime();
    tEnd = tStart;

    // step 2: loop on digits from previous orbit, starting from the one after the seed
    iDigit = iStart;
    iDigit++;
    for (; iDigit != mDigits[previousBufId].end(); iDigit++) {

      // skip digits that have already been merged to a time cluster
      if (iDigit->second) {
        continue;
      }

      // we check the time difference between the current digit and the previous one, in
      // units of bunch crossings
      const Digit* d2 = &(iDigit->first);
      uint64_t td2 = d2->getTime().getBXTime();

      // the cluster is ended if the time difference with respect to the last digit
      // is larger than 10 ADC samples, or equivalently 40 bunch crossings
      if (td2 > (tEnd + 40)) {
        break;
      }

      // the digit is added to the time cluster and marked as merged
      if (sPrint) {
        std::cout << "[mergeAndSendTimeClusters] adding digit to cluster [1]: " << *d2 << std::endl;
      }
      timeClusterDigits.push_back(d2);
      iDigit->second = true;

      // update of the time bounds
      if (td2 > tEnd) {
        tEnd = td2;
      }

      // we update the pointer and index to the last digit in the cluster
      d1 = d2;
      iEnd = iDigit;
    }

    // step 3: loop on digits from current orbit, stop when no more digits can be added to the current cluster
    for (iDigit = mDigits[currentBufId].begin(); iDigit != mDigits[currentBufId].end(); iDigit++) {

      // skip digits that have already been merged to a time cluster
      if (iDigit->second) {
        continue;
      }

      // we check the time difference between the current digit and the last one in the cluster,
      // in units of bunch crossings
      const Digit* d2 = &(iDigit->first);
      uint64_t td2 = d2->getTime().getBXTime();

      // the cluster is ended if the digit is outside of the time bounds by more than 10 ADC samples,
      // or equivalently 40 bunch crossings.
      if (((td2 + 40) < tStart) || (td2 > (tEnd + 40))) {
        break;
      }

      // the digit is added to the time cluster and marked as merged
      if (sPrint) {
        std::cout << "[mergeAndSendTimeClusters] adding digit to cluster [2]: " << *d2 << std::endl;
      }
      timeClusterDigits.push_back(d2);
      iDigit->second = true;

      // update of the time bounds
      if (td2 > tEnd) {
        tEnd = td2;
      } else if (td2 < tStart) {
        tStart = td2;
      }

      // we update the pointer to the last digit in the cluster
      // the index is not updated because it refers to the vector of digits
      // in the previous orbit
      d1 = d2;
    }

    if (!timeClusterDigits.empty()) {
      sendTimeCluster(timeClusterDigits);
    }

    // we set the new seed to the digit immediately following the last one in the current cluster
    iStart = iEnd;
    iStart++;
  }
}

//_________________________________________________________________________________________________
TimeClusterFinder::TimeClusterFinder() : mDEs{}
{
  /// default constructor: prepare the internal mapping structures
  const auto clusterHandler = [&](std::vector<const Digit*>& digits) {
    storeCluster(digits);
  };
  for (auto i = 0; i < SNDEs; i++) {
    mDEs.emplace_back(new DetectionElement(clusterHandler));
  }
}

//_________________________________________________________________________________________________
TimeClusterFinder::~TimeClusterFinder() = default;

//_________________________________________________________________________________________________
void TimeClusterFinder::init()
{
  /// load the mapping and fill the internal structures

  createMapping();
}

//_________________________________________________________________________________________________
void TimeClusterFinder::deinit()
{
  /// clear the internal structure
  reset();
  mDEIndices.clear();
}

//_________________________________________________________________________________________________
void TimeClusterFinder::reset()
{
  /// reset digits and precluster arrays. The allocated memory must be free'd by the calling function.
  size_t size = 0;
  mOutputDigits = gsl::span<Digit>(static_cast<Digit*>(nullptr), size_t{0});
  mOutputPreclusters = gsl::span<PreCluster>(static_cast<PreCluster*>(nullptr), size_t{0});
}

//_________________________________________________________________________________________________
void TimeClusterFinder::loadDigits(gsl::span<const Digit> digits)
{
  /// fill the Mapping::MpDE structure with fired pads

  for (const auto& digit : digits) {

    int deIndex = mDEIndices[digit.getDetID()];
    assert(deIndex >= 0 && deIndex < SNDEs);

    DetectionElement& de(*(mDEs[deIndex]));
    de.storeDigit(digit);
  }
}

//_________________________________________________________________________________________________
template <typename T>
bool expandArray(gsl::span<T>& array, size_t newSize)
{
  void* oldPtr = array.data();
  size_t newSizeBytes = newSize * sizeof(T);
  T* newPtr = reinterpret_cast<T*>(realloc(oldPtr, newSizeBytes));
  if (!newPtr) {
    return false;
  }
  array = gsl::span<T>(newPtr, newSize);
  return true;
}

//_________________________________________________________________________________________________
void TimeClusterFinder::storeCluster(std::vector<const Digit*>& digits)
{
  if (sPrint) {
    std::cout << "\n[storeCluster]:\n";
    for (auto& d : digits) {
      std::cout << *d << std::endl;
    }
  }

  size_t oldSize = mOutputDigits.size();
  size_t newSize = oldSize + digits.size();
  if (!expandArray(mOutputDigits, newSize)) {
    return;
  }
  if (sPrint) {
    std::cout << "[storeCluster] new digits size: " << mOutputDigits.size() << std::endl;
  }

  // append digits
  for (size_t i = 0; i < digits.size(); i++) {
    memcpy(&(mOutputDigits[i + oldSize]), digits[i], sizeof(Digit));
  }

  PreCluster precluster;
  precluster.firstDigit = oldSize;
  precluster.nDigits = digits.size();

  // append precluster
  oldSize = mOutputPreclusters.size();
  newSize = oldSize + 1;
  if (!expandArray(mOutputPreclusters, newSize)) {
    return;
  }
  if (sPrint) {
    std::cout << "[storeCluster] new preclusters size: " << mOutputPreclusters.size() << std::endl;
  }
  memcpy(&(mOutputPreclusters[oldSize]), &precluster, sizeof(PreCluster));
}

//_________________________________________________________________________________________________
int TimeClusterFinder::run()
{
}

//_________________________________________________________________________________________________
void TimeClusterFinder::getTimeClusters(gsl::span<PreCluster>& preClusters, gsl::span<Digit>& digits)
{
  preClusters = mOutputPreclusters;
  digits = mOutputDigits;
}

//_________________________________________________________________________________________________
void TimeClusterFinder::createMapping()
{
  /// Fill the internal mapping structures

  auto tStart = std::chrono::high_resolution_clock::now();

  int iDE = 0;
  mDEIndices.reserve(SNDEs);

  std::unordered_map<int, int>& deIDs = mDEIndices;

  // create the internal mapping for each DE
  mapping::forEachDetectionElement([&deIDs, &iDE](int deID) {
    deIDs.emplace(deID, iDE);
    iDE += 1;
  });

  if (mDEIndices.size() != SNDEs) {
    throw runtime_error("invalid mapping");
  }

  auto tEnd = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "create mapping in: " << std::chrono::duration<double, std::milli>(tEnd - tStart).count() << " ms";
}

} // namespace mch
} // namespace o2
