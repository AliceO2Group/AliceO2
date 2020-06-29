// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TimeClusterFinder.h
/// \brief Class to group the fired pads according to their time stamp
///
/// \author Andrea Ferrero, CEA

#ifndef ALICEO2_MCH_TIMECLUSTERFINDER_H_
#define ALICEO2_MCH_TIMECLUSTERFINDER_H_

#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <gsl/span>

#include "MCHBase/Digit.h"
#include "MCHBase/PreCluster.h"

namespace o2
{
namespace mch
{

class TimeClusterFinder
{
 public:
  TimeClusterFinder();
  ~TimeClusterFinder();

  TimeClusterFinder(const TimeClusterFinder&) = delete;
  TimeClusterFinder& operator=(const TimeClusterFinder&) = delete;
  TimeClusterFinder(TimeClusterFinder&&) = delete;
  TimeClusterFinder& operator=(TimeClusterFinder&&) = delete;

  void init();
  void deinit();
  void reset();

  void loadDigits(gsl::span<const Digit> digits);

  int run();

  void getTimeClusters(gsl::span<PreCluster>& preClusters, gsl::span<Digit>& digits);

 private:
  struct DetectionElement;

  void storeCluster(std::vector<const Digit*>& digits);

  void createMapping();

  static constexpr int SNDEs = 156; ///< number of DEs

  std::vector<std::unique_ptr<DetectionElement>> mDEs; ///< internal mapping
  std::unordered_map<int, int> mDEIndices{};           ///< maps DE indices from DE IDs

  gsl::span<Digit> mOutputDigits{};           ///< array of output digits
  gsl::span<PreCluster> mOutputPreclusters{}; ///< array of output preclusters
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_TIMECLUSTERFINDER_H_
