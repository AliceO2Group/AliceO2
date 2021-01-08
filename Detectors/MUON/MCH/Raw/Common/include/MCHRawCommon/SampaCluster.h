// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_SAMPA_CLUSTER_H
#define O2_MCH_RAW_SAMPA_CLUSTER_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <fmt/format.h>
#include "MCHRawCommon/DataFormats.h"

namespace o2
{
namespace mch
{
namespace raw
{

/// @brief Piece of data for one Sampa channel.
///
/// A SampaCluster holds a parf of the data of one Sampa time window for
/// one Sampa channel. This data can be in one of two forms :
///
/// - raw ADC samples (10 bits each)
/// - clusterSum values (20 bits each)
///
/// with an associated (local) timetamp and a cluster size.
///
/// A full time window may contains several SampaClusters.

struct SampaCluster {

  /// Constructs a cluster which holds only a charge sum (aka cluster sum)
  /// \param sampaTime must fit within 10 bits
  /// \param chargeSum must fit within 20 bits
  /// \param clusterSize must fit within 10 bits
  ///
  /// if some parameter does not fit within its expected range
  /// a std::invalid_argument exception is thrown.
  explicit SampaCluster(uint10_t sampaTime, uint20_t bunchCrossing, uint20_t chargeSum, uint10_t clusterSize);

  /// Constructs a cluster which holds a vector of raw samples
  /// \param sampaTime must fit within 10 bits
  /// \param samples : each sample must fit within 10 bits
  ///
  /// if some parameter does not fit within its expected range
  /// a std::invalid_argument exception is thrown.
  SampaCluster(uint10_t sampaTime, uint20_t bunchCrossing, const std::vector<uint10_t>& samples);

  /// nofSamples gives the number of samples of this cluster.
  /// Can be > 1 even in chargesum mode (it then indicates the number
  /// of original samples that were integrated together)
  uint16_t nofSamples() const;

  /// isClusterSum returns true if this cluster is not holding raw samples.
  bool isClusterSum() const;

  /// nof10BitWords returns the number of 10 bits words
  /// needed to store this cluster
  uint16_t nof10BitWords() const;

  /// sum returns the total charge in the cluster
  uint32_t sum() const;

  uint10_t sampaTime;            //< 10 bits for a local time stamp
  uint20_t bunchCrossing;        //< 20 bits for bunch crossing counter
  uint20_t chargeSum;            //< 20 bits for a cluster sum
  uint10_t clusterSize;          //< 10 bits for cluster size
  std::vector<uint16_t> samples; //< 10 bits for each sample
};

// ensure all clusters are either in sample mode or in
// chargesum mode, no mixing allowed
// and that all clusters share a single bunch crossing counter value
template <typename CHARGESUM>
void assertNotMixingClusters(const std::vector<SampaCluster>& data)
{
  CHARGESUM a;
  auto refValue = a();
  uint32_t bunchCrossingCounter{0};
  for (auto i = 0; i < data.size(); i++) {
    if (data[i].isClusterSum() != refValue) {
      throw std::invalid_argument(fmt::format("all cluster of this encoder should be of the same type ({}) but {}-th does not match ", (refValue ? "clusterSum" : "samples"), i));
    }
    auto bc = data[i].bunchCrossing;
    if (i == 0) {
      bunchCrossingCounter = bc;
    }
    if (bc != bunchCrossingCounter) {
      throw std::invalid_argument(fmt::format("all sampa clusters should have the same bunch crossing number : first is {} and got {}\n", bunchCrossingCounter, bc));
    }
  }
}

std::ostream& operator<<(std::ostream& os, const SampaCluster& sc);

std::string asString(const SampaCluster& sc);

} // namespace raw
} // namespace mch
} // namespace o2
#endif
