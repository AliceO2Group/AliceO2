// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_COMPUTINGQUOTAOFFER_H_
#define O2_COMPUTINGQUOTAOFFER_H_

#include <functional>
#include <cstdint>
#include <cstddef>

namespace o2::framework
{

struct ComputingQuotaOfferRef {
  int index;
};

enum struct OfferScore {
  // The offers seen so far are enough. We can proceed with the dataprocessing.
  Enough,
  // The offers seen so far are not enough, but the current one
  // is ok and we need to look for more.
  More,
  // The offer is not needed, but something else in the device
  // might use it.
  Unneeded,
  // The offer is not suitable and should be given back to the
  // driver.
  Unsuitable
};

struct ComputingQuotaOffer {
  /// How many cores it can use
  int cpu = 0;
  /// How much memory it can use
  int64_t memory = 0;
  /// How much shared memory it can allocate
  int64_t sharedMemory = 0;
  /// How much runtime it can use before giving back the resource
  /// in milliseconds.
  int64_t runtime = 0;
  /// Which task is using the offer
  int user = -1;
  /// The score for the given offer
  OfferScore score = OfferScore::Unneeded;
  /// Whether or not the offer is valid
  bool valid = false;
};

struct ComputingQuotaInfo {
  // When the offer was received
  size_t received = 0;
  // First time it was used
  size_t firstUsed = 0;
  // Last time it was used
  size_t lastUsed = 0;
};

/// A request is a function which gets applied to all available
/// offers one after the other. If the offer itself is deemed
/// is ok for running.
using ComputingQuotaRequest = std::function<OfferScore(ComputingQuotaOffer const& offer, ComputingQuotaOffer const& accumulated)>;

/// A consumer is a function which updates a given function removing the
/// amount of resources which are considered as consumed.
using ComputingQuotaConsumer = std::function<void(int id, std::array<ComputingQuotaOffer, 16>&)>;

} // namespace o2::framework

#endif // O2_COMPUTINGQUOTAOFFER_H_
