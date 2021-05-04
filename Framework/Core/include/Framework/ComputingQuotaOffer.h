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
  /// Whether or not the offer is being used
  bool used;
  /// Whether or not the offer is valid
  bool valid;
};

struct ComputingQuotaInfo {
  // When the offer was received
  size_t received = 0;
  // First time it was used
  size_t firstUsed = 0;
  // Last time it was used
  size_t lastUsed = 0;
};

/// A request is a function which evaluates to true if the offer
/// is ok for running. The higher the return value, the
/// better match is the given resource for the computation.
using ComputingQuotaRequest = std::function<int8_t(ComputingQuotaOffer const&)>;

} // namespace o2::framework

#endif // O2_COMPUTINGQUOTAOFFER_H_
