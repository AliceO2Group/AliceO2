// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CATrackerSpec.h
/// @author Matthias Richter
/// @since  2018-04-18
/// @brief  Processor spec for running TPC CA tracking

#include "Framework/DataProcessorSpec.h"
#include <utility> // std::forward

namespace o2
{
namespace tpc
{

namespace ca
{
// The CA tracker is now a wrapper to not only the actual tracking on GPU but
// also the decoding of the zero-suppressed raw format and the clusterer.
enum struct Operation {
  CAClusterer,        // run the CA clusterer
  ZSDecoder,          // run the ZS raw data decoder
  OutputTracks,       // publish tracks
  OutputCompClusters, // publish CompClusters container
  ProcessMC,          // process MC labels
  Noop,               // skip argument on the constructor
};

struct Config {
  template <typename... Args>
  Config(Args&&... args)
  {
    init(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void init(Operation const& op, Args&&... args)
  {
    switch (op) {
      case Operation::CAClusterer:
        caClusterer = true;
        break;
      case Operation::ZSDecoder:
        zsDecoder = true;
        break;
      case Operation::OutputTracks:
        outputTracks = true;
        break;
      case Operation::OutputCompClusters:
        outputCompClusters = true;
        break;
      case Operation::ProcessMC:
        processMC = true;
        break;
      case Operation::Noop:
        break;
      default:
        throw std::runtime_error("invalid CATracker operation");
    }
    if constexpr (sizeof...(args) > 0) {
      init(std::forward<Args>(args)...);
    }
  }

  bool caClusterer = false;
  bool zsDecoder = false;
  bool outputTracks = false;
  bool outputCompClusters = false;
  bool processMC = false;
};

} // namespace ca

/// create a processor spec
/// read simulated TPC clusters from file and publish
framework::DataProcessorSpec getCATrackerSpec(ca::Config const& config, std::vector<int> const& inputIds);

} // end namespace tpc
} // end namespace o2
