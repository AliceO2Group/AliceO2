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
#include "RecoWorkflow.h"
#include <utility> // std::forward

namespace o2
{
namespace framework
{
struct CompletionPolicy;
}

namespace tpc
{

namespace ca
{
// The CA tracker is now a wrapper to not only the actual tracking on GPU but
// also the decoding of the zero-suppressed raw format and the clusterer.
enum struct Operation {
  DecompressTPC,          // run cluster decompressor
  DecompressTPCFromROOT,  // the cluster decompressor input is a root object not flat
  CAClusterer,            // run the CA clusterer
  ZSDecoder,              // run the ZS raw data decoder
  ZSOnTheFly,             // use zs on the fly
  OutputTracks,           // publish tracks
  OutputCAClusters,       // publish the clusters produced by CA clusterer
  OutputCompClusters,     // publish CompClusters container
  OutputCompClustersFlat, // publish CompClusters container
  OutputQA,               // Ship QA histograms to QC
  OutputSharedClusterMap, // Ship optional shared cluster map
  ProcessMC,              // process MC labels
  SendClustersPerSector,  // Send clusters and clusters mc labels per sector
  Noop,                   // skip argument on the constructor
};

/// Helper struct to pass the individual ca::Operation flags to
/// the processor spec. The struct is initialized by a variable list of
/// constructor arguments.
struct Config {
  Config(const Config&) = default;
  template <typename... Args>
  Config(Args&&... args)
  {
    init(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void init(Operation const& op, Args&&... args)
  {
    switch (op) {
      case Operation::DecompressTPC:
        decompressTPC = true;
        break;
      case Operation::DecompressTPCFromROOT:
        decompressTPCFromROOT = true;
        break;
      case Operation::CAClusterer:
        caClusterer = true;
        break;
      case Operation::ZSDecoder:
        zsDecoder = true;
        break;
      case Operation::ZSOnTheFly:
        zsOnTheFly = true;
        break;
      case Operation::OutputTracks:
        outputTracks = true;
        break;
      case Operation::OutputCompClusters:
        outputCompClusters = true;
        break;
      case Operation::OutputCompClustersFlat:
        outputCompClustersFlat = true;
        break;
      case Operation::OutputCAClusters:
        outputCAClusters = true;
        break;
      case Operation::OutputQA:
        outputQA = true;
        break;
      case Operation::OutputSharedClusterMap:
        outputSharedClusterMap = true;
        break;
      case Operation::ProcessMC:
        processMC = true;
        break;
      case Operation::SendClustersPerSector:
        sendClustersPerSector = true;
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

  // Cannot specialize constructor to create proper copy constructor directly --> must overload init
  void init(const Config& x) { *this = x; }

  bool decompressTPC = false;
  bool decompressTPCFromROOT = false;
  bool caClusterer = false;
  bool zsDecoder = false;
  bool zsOnTheFly = false;
  bool outputTracks = false;
  bool outputCompClusters = false;
  bool outputCompClustersFlat = false;
  bool outputCAClusters = false;
  bool outputQA = false;
  bool outputSharedClusterMap = false;
  bool processMC = false;
  bool sendClustersPerSector = false;
};

} // namespace ca

/// create a processor spec for the CATracker
/// The CA tracker is actually much more than the tracker it has evolved to a
/// general interface processor for TPC GPU algorithms. This includes currently
/// decoding of zero-suppressed raw data, ca clusterer and ca tracking. The input
/// is chosen depending on the mode.
///
/// The input specs are created depending on the list of tpc sectors, with separate
/// routes per sector. If the processor is also runnig the clusterer and cluster
/// output is enabled, the outputs are created based on the list of TPC sectors.
///
/// The individual operations of the CA processor can be switched using enum
/// @ca::Operations, a configuration object @a ca::Config is used to pass the
/// configuration to the processor spec.
///
/// @param specconfig configuration options for the processor spec
/// @param tpcsectors list of sector numbers
framework::DataProcessorSpec getCATrackerSpec(o2::tpc::reco_workflow::CompletionPolicyData* policyData, ca::Config const& specconfig, std::vector<int> const& tpcsectors);

o2::framework::CompletionPolicy getCATrackerCompletionPolicy();
} // end namespace tpc
} // end namespace o2
