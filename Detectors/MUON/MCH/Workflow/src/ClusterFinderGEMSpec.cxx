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

/// \file ClusterFinderGEMSpec.cxx
/// \brief Implementation of a data processor to run the GEM MLEM cluster finder
///

#include "ClusterFinderGEMSpec.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <string>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "MCHBase/ErrorMap.h"
#include "MCHBase/PreCluster.h"
#include "DataFormatsMCH/Cluster.h"
#include "MCHClustering/ClusterFinderOriginal.h"
#include "MCHClustering/ClusterFinderGEM.h"
#include "MCHClustering/ClusterDump.h"
#include "Framework/ConfigParamRegistry.h"
#include "CommonUtils/ConfigurableParam.h"
#include "MCHClustering/ClusterizerParam.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class ClusterFinderGEMTask
{
 public:
  static constexpr int DoOriginal = 0x0001;
  static constexpr int DoGEM = 0x0002;
  static constexpr int DumpOriginal = 0x0004;
  static constexpr int DumpGEM = 0x0008;
  static constexpr int GEMOutputStream = 0x0010; // default is Original
  static constexpr int TimingStats = 0x0020;
  static constexpr char statFileName[] = "statistics.csv";
  std::fstream statStream;
  //
  bool isActive(int selectedMode) const
  {
    return (mode & selectedMode);
  }
  /* invalid
  bool isGEMDumped()
  {
    return (mode & DumpGEM);
  }

  bool isOriginalActivated()
  {
    return (mode & DoOriginal);
  }

  bool isOriginalDumped()
  {
    return (mode & DumpOriginal);
  }

  bool isGEMOutputStream() const
  {
    return (mode & GEMOutputStream);
  }

  bool isGEMTimingStats() const
  {
    return (mode & isGEMTimingStats());
  }
  */
  void saveStatistics(uint32_t orbit, uint16_t bunchCrossing, uint32_t iPreCluster, uint16_t nPads, uint16_t nbrClusters, uint16_t DEId, double duration)
  {
    statStream << iPreCluster << " " << bunchCrossing << " " << orbit << " "
               << nPads << " " << nbrClusters << " " << DEId << " " << duration << std::endl;
  }

  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the clusterizer
    LOG(info) << "initializing cluster finder";

    auto config = ic.options().get<std::string>("mch-config");
    if (!config.empty()) {
      o2::conf::ConfigurableParam::updateFromFile(config, "MCHClustering", true);
    }
    bool run2Config = ic.options().get<bool>("run2-config");

    // GG auto mode = ic.options().get<int>("mode");
    mode = ic.options().get<int>("mode");
    printf("------------------------------------> GG Mode mode=%d\n", mode);

    /// Prepare the clusterizer
    LOG(info) << "initializing cluster finder";

    if (isActive(DumpOriginal) && !isActive(DoOriginal)) {
      mode = mode & (~DumpOriginal);
    }
    if (isActive(DumpGEM) && !isActive(DoGEM)) {
      mode = mode & (~DumpGEM);
    }
    if (isActive(DumpOriginal)) {
      mOriginalDump = new ClusterDump("OrigRun2.dat", 0);
    }
    if (isActive(DumpGEM)) {
      mGEMDump = new ClusterDump("GEMRun2.dat", 0);
    }
    if (isActive(TimingStats)) {
      statStream.open(statFileName, std::fstream::out);
      statStream << "# iPrecluster bunchCrossing   orbit  nPads  nClusters  DEId  duration (in ms)" << std::endl;
    }

    //
    LOG(info) << "Configuration";
    LOG(info) << "  Mode    : " << mode;
    LOG(info) << "  Original: " << isActive(DoOriginal);
    LOG(info) << "  GEM     : " << isActive(DoGEM);
    LOG(info) << "  Dump Original:         " << isActive(DumpOriginal);
    LOG(info) << "  Dump GEM     :         " << isActive(DumpGEM);
    LOG(info) << "  GEM stream output    : " << isActive(GEMOutputStream);
    LOG(info) << "  Timing statistics: " << isActive(TimingStats);

    // mClusterFinder.init( ClusterFinderGEM::DoGEM );
    if (isActive(DoOriginal)) {
      mClusterFinderOriginal.init(run2Config);
    } else if (isActive(DoGEM)) {
      mClusterFinderGEM.init(mode, run2Config);
    }
    // Inv ??? LOG(info) << "GG = lowestPadCharge = " << ClusterizerParam::Instance().lowestPadCharge;

    /// Print the timer and clear the clusterizer when the processing is over
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>([this]() {
      LOG(info) << "cluster finder duration = " << mTimeClusterFinder.count() << " s";
      if (isActive(DoOriginal)) {
        this->mClusterFinderOriginal.deinit();
      } else if (isActive(DoGEM)) {
        this->mClusterFinderGEM.deinit();
      }
      if (isActive(DumpOriginal)) {
        delete mOriginalDump;
        mOriginalDump = nullptr;
      }
      if (isActive(DumpGEM)) {
        delete mGEMDump;
        mGEMDump = nullptr;
      }
      mErrorMap.forEach([](Error error) {
        LOGP(warning, error.asString());
      });
    });
    auto stop = [this]() {
      /// close the output file
      LOG(info) << "stop GEM";
      // this->mOutputFile.close();
    };
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the preclusters and associated digits, clusterize and send the clusters for all events in the TF

    // get the input preclusters and associated digits
    auto preClusterROFs = pc.inputs().get<gsl::span<ROFRecord>>("preclusterrofs");
    auto preClusters = pc.inputs().get<gsl::span<PreCluster>>("preclusters");
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    // LOG(info) << "received time frame with " << preClusterROFs.size() << " interactions";

    // create the output messages for clusters and attached digits
    auto& clusterROFs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"clusterrofs"});
    auto& clusters = pc.outputs().make<std::vector<Cluster>>(OutputRef{"clusters"});
    auto& usedDigits = pc.outputs().make<std::vector<Digit>>(OutputRef{"clusterdigits"});
    uint32_t iPreCluster = 0;

    clusterROFs.reserve(preClusterROFs.size());
    ErrorMap errorMap; // TODO: use this errorMap to score processing errors

    for (const auto& preClusterROF : preClusterROFs) {
      // LOG(info) << "processing interaction: time frame " << preClusterROF.getBCData().orbit << "...";
      // GG infos
      // uint16_t bc = DummyBC;       ///< bunch crossing ID of interaction
      // uint32_t orbit = DummyOrbit; ///< LHC orbit
      // clusterize every preclusters
      uint16_t bCrossing = preClusterROF.getBCData().bc;
      uint32_t orbit = preClusterROF.getBCData().orbit;
      std::chrono::duration<double> preClusterDuration{}; ///< timer
      auto tStart = std::chrono::high_resolution_clock::now();

      // Inv ??? if ( orbit==22 ) {
      //
      if (isActive(DoOriginal)) {
        mClusterFinderOriginal.reset();
      }
      if (isActive(DoGEM)) {
        mClusterFinderGEM.reset();
      }
      // Get the starting index for new cluster founds
      size_t startGEMIdx = mClusterFinderGEM.getClusters().size();
      size_t startOriginalIdx = mClusterFinderOriginal.getClusters().size();
      uint16_t nbrClusters(0);
      // std::cout << "Start index GEM=" <<  startGEMIdx << ", Original=" << startOriginalIdx << std::endl;
      for (const auto& preCluster : preClusters.subspan(preClusterROF.getFirstIdx(), preClusterROF.getNEntries())) {
        auto tPreClusterStart = std::chrono::high_resolution_clock::now();
        // Inv ??? for (const auto& preCluster : preClusters.subspan(preClusterROF.getFirstIdx(), 1102)) {
        startGEMIdx = mClusterFinderGEM.getClusters().size();
        startOriginalIdx = mClusterFinderOriginal.getClusters().size();
        // Dump preclusters
        // std::cout << "bCrossing=" << bCrossing << ", orbit=" << orbit << ", iPrecluster" << iPreCluster
        //        << ", PreCluster: digit start=" << preCluster.firstDigit <<" , digit size=" << preCluster.nDigits << std::endl;
        if (isActive(DumpOriginal)) {
          mClusterFinderGEM.dumpPreCluster(mOriginalDump, digits.subspan(preCluster.firstDigit, preCluster.nDigits), bCrossing, orbit, iPreCluster);
        }
        if (isActive(DumpGEM)) {
          mClusterFinderGEM.dumpPreCluster(mGEMDump, digits.subspan(preCluster.firstDigit, preCluster.nDigits), bCrossing, orbit, iPreCluster);
        }
        // Clusterize
        if (isActive(DoOriginal)) {
          mClusterFinderOriginal.findClusters(digits.subspan(preCluster.firstDigit, preCluster.nDigits));
          nbrClusters = mClusterFinderOriginal.getClusters().size() - startOriginalIdx;
        }
        if (isActive(DoGEM)) {
          mClusterFinderGEM.findClusters(digits.subspan(preCluster.firstDigit, preCluster.nDigits), bCrossing, orbit, iPreCluster);
          nbrClusters = mClusterFinderGEM.getClusters().size() - startGEMIdx;
        }
        // Dump clusters (results)
        // std::cout << "[Original] total clusters.size=" << mClusterFinderOriginal.getClusters().size() << std::endl;
        // std::cout << "[GEM     ] total clusters.size=" << mClusterFinderGEM.getClusters().size() << std::endl;
        if (isActive(DumpOriginal)) {
          mClusterFinderGEM.dumpClusterResults(mOriginalDump, mClusterFinderOriginal.getClusters(), startOriginalIdx, bCrossing, orbit, iPreCluster);
        }
        if (isActive(DumpGEM)) {
          mClusterFinderGEM.dumpClusterResults(mGEMDump, mClusterFinderGEM.getClusters(), startGEMIdx, bCrossing, orbit, iPreCluster);
        }
        // Timing Statistics
        if (isActive(TimingStats)) {
          auto tPreClusterEnd = std::chrono::high_resolution_clock::now();
          preClusterDuration = tPreClusterEnd - tPreClusterStart;
          int16_t nPads = preCluster.nDigits;
          int16_t DEId = digits[preCluster.firstDigit].getDetID();
          // double dt = duration_cast<duration<double>>(tPreClusterEnd - tPreClusterStart).count;
          // std::chrono::duration<double> time_span = std::chrono::duration_cast<duration<double>>(tPreClusterEnd - tPreClusterStart);
          preClusterDuration = tPreClusterEnd - tPreClusterStart;
          double dt = preClusterDuration.count();
          // In second
          dt = (dt < 1.0e-06) ? 0.0 : dt * 1000;
          saveStatistics(orbit, bCrossing, iPreCluster, nPads, nbrClusters, DEId, dt);
        }
        iPreCluster++;
      }
      // } // Inv ??? if ( orbit==22 ) {
      auto tEnd = std::chrono::high_resolution_clock::now();
      mTimeClusterFinder += tEnd - tStart;

      // fill the ouput messages
      if (isActive(GEMOutputStream)) {
        clusterROFs.emplace_back(preClusterROF.getBCData(), clusters.size(), mClusterFinderGEM.getClusters().size());
      } else {
        clusterROFs.emplace_back(preClusterROF.getBCData(), clusters.size(), mClusterFinderOriginal.getClusters().size());
      }
      //
      writeClusters(clusters, usedDigits);
    }

    // create the output message for clustering errors
    auto& clusterErrors = pc.outputs().make<std::vector<Error>>(OutputRef{"clustererrors"});
    errorMap.forEach([&clusterErrors](Error error) {
      clusterErrors.emplace_back(error);
    });
    mErrorMap.add(errorMap);

    LOGP(info, "Found {:4d} clusters from {:4d} preclusters in {:2d} ROFs",
         clusters.size(), preClusters.size(), preClusterROFs.size());
  }

 private:
  //_________________________________________________________________________________________________
  void writeClusters(std::vector<Cluster, o2::pmr::polymorphic_allocator<Cluster>>& clusters,
                     std::vector<Digit, o2::pmr::polymorphic_allocator<Digit>>& usedDigits) const
  {
    /// fill the output messages with clusters and attached digits of the current event
    /// modify the references to the attached digits according to their position in the global vector
    auto clusterOffset = clusters.size();
    if (isActive(GEMOutputStream)) {
      clusters.insert(clusters.end(), mClusterFinderGEM.getClusters().begin(), mClusterFinderGEM.getClusters().end());
    } else {
      clusters.insert(clusters.end(), mClusterFinderOriginal.getClusters().begin(), mClusterFinderOriginal.getClusters().end());
    }
    auto digitOffset = usedDigits.size();
    if (isActive(GEMOutputStream)) {
      usedDigits.insert(usedDigits.end(), mClusterFinderGEM.getUsedDigits().begin(), mClusterFinderGEM.getUsedDigits().end());
    } else {
      usedDigits.insert(usedDigits.end(), mClusterFinderOriginal.getUsedDigits().begin(), mClusterFinderOriginal.getUsedDigits().end());
    }

    for (auto itCluster = clusters.begin() + clusterOffset; itCluster < clusters.end(); ++itCluster) {
      itCluster->firstDigit += digitOffset;
    }
  }

  ClusterFinderOriginal mClusterFinderOriginal{}; ///< clusterizer
  ClusterFinderGEM mClusterFinderGEM{};           ///< clusterizer
  int mode;                                       ///< Original or GEM or both
  ClusterDump* mGEMDump;
  ClusterDump* mOriginalDump;
  ErrorMap mErrorMap{};                               ///< counting of encountered errors
  std::chrono::duration<double> mTimeClusterFinder{}; ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getClusterFinderGEMSpec(const char* specName)
{
  return DataProcessorSpec{
    specName,
    Inputs{InputSpec{"preclusterrofs", "MCH", "PRECLUSTERROFS", 0, Lifetime::Timeframe},
           InputSpec{"preclusters", "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe},
           InputSpec{"digits", "MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{{"clusterrofs"}, "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe},
            OutputSpec{{"clusters"}, "MCH", "CLUSTERS", 0, Lifetime::Timeframe},
            OutputSpec{{"clusterdigits"}, "MCH", "CLUSTERDIGITS", 0, Lifetime::Timeframe},
            OutputSpec{{"clustererrors"}, "MCH", "CLUSTERERRORS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<ClusterFinderGEMTask>()},
    Options{
      {"mch-config", VariantType::String, "", {"JSON or INI file with clustering parameters"}},
      {"run2-config", VariantType::Bool, false, {"Setup for run2 data"}},
      {"mode", VariantType::Int, ClusterFinderGEMTask::DoGEM | ClusterFinderGEMTask::GEMOutputStream, {"Running mode"}},
      // {"mode", VariantType::Int, ClusterFinderGEMTask::DoOriginal, {"Running mode"}},
      // {"mode", VariantType::Int, ClusterFinderGEMTask::DoGEM | ClusterFinderGEMTask::GEMOutputStream, {"Running mode"}},
      // {"mode", VariantType::Int, ClusterFinderGEMTask::DoGEM | ClusterFinderGEMTask::DumpGEM | ClusterFinderGEMTask::GEMOutputStream, {"Running mode"}},
      // {"mode", VariantType::Int, ClusterFinderGEMTask::DoOriginal | ClusterFinderGEMTask::GEMOutputStream, {"Running mode"}},
      //{"mode", VariantType::Int, ClusterFinderGEMTask::DoOriginal | ClusterFinderGEMTask::DumpOriginal | ClusterFinderGEMTask::GEMOutputStream, {"Running mode"}},
      // {"mode", VariantType::Int, ClusterFinderGEMTask::DoGEM, {"Running mode"}},

      // {"mode", VariantType::Int, ClusterFinderGEMTask::DoGEM | ClusterFinderGEMTask::DumpGEM | ClusterFinderGEMTask::GEMOutputStream, {"Running mode"}},
      // {"mode", VariantType::Int, ClusterFinderGEMTask::DoGEM, {"Running mode"}},
    }};
}

} // end namespace mch
} // end namespace o2
