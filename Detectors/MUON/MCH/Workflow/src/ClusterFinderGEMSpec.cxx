// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterFinderGEMSpec.cxx
/// \brief Implementation of a data processor to run the GEM MLEM cluster finder
///
/// \author Philippe Pillot, Subatech

#include "MCHWorkflow/ClusterFinderGEMSpec.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <stdexcept>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "MCHBase/PreCluster.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHClustering/ClusterFinderOriginal.h"
#include "MCHClustering/ClusterFinderGEM.h"
#include "MCHClustering/ClusterDump.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class ClusterFinderGEMTask
{
 public:
  static constexpr int DoOriginal= 0x0001;
  static constexpr int DoGEM= 0x0002;
  static constexpr int DumpOriginal= 0x0004;
  static constexpr int DumpGEM= 0x0008;
  static constexpr int GEMOutputStream = 0x0010; // default is Original
  //
  bool isGEMActivated() 
  {
    return( mode & DoGEM);    
  }
  
  bool isGEMDumped()
  {
    return( mode & DumpGEM);    
  }
  
  bool isOriginalActivated( )
  {
    return( mode & DoOriginal);    
  }

  bool isOriginalDumped( )
  {
    return( mode & DumpOriginal);    
  }

  bool isGEMOutputStream( ) const
  {
    return( mode & GEMOutputStream);    
  }
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    // mode = DoOriginal + DumpOriginal + GEMOutputStream;
    mode = DoOriginal | DumpOriginal;
    mode = DoGEM | DumpGEM;
    /// Prepare the clusterizer
    LOG(INFO) << "initializing cluster finder";

    if (isOriginalDumped() &&  ! isOriginalActivated())
        mode = mode & (~DumpOriginal);
    if (isGEMDumped() && ! isGEMActivated() )
        mode = mode & (~DumpGEM);
    if ( isOriginalDumped() )
      mOriginalDump = new ClusterDump("OrigRun2.dat", 0);
    if ( isGEMDumped() )
      mGEMDump = new ClusterDump("GEMRun2.dat", 0);
    
    //
    std::cout << "Configuration"  << std::endl;
    std::cout << "  Original: " << isOriginalActivated() << std::endl;
    std::cout << "  GEM     : " << isGEMActivated() << std::endl;
    std::cout << "  Dump Original: " << isOriginalDumped() << std::endl;
    std::cout << "  Dump GEM     : " << isGEMDumped() << std::endl;
    std::cout << "  GEM stream output: " << isGEMOutputStream() << std::endl;
    
    // mClusterFinder.init( ClusterFinderGEM::DoGEM );
    if ( isOriginalActivated() ) {
        mClusterFinderOriginal.init(  );
    } else if (isGEMActivated()) { 
        mClusterFinderOriginal.init(  );
        mClusterFinderGEM.init( mode );
    }
    
    /// Print the timer and clear the clusterizer when the processing is over
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, [this]() {
      LOG(INFO) << "cluster finder duration = " << mTimeClusterFinder.count() << " s";
      this->mClusterFinderOriginal.deinit();
      this->mClusterFinderGEM.deinit();
      
    });
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the preclusters and associated digits, clusterize and send the clusters for all events in the TF

    // get the input preclusters and associated digits
    auto preClusterROFs = pc.inputs().get<gsl::span<ROFRecord>>("preclusterrofs");
    auto preClusters = pc.inputs().get<gsl::span<PreCluster>>("preclusters");
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    //LOG(INFO) << "received time frame with " << preClusterROFs.size() << " interactions";

    // create the output messages for clusters and attached digits
    auto& clusterROFs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"clusterrofs"});
    auto& clusters = pc.outputs().make<std::vector<ClusterStruct>>(OutputRef{"clusters"});
    auto& usedDigits = pc.outputs().make<std::vector<Digit>>(OutputRef{"clusterdigits"});

    clusterROFs.reserve(preClusterROFs.size());
    for (const auto& preClusterROF : preClusterROFs) {
      LOG(INFO) << "processing interaction: ev ??? " << preClusterROF.getBCData().orbit << "...";
      // GG infos
      // uint16_t bc = DummyBC;       ///< bunch crossing ID of interaction
      // uint32_t orbit = DummyOrbit; ///< LHC orbit
      // clusterize every preclusters
      uint16_t bCrossing = preClusterROF.getBCData().bc;
      uint32_t orbit = preClusterROF.getBCData().orbit;
      uint32_t iPreCluster = 0;
      auto tStart = std::chrono::high_resolution_clock::now();
      //
      if (isOriginalActivated()) mClusterFinderOriginal.reset();
      if (isGEMActivated()) {
          mClusterFinderOriginal.reset();
          mClusterFinderGEM.reset();
      }
      // Get the starting index for new cluster founds
      size_t startGEMIdx = mClusterFinderGEM.getClusters().size();
      size_t startOriginalIdx = mClusterFinderOriginal.getClusters().size();
      std::cout << "start GEM Idx " <<  startGEMIdx << " Original" << startOriginalIdx << std::endl;
      for (const auto& preCluster : preClusters.subspan(preClusterROF.getFirstIdx(), preClusterROF.getNEntries())) {
        // Dump preclusters
          std::cout << "Debug ??? Dump preCluster digit start/size=" << preCluster.firstDigit <<" / " << preCluster.nDigits << std::endl;
        if ( isOriginalDumped())
            mClusterFinderGEM.dumpPreCluster( mOriginalDump, digits.subspan(preCluster.firstDigit, preCluster.nDigits), bCrossing, orbit, iPreCluster);  
        if ( isGEMDumped()) 
            mClusterFinderGEM.dumpPreCluster( mGEMDump, digits.subspan(preCluster.firstDigit, preCluster.nDigits), bCrossing, orbit, iPreCluster);  
        // Clusterize
        if (isOriginalActivated())
            mClusterFinderOriginal.findClusters(digits.subspan(preCluster.firstDigit, preCluster.nDigits));
        if (isGEMActivated())
            mClusterFinderGEM.findClusters(digits.subspan(preCluster.firstDigit, preCluster.nDigits), bCrossing, orbit, iPreCluster);
        // Dump clusters (results)
          std::cout << "Debug ??? Original found clusters.size=" << mClusterFinderOriginal.getClusters().size() << std::endl;
        if ( isOriginalDumped()) mClusterFinderGEM.dumpClusterResults( mOriginalDump, mClusterFinderOriginal.getClusters(), startOriginalIdx, bCrossing, orbit, iPreCluster);  
        if ( isGEMDumped()) mClusterFinderGEM.dumpClusterResults( mGEMDump, mClusterFinderGEM.getClusters(), startGEMIdx, bCrossing, orbit, iPreCluster);  
        // if ( isGEMDumped())    
        iPreCluster++;
      }
      auto tEnd = std::chrono::high_resolution_clock::now();
      mTimeClusterFinder += tEnd - tStart;

      // fill the ouput messages
      if ( isGEMOutputStream() )
        clusterROFs.emplace_back(preClusterROF.getBCData(), clusters.size(), mClusterFinderGEM.getClusters().size());
      else
        clusterROFs.emplace_back(preClusterROF.getBCData(), clusters.size(), mClusterFinderOriginal.getClusters().size());
      //
      writeClusters(clusters, usedDigits);
    }
  }

 private:
  //_________________________________________________________________________________________________
  void writeClusters(std::vector<ClusterStruct, o2::pmr::polymorphic_allocator<ClusterStruct>>& clusters,
                     std::vector<Digit, o2::pmr::polymorphic_allocator<Digit>>& usedDigits) const
  {
    /// fill the output messages with clusters and attached digits of the current event
    /// modify the references to the attached digits according to their position in the global vector
    auto clusterOffset = clusters.size();
    if ( isGEMOutputStream() ) {
      clusters.insert(clusters.end(), mClusterFinderGEM.getClusters().begin(), mClusterFinderGEM.getClusters().end());
    } else {
      clusters.insert(clusters.end(), mClusterFinderOriginal.getClusters().begin(), mClusterFinderOriginal.getClusters().end());
    }    
    auto digitOffset = usedDigits.size();
    if ( isGEMOutputStream() ) 
      usedDigits.insert(usedDigits.end(), mClusterFinderGEM.getUsedDigits().begin(), mClusterFinderGEM.getUsedDigits().end());
    else
      usedDigits.insert(usedDigits.end(), mClusterFinderOriginal.getUsedDigits().begin(), mClusterFinderOriginal.getUsedDigits().end());
        
    for (auto itCluster = clusters.begin() + clusterOffset; itCluster < clusters.end(); ++itCluster) {
      itCluster->firstDigit += digitOffset;
    }
  }

  ClusterFinderOriginal mClusterFinderOriginal{};                  ///< clusterizer
  ClusterFinderGEM mClusterFinderGEM{};                            ///< clusterizer
  int mode;                                                        ///< Original or GEM or both
  ClusterDump *mGEMDump;
  ClusterDump *mOriginalDump;
  std::chrono::duration<double> mTimeClusterFinder{}; ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getClusterFinderGEMSpec()
{
  return DataProcessorSpec{
    "ClusterFinderGEM",
    Inputs{InputSpec{"preclusterrofs", "MCH", "PRECLUSTERROFS", 0, Lifetime::Timeframe},
           InputSpec{"preclusters", "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe},
           InputSpec{"digits", "MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{{"clusterrofs"}, "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe},
            OutputSpec{{"clusters"}, "MCH", "CLUSTERS", 0, Lifetime::Timeframe},
            OutputSpec{{"clusterdigits"}, "MCH", "CLUSTERDIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<ClusterFinderGEMTask>()},
    Options{}};
}

} // end namespace mch
} // end namespace o2
