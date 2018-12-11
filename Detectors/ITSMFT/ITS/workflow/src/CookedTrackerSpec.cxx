// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CookedTrackerSpec.cxx

#include <vector>

#include "TGeoGlobalMagField.h"

#include "Framework/ControlService.h"
#include "ITSWorkflow/CookedTrackerSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "ITSReconstruction/CookedTracker.h"
#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSBase/GeometryTGeo.h"

using namespace o2::framework;

namespace o2
{
namespace ITS
{

DataProcessorSpec getCookedTrackerSpec()
{
  auto init = [](InitContext &ic) {
    auto filename = ic.options().get<std::string>("grp-file");
    
    return [filename](ProcessingContext &pc) {
      static bool done=false;
      if (done) return;

      const auto grp = o2::parameters::GRPObject::loadFrom(filename.c_str());
      if (grp) {
        o2::Base::Propagator::initFieldFromGRP(grp);
        auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
      
        auto compClusters = pc.inputs().get<const std::vector<o2::ITSMFT::CompClusterExt>>("compClusters");
        auto clusters = pc.inputs().get<const std::vector<o2::ITSMFT::Cluster>>("clusters");
        auto labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");

        LOG(INFO)<<"ITSCookedTracker pulled "<<clusters.size()<<" clusters, "
                 << labels->getIndexedSize() << " MC label objects";

        o2::Base::GeometryManager::loadGeometry();
        o2::ITS::GeometryTGeo* geom = o2::ITS::GeometryTGeo::Instance();
        geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2GRot));

        std::vector<o2::ITS::TrackITS> tracks;
        o2::dataformats::MCTruthContainer<o2::MCCompLabel> trackLabels;

        o2::ITS::CookedTracker tracker;
        tracker.setBz(field->solenoidField());
        tracker.setGeometry(geom);
        tracker.setMCTruthContainers(labels.get(), &trackLabels);
	bool continuous = grp->isDetContinuousReadOut("ITS");
	LOG(INFO)<<"ITSCookedTracker RO: continuous="<<continuous;
        tracker.setContinuousMode(continuous);
      
        std::vector<std::array<Double_t, 3>> vertices; //FIXME :  run an actual vertex finder !
        vertices.push_back({0.,0.,0.});
        tracker.setVertices(vertices);

        tracker.process(clusters,tracks);
	
        LOG(INFO)<<"ITSCookedTracker pushed "<<tracks.size()<<" tracks";
        pc.outputs().snapshot(Output{"ITS","TRACKS",     0, Lifetime::Timeframe}, tracks);
        pc.outputs().snapshot(Output{"ITS","TRACKSMCTR", 0, Lifetime::Timeframe}, trackLabels);
      } else {
	LOG(ERROR)<<"Cannot retrieve GRP from the "<< filename.c_str()<<" file !";
      }
      done=true;
      //pc.services().get<ControlService>().readyToQuit(true);
    };
  };

  return DataProcessorSpec {
    "its-cooked-tracker",
    Inputs{
      InputSpec{"compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe},
      InputSpec{"clusters", "ITS", "CLUSTERS",     0, Lifetime::Timeframe},
      InputSpec{"labels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe}
    },
    Outputs{
      OutputSpec{"ITS", "TRACKS",     0, Lifetime::Timeframe},
      OutputSpec{"ITS", "TRACKSMCTR", 0, Lifetime::Timeframe}
    },
    AlgorithmSpec{ init },
    Options{
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the output file"}},
    }
  };
}
  
}
}
