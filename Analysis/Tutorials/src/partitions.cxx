// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief Partitions are subsets of tables.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct PartitionOutside {
  float fPI = static_cast<float>(M_PI);
  Configurable<float> ptlow{"pTlow", 0.5f, "Lowest pT"};
  Configurable<float> ptup{"pTup", 2.0f, "highest pT"};
  Filter ptFilter_a = aod::track::pt > ptlow;
  Filter ptFilter_b = aod::track::pt < ptup;

  Configurable<float> etalow{"etaLow", -1.0f, "lowest eta"};
  Configurable<float> etaup{"etaUp", 1.0f, "highest eta"};
  Filter etafilter = (aod::track::eta < etaup) && (aod::track::eta > etalow);

  Configurable<float> philow{"phiLow", 1.0f, "lowest phi"};
  Configurable<float> phiup{"phiUp", 2.0f, "highest phi"};

  // all defined filters are applied
  using myTracks = soa::Filtered<aod::Tracks>;

  // definition of partitions
  Partition<myTracks> leftPhi = aod::track::phiraw < philow;
  Partition<myTracks> midPhi = aod::track::phiraw >= philow && aod::track::phiraw < phiup;
  Partition<myTracks> rightPhi = aod::track::phiraw >= phiup;

  // partitions are created and provided within the process function
  void process(aod::Collision const& collision, myTracks const& tracks)
  {

    // all defined partitions are available
    LOGF(INFO, "Collision: %d [N = %d] [left phis = %d] [mid phis = %d] [right phis = %d]",
         collision.globalIndex(), tracks.size(), leftPhi.size(), midPhi.size(), rightPhi.size());

    for (auto& track : leftPhi) {
      LOGF(INFO, "id = %d, from collision: %d, collision: %d; eta:  %.3f < %.3f < %.3f; phi: %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.collision().globalIndex(), collision.globalIndex(), (float)etalow, track.eta(), (float)etaup, track.phiraw(), (float)philow, (float)ptlow, track.pt(), (float)ptup);
    }
    for (auto& track : midPhi) {
      LOGF(INFO, "id = %d, from collision: %d, collision: %d; eta: %.3f < %.3f < %.3f; phi: %.3f <= %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.collision().globalIndex(), collision.globalIndex(), (float)etalow, track.eta(), (float)etaup, (float)philow, track.phiraw(), (float)phiup, (float)ptlow, track.pt(), (float)ptup);
    }
    for (auto& track : rightPhi) {
      LOGF(INFO, "id = %d, from collision: %d, collision: %d; eta: %.3f < %.3f < %.3f; phi: %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.collision().globalIndex(), collision.globalIndex(), (float)etalow, track.eta(), (float)etaup, (float)phiup, track.phiraw(), (float)ptlow, track.pt(), (float)ptup);
    }
  }
};

// Partition inside process
// Caveat: partitioned table cannot be passed as const& to process()
struct PartitionInside {
  void process(aod::Collisions const& collisions, aod::Tracks& tracks)
  {
    for (auto& c : collisions) {

      // create the partition groupedTracks
      Partition<aod::Tracks> groupedTracks = aod::track::collisionId == c.globalIndex();
      groupedTracks.bindTable(tracks);

      // loop over the partition groupedTracks
      for (auto& t : groupedTracks) {
        LOGF(INFO, "collision global index: %d grouped track collision id: %d", c.globalIndex(), t.collisionId());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<PartitionOutside>(cfgc),
    adaptAnalysisTask<PartitionInside>(cfgc),
  };
}
