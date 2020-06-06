// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Analysis/SecondaryVertex.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>

using namespace o2;
using namespace o2::framework;

struct HFTrackIndexSkimsCreator {

  Produces<aod::SecVtx2Prong> secvtx2prong;
  Configurable<int> triggerindex{"triggerindex", -1, "trigger index"};

  void process(aod::Collision const& collision,
               aod::BCs const& bcs,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    int trigindex = int{triggerindex};
    if (trigindex != -1) {
      //LOGF(info, "Selecting on trigger bit %d", trigindex);
      uint64_t triggerMask = collision.bc().triggerMask();
      bool isTriggerClassFired = triggerMask & 1ul << (trigindex - 1);
      if (!isTriggerClassFired)
        return;
    }

    LOGF(info, "N. of Tracks for collision: %d", tracks.size());
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(5.0);
    // After finding the vertex, propagate tracks to the DCA. This is default anyway
    df.setPropagateToPCA(true);
    // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    df.setMaxR(200);
    // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    df.setMaxDZIni(4);
    // stop iterations if max correction is below this value. This is default anyway
    df.setMinParamChange(1e-3);
    // stop iterations if chi2 improves by less that this factor
    df.setMinRelChi2Change(0.9);

    for (auto it0 = tracks.begin(); it0 != tracks.end(); ++it0) {
      auto& track_0 = *it0;

      UChar_t clustermap_0 = track_0.itsClusterMap();
      //fill track distribution before selection
      bool isselected_0 = track_0.tpcNClsFound() > 70 && track_0.flags() & 0x4;
      isselected_0 = isselected_0 && (TESTBIT(clustermap_0, 0) || TESTBIT(clustermap_0, 1));
      if (!isselected_0)
        continue;

      float x0_ = track_0.x();
      float alpha0_ = track_0.alpha();
      std::array<float, 5> arraypar0 = {track_0.y(), track_0.z(), track_0.snp(),
                                        track_0.tgl(), track_0.signed1Pt()};
      std::array<float, 15> covpar0 = {track_0.cYY(), track_0.cZY(), track_0.cZZ(),
                                       track_0.cSnpY(), track_0.cSnpZ(),
                                       track_0.cSnpSnp(), track_0.cTglY(), track_0.cTglZ(),
                                       track_0.cTglSnp(), track_0.cTglTgl(),
                                       track_0.c1PtY(), track_0.c1PtZ(), track_0.c1PtSnp(),
                                       track_0.c1PtTgl(), track_0.c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar0(x0_, alpha0_, arraypar0, covpar0);
      for (auto it1 = it0 + 1; it1 != tracks.end(); ++it1) {
        auto& track_1 = *it1;
        UChar_t clustermap_1 = track_1.itsClusterMap();
        bool isselected_1 = track_1.tpcNClsFound() > 70 && track_1.flags() & 0x4;
        isselected_1 = isselected_1 && (TESTBIT(clustermap_1, 0) || TESTBIT(clustermap_1, 1));
        if (!isselected_1)
          continue;
        if (track_0.signed1Pt() * track_1.signed1Pt() > 0)
          continue;
        float x1_ = track_1.x();
        float alpha1_ = track_1.alpha();
        std::array<float, 5> arraypar1 = {track_1.y(), track_1.z(), track_1.snp(),
                                          track_1.tgl(), track_1.signed1Pt()};
        std::array<float, 15> covpar1 = {track_1.cYY(), track_1.cZY(), track_1.cZZ(),
                                         track_1.cSnpY(), track_1.cSnpZ(),
                                         track_1.cSnpSnp(), track_1.cTglY(), track_1.cTglZ(),
                                         track_1.cTglSnp(), track_1.cTglTgl(),
                                         track_1.c1PtY(), track_1.c1PtZ(), track_1.c1PtSnp(),
                                         track_1.c1PtTgl(), track_1.c1Pt21Pt2()};
        o2::track::TrackParCov trackparvar1(x1_, alpha1_, arraypar1, covpar1);
        df.setUseAbsDCA(true);
        int nCand = df.process(trackparvar0, trackparvar1);
        if (nCand == 0)
          continue;
        const auto& vtx = df.getPCACandidate();
        //LOGF(info, "vertex x %f", vtx[0]);
        o2::track::TrackParCov trackdec0 = df.getTrack(0);
        o2::track::TrackParCov trackdec1 = df.getTrack(1);
        std::array<float, 3> pvec0;
        std::array<float, 3> pvec1;
        trackdec0.getPxPyPzGlo(pvec0);
        trackdec1.getPxPyPzGlo(pvec1);
        float masspion = 0.140;
        float masskaon = 0.494;
        float mass_ = invmass2prongs(pvec0[0], pvec0[1], pvec0[2], masspion,
                                     pvec1[0], pvec1[1], pvec1[2], masskaon);
        float masssw_ = invmass2prongs(pvec0[0], pvec0[1], pvec0[2], masskaon,
                                       pvec1[0], pvec1[1], pvec1[2], masspion);
        secvtx2prong(track_0.collisionId(),
                     collision.posX(), collision.posY(), collision.posZ(),
                     vtx[0], vtx[1], vtx[2], track_0.globalIndex(),
                     pvec0[0], pvec0[1], pvec0[2], track_0.y(),
                     track_1.globalIndex(), pvec1[0], pvec1[1], pvec1[2], track_1.y(),
                     mass_, masssw_);
      }
    }
  }
};


WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFTrackIndexSkimsCreator>("vertexerhf-hftrackindexskimscreator")};
}
