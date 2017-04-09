//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE ITSU Project       *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIITSUCATRACKER_H
#define ALIITSUCATRACKER_H

#include <vector>
#include <array>

#include "ITSReconstruction/CAaux.h"
#include "ITSReconstruction/CATrackingStation.h"
#include "DetectorsBase/Track.h"

namespace o2 {
  namespace ITS {
    namespace CA {
      typedef o2::Base::Track::TrackParCov TrackPC;

      class Tracker {
        public:
          Tracker(TrackingStation *stations[7]);
          // These functions must be implemented
          int Clusters2Tracks();
          int  PropagateBack();
          int  RefitInward();
          int  LoadClusters();
          void UnloadClusters();
          // Possibly, other public functions
          float    GetMaterialBudget(const double* p0, const double* p1, double& x2x0, double& rhol) const;
          bool     GetSAonly() const { return mSAonly; }
          void     SetChi2Cut(float cut) { mChi2Cut = cut; }
          void     SetPhiCut(float cut) { mPhiCut = cut; }
          void     SetSAonly(bool sa = true) { mSAonly = sa; }
          void     SetZCut(float cut) { mZCut = cut; }
          //
          float    GetX() const { return mVertex[0]; }
          float    GetY() const { return mVertex[1]; }
          float    GetZ() const { return mVertex[2]; }
          template<typename F> void SetVertex(F v[3]) { for(int i=0;i<3;++i) mVertex[i]=v[i]; }
        private:
          Tracker(const Tracker&);
          Tracker &operator=(const Tracker &tr);
          //
          bool   CellParams(int l, const Cluster& c1, const Cluster& c2, const Cluster& c3, float &curv, std::array<float,3> &np);
          void   CellsTreeTraversal(std::vector<Road> &roads, const int &iD, const int &doubl);
          void   FindTracksCA(int iteration);
          void   MakeCells(int iteration);
          bool   RefitAt(float xx, Track* t);
          void   SetCuts(int it);
          void   SetLabel(Track &t, float wrong);
          //
          TrackingStation**     mLayer;
          std::vector<bool>          mUsedClusters[7];
          float                 mChi2Cut;
          float                 mPhiCut;
          float                 mZCut;
          std::vector<Doublets>      mDoublets[6];
          std::vector<Cell>          mCells[5];
          std::vector<Track>         mCandidates[4];
          bool                  mSAonly;             // true if the standalone tracking only
          // Cuts
          float mCPhi;
          float mCDTanL;
          float mCDPhi;
          float mCZ;
          float mCDCAz[5];
          float mCDCAxy[5];
          float mCDN[4];
          float mCDP[4];
          float mCDZ[6];
          //
          float mVertex[3];
          float mBz;
          //
          static const float              mkChi2Cut;      // chi2 cut during track merging
          static const int                mkNumberOfIterations;
          static const float              mkR[7];
          //
      };
    } // namespace CA
  } // namespace ITS
} // namespace AliceO2

#endif // ALIITSUCATRACKER_H
