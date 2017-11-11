// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//-*- Mode: C++ -*-
// **************************************************************************
// This file is property of and copyright by the ALICE ITSU Project         *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Author: Maximiliano Puccio <maximiliano.puccio@cern.ch>          *
//                 for the ITS project                                      *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include "ITSReconstruction/CATracker.h"

// STD
#include <algorithm>
#include <cassert>
// ROOT
#include <TMath.h>
#include <Riostream.h>
// ALIROOT ITSU
#include "DetectorsBase/Track.h"

//TODO: setting Bz only once at the initialisation

using namespace o2::ITS::CA;

using TMath::Sort;
using std::sort;
using std::cout;
using std::endl;
using std::flush;
using std::vector;
using std::array;

// tolerance for layer on-surface check
const float Tracker::mkChi2Cut =  600.f;
const int Tracker::mkNumberOfIterations =  2;
const float Tracker::mkR[7] = {2.33959,3.14076,3.91924,19.6213,24.5597,34.388,39.3329};
//
const float kmaxDCAxy[5] = {0.05f,0.04f,0.05f,0.2f,0.4f};
const float kmaxDCAz[5] = {0.2f,0.4f,0.5f,0.6f,3.f};
const float kmaxDN[4] = {0.002f,0.009f,0.002f,0.005f};
const float kmaxDP[4] = {0.008f,0.0025f,0.003f,0.0035f};
const float kmaxDZ[6] = {0.1f,0.1f,0.3f,0.3f,0.3f,0.3f};
const float kDoublTanL = 0.025;
const float kDoublPhi = 0.14;

const float kmaxDCAxy1[5] = {1.f,0.4f,0.4f,1.5f,3.f};
const float kmaxDCAz1[5] = {1.f,0.4f,0.4f,1.5f,3.f};
const float kmaxDN1[4] = {0.005f,0.0035f,0.009f,0.03f};
const float kmaxDP1[4] = {0.02f,0.005f,0.006f,0.007f};
const float kmaxDZ1[6] = {1.f,1.f,1.5f,1.5f,1.5f,1.5f};
const float kDoublTanL1 = 0.05f;
const float kDoublPhi1 = 0.2f;

  Tracker::Tracker(TrackingStation* stations[7])
  :mLayer(stations)
  ,mUsedClusters()
  ,mChi2Cut(mkChi2Cut)
  ,mPhiCut(1)
  ,mZCut(0.5f)
  ,mCandidates()
  ,mSAonly(true)
  ,mCPhi()
  ,mCDTanL()
  ,mCDPhi()
  ,mCZ()
  ,mCDCAz()
  ,mCDCAxy()
  ,mCDN()
   ,mCDP()
   ,mCDZ() {
     // This default constructor needs to be provided
   }

bool Tracker::CellParams(int l, const ClsInfo_t& c1, const ClsInfo_t& c2, const ClsInfo_t& c3,
    float &curv, array<float,3> &n) {
  // Calculation of cell params and filtering using a DCA cut wrt beam line position.
  // The hit are mapped on a paraboloid space: there a circle is described as plane.
  // The parameter n of the cells is the normal vector to the plane describing the circle in the
  // paraboloid.

  // Mapping the hits
  const array<float,3> mHit0{c1.x, c1.y, c1.r * c1.r};
  const array<float,3> mHit1{c2.x, c2.y, c2.r * c2.r};
  const array<float,3> mHit2{c3.x, c3.y, c3.r * c3.r};
  // Computing the deltas
  const array<float,3> mD10{mHit1[0] - mHit0[0],mHit1[1] - mHit0[1],mHit1[2] - mHit0[2]};
  const array<float,3> mD20{mHit2[0] - mHit0[0],mHit2[1] - mHit0[1],mHit2[2] - mHit0[2]};
  // External product of the deltas -> n
  n[0] = (mD10[1] * mD20[2]) - (mD10[2] * mD20[1]);
  n[1] = (mD10[2] * mD20[0]) - (mD10[0] * mD20[2]);
  n[2] = (mD10[0] * mD20[1]) - (mD10[1] * mD20[0]);
  // Normalisation
  float norm = sqrt((n[0] * n[0]) + (n[1] * n[1]) + (n[2] * n[2]));
  if (norm < 1e-20f || fabs(n[2]) < 1e-20f)
    return false;
  else
    norm = 1.f / norm;
  n[0] *= norm;
  n[1] *= norm;
  n[2] *= norm;
  // Center of the circle
  const float c[2]{-0.5f * n[0] / n[2], -0.5f * n[1] / n[2]};
  // Constant
  const float k = - n[0] * mHit1[0] - n[1] * mHit1[1] - n[2] * mHit1[2];
  // Radius of the circle
  curv = sqrt((1.f - n[2] * n[2] - 4.f * k * n[2]) / (4.f * n[2] * n[2]));
  // Distance of closest approach to the beam line
  const float dca = fabs(curv - sqrt(c[0] * c[0] + c[1] * c[1]));
  // Cut on the DCA
  if (dca > mCDCAxy[l]) {
    return false;
  }

  curv = 1.f / curv;
  return true;
}

void Tracker::CellsTreeTraversal(vector<Road> &roads,
    const int &iD, const int &doubl) {

  // Each cells contains a list of neighbours. Each neighbour has presumably other neighbours.
  // This chain of neighbours is, as a matter of fact, a tree and this function goes recursively
  // through it. This function implements a depth first tree browsing.

  // End of the road
  if (doubl < 0) return;

  // [1] add current cell to current cell
  roads.back().AddElement(doubl,iD);
  // We want the right number of elements in the roads also in the case of multiple neighbours
  const int currentN = roads.back().N;

  // [2] loop on the neighbours of the current cell
  for (size_t iN = 0; iN < mCells[doubl][iD].NumberOfNeighbours(); ++iN) {
    const int currD = doubl - 1;
    const int neigh = mCells[doubl][iD](iN);

    // [3] for each neighbour one road
    if (iN > 0) {
      roads.push_back(roads.back());
      roads.back().N = currentN;
    }
    // [4] play this game again until the end of the road
    CellsTreeTraversal(roads,neigh,currD);
  }

  mCells[doubl][iD].SetLevel(0u); // Level = -1
}

int Tracker::Clusters2Tracks() {
  // This is the main tracking function
  // The clusters must already be loaded

  int ntrk = 0, ngood = 0;
  for (int iteration = 0; iteration < mkNumberOfIterations; ++iteration) {

    mCandidates[0].clear();
    mCandidates[1].clear();
    mCandidates[2].clear();
    mCandidates[3].clear();

    MakeCells(iteration);
    FindTracksCA(iteration);

    for (int iL = 3; iL >= 0; --iL) {
      const int nCand = mCandidates[iL].size();
      int index[nCand];
      float chi2[nCand];

      for (int iC = 0; iC < nCand; ++iC) {
        Track &tr = mCandidates[iL][iC];
        chi2[iC] = tr.GetChi2();
        index[iC] = iC;
        //CookLabel(tr,0.f);
      }

      TMath::Sort(nCand,chi2,index,false);

      for (int iUC = 0; iUC < nCand; ++iUC) {
        const int iC = index[iUC];
        if (chi2[iC] < 0.f) {
          continue;
        }

        Track* tr = &mCandidates[iL][iC];
        int* idx = tr->Clusters();
        bool sharingCluster = false;
        for (int k = 0; k < 7; ++k) {
          if (idx[k] > -1) {
            if (mUsedClusters[k][idx[k]]) {
              sharingCluster = true;
              break;
            }
          }
        }

        if (sharingCluster)
          continue;

        for (int k = 0; k < 7; ++k) {
          if (idx[k] > -1)
            mUsedClusters[k][idx[k]] = true;
        }

        /*        AliESDtrack outTrack;
                  CookLabel(tr,0.f);
                  ntrk++;
                  if(tr->GetLabel() >= 0) {
                  ngood++;
                  }

                  outTrack.UpdateTrackParams(tr,AliESDtrack::kITSin);
                  outTrack.SetLabel(tr->GetLabel());
                  if (mSAonly) outTrack.SetStatus(AliESDtrack::kITSpureSA);
                  event->AddTrack(&outTrack);*/
      }
    }
  }
  //Info("Clusters2Tracks","Reconstructed tracks: %d",ntrk);
  //if (ntrk)
  //  Info("Clusters2Tracks","Good tracks/reconstructed: %f",float(ngood)/ntrk);
  //
  return 0;
}

void Tracker::FindTracksCA(int iteration) {
  // Main pattern recognition routine. It has 4 steps (planning to split in different methods)
  // 1. Tracklet finding (using vertex position)
  // 2. Tracklet association, cells building
  // 3. Handshake between neighbour cells
  // 4. Candidates ("roads") finding and fitting

  // Road finding and fitting. The routine starts from cells with level 5 since they are the head
  // of a full road (candidate with 7 points). Minimum level is 2, so candidates at the end have
  // at least 4 points.
  const int itLevelLimit[3] = {4, 4, 1};
  for (int level = 5; level > itLevelLimit[iteration]; --level) {
    vector<Road> roads;
    // Road finding. For each cell at level $(level) a loop on their neighbours to start building
    // the roads.
    for (int iCL = 4; iCL >= level - 1; --iCL) {
      for (size_t iCell = 0; iCell < mCells[iCL].size(); ++iCell) {
        if (mCells[iCL][iCell].GetLevel() != level)
          continue;
        // [1] Add current cell to road
        roads.emplace_back(iCL,iCell);
        // [2] Loop on current cell neighbours
        for(size_t iN = 0; iN < mCells[iCL][iCell].NumberOfNeighbours(); ++iN) {
          const int currD = iCL - 1;
          const int neigh = mCells[iCL][iCell](iN);
          // [3] if more than one neighbour => more than one road, one road for each neighbour
          if(iN > 0) {
            roads.emplace_back(iCL,iCell);
          }
          // [4] Essentially the neighbour became the current cell and then go to [1]
          CellsTreeTraversal(roads,neigh,currD);
        }
        mCells[iCL][iCell].SetLevel(0u); // Level = -1
      }
    }

    // Roads fitting
    for (size_t iR = 0; iR < roads.size(); ++iR) {
      if (roads[iR].N != level)
        continue;
      int indices[7] = {-1};
      int first = -1,last = -1;
      for(int i = 0; i < 5; ++i) {
        if (roads[iR][i] < 0)
          continue;

        if (first < 0) {
          indices[i] = mCells[i][roads[iR][i]].x();
          indices[i + 1] = mCells[i][roads[iR][i]].y();
          first = i;
        }
        indices[i + 2] = mCells[i][roads[iR][i]].z();
        last = i;
      }
      const int mid = (last + first) / 2;
      const Cluster& cl0 = (*mLayer[first])[mCells[first][roads[iR][first]].x()];
      const Cluster& cl1 = (*mLayer[mid + 1])[mCells[mid][roads[iR][mid]].y()];
      const Cluster& cl2 = (*mLayer[last + 2])[mCells[last][roads[iR][last]].z()];
      // Init track parameters
      float cv  = Curvature(cl0.x,cl0.y,cl1.x,cl1.y,cl2.x,cl2.y);
      float tgl = TanLambda(cl0.x,cl0.y,cl2.x,cl2.y,cl0.z,cl2.z);

      ITSDetInfo_t det = (*mLayer[last + 2]).GetDetInfo(cl2.detid);
      float x = det.xTF + cl2.x; // I'd like to avoit using AliITSUClusterPix...
      float alp = det.phiTF;
      std::array<float,5> par {cl2.y,cl2.z,0,tgl,cv};
      std::array<float,15> cov {
        5.f*5.f,
        0.f,  5.f*5.f,
        0.f,  0.f  , 0.7f*0.7f,
        0.f,  0.f,   0.f,       0.7f*0.7f,
        0.f,  0.f,   0.f,       0.f,       10.f
      };
      Track tt{x,alp,par,cov,indices};
      if (RefitAt(2.1, &tt))
        mCandidates[level - 2].push_back(tt);
    }
  }
}

int Tracker::LoadClusters() {
  // This function reads the ITSU clusters from the tree,
  // sort them, distribute over the internal tracker arrays, etc

  // I consider a single vertex event for the moment.
  //TODO: pile-up (trivial here), fast reco of primary vertices (not so trivial)
  for (int iL = 0; iL < 7; ++iL) {
    mLayer[iL]->SortClusters(mVertex);
    mUsedClusters[iL].resize(mLayer[iL]->GetNClusters(),false);
  }
  return 0;
}

void Tracker::MakeCells(int iteration) {

  SetCuts(iteration);
  if (iteration >= 1) {
    for (int i = 0; i < 5; ++i)
      vector<Cell>().swap(mCells[i]);
    for (int i = 0; i < 6; ++i)
      vector<Doublets>().swap(mDoublets[i]);
  }

  // Trick to speed up the navigation of the doublets array. The lookup table is build like:
  // dLUT[l][i] = n;
  // where n is the index inside mDoublets[l+1] of the first doublets that uses the point
  // mLayer[l+1][i]
  vector<int> dLUT[5];
  for (int iL = 0; iL < 6; ++iL) {
    if (mLayer[iL]->GetNClusters() == 0) continue;
    if (iL < 5)
      dLUT[iL].resize((*mLayer[iL + 1]).GetNClusters(),-1);
    if (dLUT[iL - 1].size() == 0u)
      continue;
    for (int iC = 0; iC < mLayer[iL]->GetNClusters(); ++iC) {
      const ClsInfo_t& cls = mLayer[iL]->GetClusterInfo(iC);
      if (mUsedClusters[iL][iC]) {
        continue;
      }
      const float tanL = (cls.z - GetZ()) / cls.r;
      const float extz = tanL * (mkR[iL + 1] - cls.r) + cls.z;
      const int nClust = mLayer[iL + 1]->SelectClusters(extz - 2 * mCZ, extz + 2 * mCZ,
          cls.phi - mCPhi, cls.phi + mCPhi);
      bool first = true;

      for (int iC2 = 0; iC2 < nClust; ++iC2) {
        const int iD2 = mLayer[iL + 1]->GetNextClusterInfoID();
        const ClsInfo_t& cls2 = mLayer[iL + 1]->GetClusterInfo(iD2);
        if (mUsedClusters[iL + 1][iC2]) {
          continue;
        }
        const float dz = tanL * (cls2.r - cls.r) + cls.z - cls2.z;
        if (fabs(dz) < mCDZ[iL] && CompareAngles(cls.phi, cls2.phi, mCPhi)) {
          if (first && iL > 0) {
            dLUT[iL - 1][iC] = mDoublets[iL].size();
            first = false;
          }
          const float dTanL = (cls.z - cls2.z) / (cls.r - cls2.r);
          const float phi = atan2(cls.y - cls2.y, cls.x - cls2.x);
          mDoublets[iL].emplace_back(iC,iD2,dTanL,phi);
        }
      }
      mLayer[iL + 1]->ResetFoundIterator();
    }
  }

  // Trick to speed up the navigation of the cells array. The lookup table is build like:
  // tLUT[l][i] = n;
  // where n is the index inside mCells[l+1] of the first cells that uses the doublet
  // mDoublets[l+1][i]
  vector<int> tLUT[4];
  tLUT[0].resize(mDoublets[1].size(),-1);
  tLUT[1].resize(mDoublets[2].size(),-1);
  tLUT[2].resize(mDoublets[3].size(),-1);
  tLUT[3].resize(mDoublets[4].size(),-1);

  for (int iD = 0; iD < 5; ++iD) {
    if (mDoublets[iD + 1].size() == 0u || mDoublets[iD].size() == 0u) continue;

    for (size_t iD0 = 0; iD0 < mDoublets[iD].size(); ++iD0) {
      const int idx = mDoublets[iD][iD0].y;
      bool first = true;
      if (dLUT[iD][idx] == -1) continue;
      for (size_t iD1 = dLUT[iD][idx]; iD1 < mDoublets[iD + 1].size(); ++iD1) {
        if (idx != mDoublets[iD + 1][iD1].x) break;
        if (fabs(mDoublets[iD][iD0].tanL - mDoublets[iD + 1][iD1].tanL) < mCDTanL &&
            fabs(mDoublets[iD][iD0].phi - mDoublets[iD + 1][iD1].phi) < mCDPhi) {
          const float tan = 0.5f * (mDoublets[iD][iD0].tanL + mDoublets[iD + 1][iD1].tanL);
          const float extz = -tan * (*mLayer[iD])[mDoublets[iD][iD0].x].r +
            (*mLayer[iD])[mDoublets[iD][iD0].x].z;
          if (fabs(extz - GetZ()) < mCDCAz[iD]) {
            float curv = 0.f;
            array<float,3> n {0.f};
            if (CellParams(iD,(*mLayer[iD])[mDoublets[iD][iD0].x],(*mLayer[iD + 1])[mDoublets[iD][iD0].y],
                  (*mLayer[iD + 2])[mDoublets[iD + 1][iD1].y],curv,n)) {
              if (first && iD > 0) {
                tLUT[iD - 1][iD0] = mCells[iD].size();
                first = false;
              }
              mCells[iD].emplace_back(mDoublets[iD][iD0].x,mDoublets[iD][iD0].y,
                    mDoublets[iD + 1][iD1].y,iD0,iD1,curv,n);
            }
          }
        }
      }
    }
  }

  // Adjacent cells: cells that share 2 points. In the following code adjacent cells are combined.
  // If they meet some requirements (~ same curvature, ~ same n) the innermost cell id is added
  // to the list of neighbours of the outermost cell. When the cell is added to the neighbours of
  // the outermost cell the "level" of the latter is set to the level of the innermost one + 1.
  // ( only if $(level of the innermost) + 1 > $(level of the outermost) )
  for (int iD = 0; iD < 4; ++iD) {
    if (mCells[iD + 1].size() == 0u || tLUT[iD].size() == 0u) continue; // TODO: dealing with holes
    for (size_t c0 = 0; c0 < mCells[iD].size(); ++c0) {
      const int idx = mCells[iD][c0].d1();
      if (tLUT[iD][idx] == -1) continue;
      for (size_t c1 = tLUT[iD][idx]; c1 < mCells[iD + 1].size(); ++c1) {
        if (idx != mCells[iD + 1][c1].d0()) break;
        auto& n0 = mCells[iD][c0].GetN();
        auto& n1 = mCells[iD + 1][c1].GetN();
        const float dn2 = ((n0[0] - n1[0]) * (n0[0] - n1[0]) + (n0[1] - n1[1]) * (n0[1] - n1[1]) +
            (n0[2] - n1[2]) * (n0[2] - n1[2]));
        const float dp = fabs(mCells[iD][c0].GetCurvature() - mCells[iD + 1][c1].GetCurvature());
        if (dn2 < mCDN[iD] && dp < mCDP[iD]) {
          mCells[iD + 1][c1].Combine(mCells[iD][c0], c0);
        }
      }
    }
  }
}

int Tracker::PropagateBack() {

  /*int n=event->GetNumberOfTracks();
    int ntrk=0;
    int ngood=0;
    for (int i=0; i<n; i++) {
    AliESDtrack *esdTrack=event->GetTrack(i);

    if ((esdTrack->GetStatus()&AliESDtrack::kITSin)==0) continue;
    if (esdTrack->IsOn(AliESDtrack::kTPCin)) continue; //skip a TPC+ITS track

    AliITSUTrackCooked track(*esdTrack);
    AliITSUTrackCooked temp(*esdTrack);

    temp.ResetCovariance(10.);
    temp.ResetClusters();

    if (RefitAt(40., &temp, &track)) {

    CookLabel(&temp, 0.); //For comparison only
    int label = temp.GetLabel();
    if (label > 0) ngood++;

    esdTrack->UpdateTrackParams(&temp,AliESDtrack::kITSout);
    ntrk++;
    }
    }*/

  //Info("PropagateBack","Back propagated tracks: %d",ntrk);
  //if (ntrk)
  //  Info("PropagateBack","Good tracks/back propagated: %f",float(ngood)/ntrk);

  return 0;
}

bool Tracker::RefitAt(float xx, Track *track) {
  // This function refits the track "t" at the position "x" using
  // the clusters from "c"

  const int nLayers = 7;
  int* index = track->Clusters();

  int from, to, step;
  if (xx > track->getX()) {
    from = 0;
    to = nLayers;
    step = +1;
  } else {
    from = nLayers - 1;
    to = -1;
    step = -1;
  }

  for (int i = from; i != to; i += step) {
    int idx = index[i];
    if (idx >= 0) {
      const Cluster &cl = (*mLayer[i])[idx];
      float xr = cl.x, ar = mLayer[i]->GetDetInfo(cl.detid).phiTF;
      if (!track->rotate(ar) || !track->propagateTo(xr, mBz)) {
        return false;
      }
      track->Update(cl);
    } else {
      float r = mkR[i];
      float phi,z;
      if (!track->GetPhiZat(r,mBz,phi,z)) {
        return false;
      }
      if (!track->rotate(phi) || !track->propagateTo(r, mBz)) {
        return false;
      }
    }
    float xx0 = (i > 2) ? 0.008 : 0.003;  // Rough layer thickness
    float x0  = 9.36; // Radiation length of Si [cm]
    float rho = 2.33; // Density of Si [g/cm^3]
    float mass = 1.39569997787475586e-01; // pion mass
    track->correctForMaterial(xx0, - step * xx0 * x0 * rho, mass, true);
  }

  if (!track->propagateTo(xx,mBz)) return false;
  return true;
}

int Tracker::RefitInward() {
  // Some final refit, after the outliers get removed by the smoother ?
  // The clusters must be loaded

  /*int n = event->GetNumberOfTracks();
    int ntrk = 0;
    int ngood = 0;
    for (int i = 0; i < n; i++) {
    AliESDtrack *esdTrack = event->GetTrack(i);

    if ((esdTrack->GetStatus() & AliESDtrack::kITSout) == 0) continue;
    if (esdTrack->IsOn(AliESDtrack::kTPCin)) continue; //skip a TPC+ITS track
    AliITSUTrackCooked track(*esdTrack);
    AliITSUTrackCooked temp(*esdTrack);

    temp.ResetCovariance(10.);
    temp.ResetClusters();

    if (!RefitAt(2.1, &temp, &track)) continue;
  //Cross the beam pipe
  if (!temp.PropagateTo(1.8, 2.27e-3, 35.28 * 1.848)) continue;

  CookLabel(&temp, 0.); //For comparison only
  int label = temp.GetLabel();
  if (label > 0) ngood++;

  esdTrack->UpdateTrackParams(&temp,AliESDtrack::kITSrefit);
  ntrk++;
  }*/

  //Info("RefitInward","Refitted tracks: %d",ntrk);
  //if (ntrk)
  //  Info("RefitInward","Good tracks/refitted: %f",float(ngood)/ntrk);

  return 0;
}

void Tracker::SetCuts(int it) {
  switch (it) {
    case 0:
      mCPhi = mPhiCut;
      mCDTanL = kDoublTanL;
      mCDPhi = kDoublPhi;
      mCZ = mZCut;
      for (int i = 0; i < 5; ++i) {
        mCDCAxy[i] = kmaxDCAxy[i];
        mCDCAz[i] = kmaxDCAz[i];
      }
      for (int i = 0; i < 4; ++i) {
        mCDN[i] = kmaxDN[i];
        mCDP[i] = kmaxDP[i];
      }
      for (int i = 0; i < 6; ++i) {
        mCDZ[i] = kmaxDZ[i];
      }
      break;

    default:
      mCPhi = 3.f * mPhiCut;
      mCDTanL = kDoublTanL1;
      mCDPhi = kDoublPhi1;
      mCZ = mZCut;
      for (int i = 0; i < 5; ++i) {
        mCDCAxy[i] = kmaxDCAxy1[i];
        mCDCAz[i] = kmaxDCAz1[i];
      }
      for (int i = 0; i < 4; ++i) {
        mCDN[i] = kmaxDN1[i];
        mCDP[i] = kmaxDP1[i];
      }
      for (int i = 0; i < 6; ++i) {
        mCDZ[i] = kmaxDZ1[i];
      }
      break;
  }
}

void Tracker::UnloadClusters() {
  /// This function unloads ITSU clusters from the memory
  for (int i = 0;i < 7;++i)
    mUsedClusters[i].clear();
  for (int i = 0; i < 6; ++i)
    mDoublets[i].clear();
  for (int i = 0; i < 5; ++i)
    mCells[i].clear();
  for (int i = 0; i < 4; ++i)
    mCandidates[i].clear();
}

