#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Cluster.h"
#include "Framework/Logger.h"

#include <vector>
#include <cassert>
#include <algorithm>
#endif

void printTracklet(o2::its::Tracklet& tr);
void printCluster(o2::its::Cluster& cl);
void printClusters(o2::its::Cluster&, o2::its::Cluster&);

void Xcheck()
{
  TFile* old_data = TFile::Open("artefacts_old.root", "read");
  TFile* new_data = TFile::Open("artefacts_tf.root", "read");
  TTree* old_Tree_tracklets = (TTree*)old_data->Get("tracklets");
  TTree* new_Tree_tracklets = (TTree*)new_data->Get("tracklets");
  TTree* old_Tree_lines = (TTree*)old_data->Get("lines");
  TTree* new_Tree_lines = (TTree*)new_data->Get("lines");

  // Old  data Trees
  std::vector<o2::its::Tracklet>* old_tracklets01 = nullptr;
  old_Tree_tracklets->SetBranchAddress("Tracklets0", &old_tracklets01);
  std::vector<o2::its::Tracklet>* old_tracklets12 = nullptr;
  old_Tree_tracklets->SetBranchAddress("Tracklets1", &old_tracklets12);
  std::vector<o2::its::Cluster>* old_Clusters0 = nullptr;
  old_Tree_tracklets->SetBranchAddress("clusters0", &old_Clusters0);
  std::vector<o2::its::Cluster>* old_Clusters1 = nullptr;
  old_Tree_tracklets->SetBranchAddress("clusters1", &old_Clusters1);
  std::vector<o2::its::Cluster>* old_Clusters2 = nullptr;
  old_Tree_tracklets->SetBranchAddress("clusters2", &old_Clusters2);
  std::vector<o2::its::Line>* old_lines = nullptr;
  old_Tree_lines->SetBranchAddress("Lines", &old_lines);
  std::vector<int>* old_N_tracklets01 = nullptr;
  old_Tree_lines->SetBranchAddress("NTrackletCluster01", &old_N_tracklets01);
  std::vector<int>* old_N_tracklets12 = nullptr;
  old_Tree_lines->SetBranchAddress("NTrackletCluster12", &old_N_tracklets12);

  // New data Trees
  std::vector<o2::its::Tracklet>* tf_tracklets01 = nullptr;
  new_Tree_tracklets->SetBranchAddress("Tracklets0", &tf_tracklets01);
  std::vector<o2::its::Tracklet>* tf_tracklets12 = nullptr;
  new_Tree_tracklets->SetBranchAddress("Tracklets1", &tf_tracklets12);
  std::vector<o2::its::Cluster>* tf_Clusters0 = nullptr;
  new_Tree_tracklets->SetBranchAddress("clusters0", &tf_Clusters0);
  std::vector<o2::its::Cluster>* tf_Clusters1 = nullptr;
  new_Tree_tracklets->SetBranchAddress("clusters1", &tf_Clusters1);
  std::vector<o2::its::Cluster>* tf_Clusters2 = nullptr;
  new_Tree_tracklets->SetBranchAddress("clusters2", &tf_Clusters2);
  std::vector<o2::its::Line>* tf_lines = nullptr;
  new_Tree_lines->SetBranchAddress("Lines", &tf_lines);
  std::vector<int>* tf_N_tracklets01 = nullptr;
  new_Tree_lines->SetBranchAddress("NTrackletCluster01", &tf_N_tracklets01);
  std::vector<int>* tf_N_tracklets12 = nullptr;
  new_Tree_lines->SetBranchAddress("NTrackletCluster12", &tf_N_tracklets12);

  int old_entries = old_Tree_tracklets->GetEntriesFast();
  int new_entries = new_Tree_tracklets->GetEntriesFast();
  // assert(old_entries == new_entries);
  if (old_entries != new_entries) {
    LOGP(fatal, "Old entries: {} New entries: {}", old_entries, new_entries);
  }
  for (int iEntry = 0; iEntry < old_entries; iEntry++) {
    old_Tree_tracklets->GetEntry(iEntry);
    new_Tree_tracklets->GetEntry(iEntry);
    LOGP(info, "Entry {}: clusters: {} <==> {} ", iEntry, old_Clusters0->size(), tf_Clusters0->size());
    // Clusters
    if (old_Clusters0->size() != tf_Clusters0->size()) {
      LOGP(fatal, "old: N Clusters L0={}, tf: N Clusters L0={}", old_Clusters0->size(), tf_Clusters0->size());
    } else {
      std::sort(old_Clusters0->begin(), old_Clusters0->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      std::sort(tf_Clusters0->begin(), tf_Clusters0->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      for (size_t iCl{0}; iCl < old_Clusters0->size(); ++iCl) {
        if (!((*old_Clusters0)[iCl] == (*tf_Clusters0)[iCl])) {
          LOGP(info, "\nPrevious data:");
          old_Tree_tracklets->GetEntry(iEntry - 1);
          new_Tree_tracklets->GetEntry(iEntry - 1);
          printClusters((*old_Clusters0).back(), (*tf_Clusters0).back());
          old_Tree_tracklets->GetEntry(iEntry);
          new_Tree_tracklets->GetEntry(iEntry);
          LOGP(info, "\nCurrent data:");
          printClusters((*old_Clusters0)[iCl], (*tf_Clusters0)[iCl]);
          LOGP(info, "\nNext data:");
          printClusters((*old_Clusters0)[iCl + 1], (*tf_Clusters0)[iCl + 1]);

          LOGP(fatal, "Clusters L0 mismatch at position {}/{}", iCl, old_Clusters0->size());
        }
      }
    };
    if (old_Clusters1->size() != tf_Clusters1->size()) {
      LOGP(fatal, "old: N Clusters L1={}, tf: N Clusters L1={}", old_Clusters1->size(), tf_Clusters1->size());
    } else {
      std::sort(old_Clusters1->begin(), old_Clusters1->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      std::sort(tf_Clusters1->begin(), tf_Clusters1->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      for (size_t iCl{0}; iCl < old_Clusters1->size(); ++iCl) {
        if (!((*old_Clusters1)[iCl] == (*tf_Clusters1)[iCl])) {
          printCluster((*old_Clusters1)[iCl]);
          printCluster((*tf_Clusters1)[iCl]);
          LOGP(fatal, "Clusters L1 mismatch");
        }
      }
    };
    if (old_Clusters2->size() != tf_Clusters2->size()) {
      LOGP(fatal, "old: N Clusters L2={}, tf: N Clusters L2={}", old_Clusters2->size(), tf_Clusters2->size());
    } else {
      std::sort(old_Clusters2->begin(), old_Clusters2->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      std::sort(tf_Clusters2->begin(), tf_Clusters2->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      for (size_t iCl{0}; iCl < old_Clusters2->size(); ++iCl) {
        if (!((*old_Clusters2)[iCl] == (*tf_Clusters2)[iCl])) {
          printCluster((*old_Clusters2)[iCl]);
          printCluster((*tf_Clusters2)[iCl]);
          LOGP(fatal, "Clusters L2 mismatch");
        }
      }
    };
    // Tracklets
    if (old_tracklets01->size() != tf_tracklets01->size()) {
      auto max_range = std::max(old_tracklets01->size(), tf_tracklets01->size());
      auto min_range = std::min(old_tracklets01->size(), tf_tracklets01->size());
      for (size_t iTracklet{0}; iTracklet < max_range; ++iTracklet) {
        if (iTracklet < min_range) {
          LOGP(fatal, "[Tracklets]: old_idxs: {} {}, new_idxs: {} {}", (*old_tracklets01)[iTracklet].firstClusterIndex, (*old_tracklets01)[iTracklet].secondClusterIndex, (*tf_tracklets01)[iTracklet].firstClusterIndex, (*tf_tracklets01)[iTracklet].secondClusterIndex);
        } else {
          LOGP(fatal, "[Tracklets]: old_idxs: {} {}, new_idxs: - -", (*old_tracklets01)[iTracklet].firstClusterIndex, (*old_tracklets01)[iTracklet].secondClusterIndex);
        }
      }
      continue;
    }
    if (old_tracklets12->size() != tf_tracklets12->size()) {
      LOGP(fatal, "[Tracklets]: old Tracklets1 size: {} new Tracklets1 size: {}", old_tracklets12->size(), tf_tracklets12->size());
      continue;
    }
    std::sort(old_tracklets01->begin(), old_tracklets01->end(), [](o2::its::Tracklet& c1, o2::its::Tracklet& c2) { return c1.tanLambda < c2.tanLambda; });
    std::sort(tf_tracklets01->begin(), tf_tracklets01->end(), [](o2::its::Tracklet& c1, o2::its::Tracklet& c2) { return c1.tanLambda < c2.tanLambda; });
    for (size_t iTracklet{0}; iTracklet < old_tracklets01->size(); ++iTracklet) {
      if ((*old_tracklets01)[iTracklet] != (*tf_tracklets01)[iTracklet]) {
        printTracklet((*old_tracklets01)[iTracklet]);
        printTracklet((*tf_tracklets01)[iTracklet]);
        LOGP(fatal, "[Tracklets 01]: Tracklets0 mismatch at index {}", iTracklet);
      }
    }
    for (size_t iTracklet{0}; iTracklet < old_tracklets12->size(); ++iTracklet) {
      std::sort(old_tracklets12->begin(), old_tracklets12->end(), [](o2::its::Tracklet& c1, o2::its::Tracklet& c2) { return c1.tanLambda < c2.tanLambda; });
      std::sort(tf_tracklets12->begin(), tf_tracklets12->end(), [](o2::its::Tracklet& c1, o2::its::Tracklet& c2) { return c1.tanLambda < c2.tanLambda; });
      if ((*old_tracklets12)[iTracklet] != (*tf_tracklets12)[iTracklet]) {
        printTracklet((*old_tracklets12)[iTracklet]);
        printTracklet((*tf_tracklets12)[iTracklet]);
        LOGP(fatal, "[Tracklets 12]: Tracklets1 mismatch at index {}", iTracklet);
      }
    }

    // Lines
    old_Tree_lines->GetEntry(iEntry);
    new_Tree_lines->GetEntry(iEntry);
    if (old_lines->size() != tf_lines->size()) {
      LOGP(fatal, "[Lines]: Mismatch: {} {}", old_lines->size(), tf_lines->size());
    }
    sort(old_lines->begin(), old_lines->end(), [](o2::its::Line& l1, o2::its::Line& l2) { return l1.originPoint[0] < l2.originPoint[0]; });
    sort(tf_lines->begin(), tf_lines->end(), [](o2::its::Line& l1, o2::its::Line& l2) { return l1.originPoint[0] < l2.originPoint[0]; });
    for (size_t iLine{0}; iLine < old_lines->size(); ++iLine) {
      if (!((*old_lines)[iLine] == (*tf_lines)[iLine])) {
        LOGP(error, "[Lines]: mismatch at index {}/{}", iLine, old_lines->size());
        LOGP(error, "\told: x={}, y={}, z={}", (*old_lines)[iLine].originPoint[0], (*old_lines)[iLine].originPoint[1], (*old_lines)[iLine].originPoint[2]);
        LOGP(error, "\t     cx={}, cy={}, cz={}", (*old_lines)[iLine].cosinesDirector[0], (*old_lines)[iLine].cosinesDirector[1], (*old_lines)[iLine].cosinesDirector[2]);

        LOGP(error, "\tnew: x={}, y={}, z={}", (*tf_lines)[iLine].originPoint[0], (*tf_lines)[iLine].originPoint[1], (*tf_lines)[iLine].originPoint[2]);
        LOGP(error, "\t     cx={}, cy={}, cz={}", (*tf_lines)[iLine].cosinesDirector[0], (*tf_lines)[iLine].cosinesDirector[1], (*tf_lines)[iLine].cosinesDirector[2]);
      }
    }

    // NTracklets
    if (old_N_tracklets01->size() != tf_N_tracklets01->size()) {
      LOGP(fatal, "[NTracklets 01]: Mismatch: {} {}", old_N_tracklets01->size(), tf_N_tracklets01->size());
    }
    for (size_t iNtrac{0}; iNtrac < old_N_tracklets01->size(); ++iNtrac) {
      if ((*old_N_tracklets01)[iNtrac] != (*tf_N_tracklets01)[iNtrac]) {
        LOGP(fatal, "[NTracklets 01] {} <-> {}", (*old_N_tracklets01)[iNtrac], (*tf_N_tracklets01)[iNtrac]);
      }
    }

    if (old_N_tracklets12->size() != tf_N_tracklets12->size()) {
      LOGP(fatal, "[NTracklets 12]: Mismatch: {} {}", old_N_tracklets12->size(), tf_N_tracklets12->size());
    }
    for (size_t iNtrac{0}; iNtrac < old_N_tracklets12->size(); ++iNtrac) {
      if ((*old_N_tracklets12)[iNtrac] != (*tf_N_tracklets12)[iNtrac]) {
        LOGP(fatal, "[NTracklets 12] {} <-> {}", (*old_N_tracklets12)[iNtrac], (*tf_N_tracklets12)[iNtrac]);
      }
    }
  }
}

void printTracklet(o2::its::Tracklet& tr)
{
  LOGP(info, "1st Id: {}\n2nd Id: {}\ntanL: {}\nphi: {}\n------", tr.firstClusterIndex, tr.secondClusterIndex, tr.tanLambda, tr.phi);
};

void printCluster(o2::its::Cluster& cl)
{
  LOGP(info, "x: {}\ny: {}\nz: {}\nphi: {}\nr: {}\nid: {}\ntab id: {}\n------", cl.xCoordinate, cl.yCoordinate, cl.zCoordinate, cl.phi, cl.radius, cl.clusterId, cl.indexTableBinIndex);
};

void printClusters(o2::its::Cluster& cl1, o2::its::Cluster& cl2)
{
  LOGP(info, "\nx: {} <==> {} \ny: {} <==> {} \nz: {} <==> {} \nphi: {} <==> {} \nr: {} <==> {} \nid: {} <==> {} \ntab id: {} <==> {} \n------",
       cl1.xCoordinate,
       cl2.xCoordinate,
       cl1.yCoordinate,
       cl2.yCoordinate,
       cl1.zCoordinate,
       cl2.zCoordinate,
       cl1.phi,
       cl2.phi,
       cl1.radius,
       cl2.radius,
       cl1.clusterId,
       cl2.clusterId,
       cl1.indexTableBinIndex,
       cl2.indexTableBinIndex);
}