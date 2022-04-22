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

void CheckVertexerArtefacts(std::string file1 = "artefacts_old.root", std::string file2 = "artefacts_tf.root")
{
  TFile* file1_data = TFile::Open(file1.data(), "read");
  TFile* file2_data = TFile::Open(file2.data(), "read");
  TTree* file1_Tree_tracklets = (TTree*)file1_data->Get("tracklets");
  TTree* file2_Tree_tracklets = (TTree*)file2_data->Get("tracklets");
  TTree* file1_Tree_lines = (TTree*)file1_data->Get("lines");
  TTree* file2_Tree_lines = (TTree*)file2_data->Get("lines");
  TTree* file1_Tree_Clulines = (TTree*)file1_data->Get("clusterlines");
  TTree* file2_Tree_Clulines = (TTree*)file2_data->Get("clusterlines");

  // file 1  data Trees
  std::vector<o2::its::Tracklet>* file1_tracklets01 = nullptr;
  file1_Tree_tracklets->SetBranchAddress("Tracklets0", &file1_tracklets01);
  std::vector<o2::its::Tracklet>* file1_tracklets12 = nullptr;
  file1_Tree_tracklets->SetBranchAddress("Tracklets1", &file1_tracklets12);
  std::vector<o2::its::Cluster>* file1_Clusters0 = nullptr;
  file1_Tree_tracklets->SetBranchAddress("clusters0", &file1_Clusters0);
  std::vector<o2::its::Cluster>* file1_Clusters1 = nullptr;
  file1_Tree_tracklets->SetBranchAddress("clusters1", &file1_Clusters1);
  std::vector<o2::its::Cluster>* file1_Clusters2 = nullptr;
  file1_Tree_tracklets->SetBranchAddress("clusters2", &file1_Clusters2);
  std::vector<o2::its::Line>* file1_lines = nullptr;
  file1_Tree_lines->SetBranchAddress("Lines", &file1_lines);
  std::vector<int>* file1_N_tracklets01 = nullptr;
  file1_Tree_lines->SetBranchAddress("NTrackletCluster01", &file1_N_tracklets01);
  std::vector<int>* file1_N_tracklets12 = nullptr;
  file1_Tree_lines->SetBranchAddress("NTrackletCluster12", &file1_N_tracklets12);
  std::vector<o2::its::ClusterLines>* file1_ClusterLines_post = nullptr;
  file1_Tree_Clulines->SetBranchAddress("cllines_post", &file1_ClusterLines_post);
  std::vector<o2::its::ClusterLines>* file1_ClusterLines_pre = nullptr;
  file1_Tree_Clulines->SetBranchAddress("cllines_pre", &file1_ClusterLines_pre);

  // file2 data Trees
  std::vector<o2::its::Tracklet>* file2_tracklets01 = nullptr;
  file2_Tree_tracklets->SetBranchAddress("Tracklets0", &file2_tracklets01);
  std::vector<o2::its::Tracklet>* file2_tracklets12 = nullptr;
  file2_Tree_tracklets->SetBranchAddress("Tracklets1", &file2_tracklets12);
  std::vector<o2::its::Cluster>* file2_Clusters0 = nullptr;
  file2_Tree_tracklets->SetBranchAddress("clusters0", &file2_Clusters0);
  std::vector<o2::its::Cluster>* file2_Clusters1 = nullptr;
  file2_Tree_tracklets->SetBranchAddress("clusters1", &file2_Clusters1);
  std::vector<o2::its::Cluster>* file2_Clusters2 = nullptr;
  file2_Tree_tracklets->SetBranchAddress("clusters2", &file2_Clusters2);
  std::vector<o2::its::Line>* file2_lines = nullptr;
  file2_Tree_lines->SetBranchAddress("Lines", &file2_lines);
  std::vector<int>* file2_N_tracklets01 = nullptr;
  file2_Tree_lines->SetBranchAddress("NTrackletCluster01", &file2_N_tracklets01);
  std::vector<int>* file2_N_tracklets12 = nullptr;
  file2_Tree_lines->SetBranchAddress("NTrackletCluster12", &file2_N_tracklets12);
  std::vector<o2::its::ClusterLines>* file2_ClusterLines_post = nullptr;
  file2_Tree_Clulines->SetBranchAddress("cllines_post", &file2_ClusterLines_post);
  std::vector<o2::its::ClusterLines>* file2_ClusterLines_pre = nullptr;
  file2_Tree_Clulines->SetBranchAddress("cllines_pre", &file2_ClusterLines_pre);

  int file1_entries = file1_Tree_tracklets->GetEntriesFast();
  int file2_entries = file2_Tree_tracklets->GetEntriesFast();
  // assert(file1_entries == file2_entries);
  if (file1_entries != file2_entries) {
    LOGP(fatal, "file 1 entries: {} file2 entries: {}", file1_entries, file2_entries);
  }
  LOGP(info, "Processing {} entries...", file1_entries);
  for (int iEntry = 0; iEntry < file1_entries; iEntry++) {
    file1_Tree_tracklets->GetEntry(iEntry);
    file2_Tree_tracklets->GetEntry(iEntry);
    // LOGP(info, "Entry {}: clusters: {} <==> {} ", iEntry, file1_Clusters0->size(), file2_Clusters0->size());
    //////////////
    // Clusters //
    //////////////
    if (file1_Clusters0->size() != file2_Clusters0->size()) {
      LOGP(fatal, "old: N Clusters L0={}, tf: N Clusters L0={}", file1_Clusters0->size(), file2_Clusters0->size());
    } else {
      // std::sort(file1_Clusters0->begin(), file1_Clusters0->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      // std::sort(file2_Clusters0->begin(), file2_Clusters0->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      for (size_t iCl{0}; iCl < file1_Clusters0->size(); ++iCl) {
        if (!((*file1_Clusters0)[iCl] == (*file2_Clusters0)[iCl])) {
          LOGP(info, "\nPrevious data:");
          file1_Tree_tracklets->GetEntry(iEntry - 1);
          file2_Tree_tracklets->GetEntry(iEntry - 1);
          printClusters((*file1_Clusters0).back(), (*file2_Clusters0).back());
          file1_Tree_tracklets->GetEntry(iEntry);
          file2_Tree_tracklets->GetEntry(iEntry);
          LOGP(info, "\nCurrent data:");
          printClusters((*file1_Clusters0)[iCl], (*file2_Clusters0)[iCl]);
          LOGP(info, "\nNext data:");
          printClusters((*file1_Clusters0)[iCl + 1], (*file2_Clusters0)[iCl + 1]);

          LOGP(fatal, "Clusters L0 mismatch at position {}/{}", iCl, file1_Clusters0->size());
        }
      }
    }
    if (file1_Clusters1->size() != file2_Clusters1->size()) {
      LOGP(fatal, "old: N Clusters L1={}, tf: N Clusters L1={}", file1_Clusters1->size(), file2_Clusters1->size());
    } else {
      // std::sort(file1_Clusters1->begin(), file1_Clusters1->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      // std::sort(file2_Clusters1->begin(), file2_Clusters1->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      for (size_t iCl{0}; iCl < file1_Clusters1->size(); ++iCl) {
        if (!((*file1_Clusters1)[iCl] == (*file2_Clusters1)[iCl])) {
          printCluster((*file1_Clusters1)[iCl]);
          printCluster((*file2_Clusters1)[iCl]);
          LOGP(fatal, "Clusters L1 mismatch");
        }
      }
    };
    if (file1_Clusters2->size() != file2_Clusters2->size()) {
      LOGP(fatal, "old: N Clusters L2={}, tf: N Clusters L2={}", file1_Clusters2->size(), file2_Clusters2->size());
    } else {
      // std::sort(file1_Clusters2->begin(), file1_Clusters2->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      // std::sort(file2_Clusters2->begin(), file2_Clusters2->end(), [](o2::its::Cluster& c1, o2::its::Cluster& c2) { return c1.xCoordinate < c2.xCoordinate; });
      for (size_t iCl{0}; iCl < file1_Clusters2->size(); ++iCl) {
        if (!((*file1_Clusters2)[iCl] == (*file2_Clusters2)[iCl])) {
          printCluster((*file1_Clusters2)[iCl]);
          printCluster((*file2_Clusters2)[iCl]);
          LOGP(fatal, "Clusters L2 mismatch");
        }
      }
    };
    ///////////////
    // Tracklets //
    ///////////////
    if (file1_tracklets01->size() != file2_tracklets01->size()) {
      auto max_range = std::max(file1_tracklets01->size(), file2_tracklets01->size());
      auto min_range = std::min(file1_tracklets01->size(), file2_tracklets01->size());
      for (size_t iTracklet{0}; iTracklet < max_range; ++iTracklet) {
        if (iTracklet < min_range) {
          LOGP(fatal, "[Tracklets]: old_idxs: {} {}, new_idxs: {} {}", (*file1_tracklets01)[iTracklet].firstClusterIndex, (*file1_tracklets01)[iTracklet].secondClusterIndex, (*file2_tracklets01)[iTracklet].firstClusterIndex, (*file2_tracklets01)[iTracklet].secondClusterIndex);
        } else {
          LOGP(fatal, "[Tracklets]: old_idxs: {} {}, new_idxs: - -", (*file1_tracklets01)[iTracklet].firstClusterIndex, (*file1_tracklets01)[iTracklet].secondClusterIndex);
        }
      }
      continue;
    }
    if (file1_tracklets12->size() != file2_tracklets12->size()) {
      LOGP(fatal, "[Tracklets]: file 1 Tracklets1 size: {} file2 Tracklets1 size: {}", file1_tracklets12->size(), file2_tracklets12->size());
      continue;
    }
    // std::sort(file1_tracklets01->begin(), file1_tracklets01->end(), [](o2::its::Tracklet& c1, o2::its::Tracklet& c2) { return c1.tanLambda < c2.tanLambda; });
    // std::sort(file2_tracklets01->begin(), file2_tracklets01->end(), [](o2::its::Tracklet& c1, o2::its::Tracklet& c2) { return c1.tanLambda < c2.tanLambda; });
    for (size_t iTracklet{0}; iTracklet < file1_tracklets01->size(); ++iTracklet) {
      if ((*file1_tracklets01)[iTracklet] != (*file2_tracklets01)[iTracklet]) {
        printTracklet((*file1_tracklets01)[iTracklet]);
        printTracklet((*file2_tracklets01)[iTracklet]);
        LOGP(fatal, "[Tracklets 01]: Tracklets0 mismatch at index {}", iTracklet);
      }
    }
    for (size_t iTracklet{0}; iTracklet < file1_tracklets12->size(); ++iTracklet) {
      // std::sort(file1_tracklets12->begin(), file1_tracklets12->end(), [](o2::its::Tracklet& c1, o2::its::Tracklet& c2) { return c1.tanLambda < c2.tanLambda; });
      // std::sort(file2_tracklets12->begin(), file2_tracklets12->end(), [](o2::its::Tracklet& c1, o2::its::Tracklet& c2) { return c1.tanLambda < c2.tanLambda; });
      if ((*file1_tracklets12)[iTracklet] != (*file2_tracklets12)[iTracklet]) {
        printTracklet((*file1_tracklets12)[iTracklet]);
        printTracklet((*file2_tracklets12)[iTracklet]);
        LOGP(fatal, "[Tracklets 12]: Tracklets1 mismatch at index {}", iTracklet);
      }
    }
    ///////////
    // Lines //
    ///////////
    file1_Tree_lines->GetEntry(iEntry);
    file2_Tree_lines->GetEntry(iEntry);
    if (file1_lines->size() != file2_lines->size()) {
      LOGP(fatal, "[Lines]: Mismatch: {} {}", file1_lines->size(), file2_lines->size());
    }
    // sort(file1_lines->begin(), file1_lines->end(), [](o2::its::Line& l1, o2::its::Line& l2) { return l1.originPoint[0] < l2.originPoint[0]; });
    // sort(file2_lines->begin(), file2_lines->end(), [](o2::its::Line& l1, o2::its::Line& l2) { return l1.originPoint[0] < l2.originPoint[0]; });
    for (size_t iLine{0}; iLine < file1_lines->size(); ++iLine) {
      if (!((*file1_lines)[iLine] == (*file2_lines)[iLine])) {
        LOGP(error, "[Lines]: mismatch at index {}/{}", iLine, file1_lines->size());
        LOGP(error, "\told: x={}, y={}, z={}", (*file1_lines)[iLine].originPoint[0], (*file1_lines)[iLine].originPoint[1], (*file1_lines)[iLine].originPoint[2]);
        LOGP(error, "\t     cx={}, cy={}, cz={}", (*file1_lines)[iLine].cosinesDirector[0], (*file1_lines)[iLine].cosinesDirector[1], (*file1_lines)[iLine].cosinesDirector[2]);

        LOGP(error, "\tnew: x={}, y={}, z={}", (*file2_lines)[iLine].originPoint[0], (*file2_lines)[iLine].originPoint[1], (*file2_lines)[iLine].originPoint[2]);
        LOGP(error, "\t     cx={}, cy={}, cz={}", (*file2_lines)[iLine].cosinesDirector[0], (*file2_lines)[iLine].cosinesDirector[1], (*file2_lines)[iLine].cosinesDirector[2]);
      }
    }

    ////////////////
    // NTracklets //
    ////////////////
    if (file1_N_tracklets01->size() != file2_N_tracklets01->size()) {
      LOGP(fatal, "[NTracklets 01]: Mismatch: {} {}", file1_N_tracklets01->size(), file2_N_tracklets01->size());
    }
    for (size_t iNtrac{0}; iNtrac < file1_N_tracklets01->size(); ++iNtrac) {
      if ((*file1_N_tracklets01)[iNtrac] != (*file2_N_tracklets01)[iNtrac]) {
        LOGP(fatal, "[NTracklets 01] {} <-> {}", (*file1_N_tracklets01)[iNtrac], (*file2_N_tracklets01)[iNtrac]);
      }
    }

    if (file1_N_tracklets12->size() != file2_N_tracklets12->size()) {
      LOGP(fatal, "[NTracklets 12]: Mismatch: {} {}", file1_N_tracklets12->size(), file2_N_tracklets12->size());
    }
    for (size_t iNtrac{0}; iNtrac < file1_N_tracklets12->size(); ++iNtrac) {
      if ((*file1_N_tracklets12)[iNtrac] != (*file2_N_tracklets12)[iNtrac]) {
        LOGP(fatal, "[NTracklets 12] {} <-> {}", (*file1_N_tracklets12)[iNtrac], (*file2_N_tracklets12)[iNtrac]);
      }
    }
    ///////////////////
    // Cluster Lines //
    ///////////////////
    file1_Tree_Clulines->GetEntry(iEntry);
    file2_Tree_Clulines->GetEntry(iEntry);
    if (file1_ClusterLines_pre->size() != file2_ClusterLines_pre->size()) {
      LOGP(fatal, "[Pre ClusterLines, rof: {}]: Mismatch {} {}", iEntry, file1_ClusterLines_pre->size(), file2_ClusterLines_pre->size());
    }
    // LOGP(info, "Rof {}: ClusterLines: {} <=> {}", iEntry, file1_ClusterLines_pre->size(), file2_ClusterLines_pre->size());
    for (size_t iClusLine{0}; iClusLine < file2_ClusterLines_pre->size(); ++iClusLine) {
      if (!((*file1_ClusterLines_pre)[iClusLine] == (*file2_ClusterLines_pre)[iClusLine])) {
        // for (auto i{0}; i < 6; ++i) {
        //   LOGP(info, "rms2: {} {}", (*file1_ClusterLines_pre)[iClusLine].getRMS2()[i], (*file2_ClusterLines_pre)[iClusLine].getRMS2()[i]);
        // }
        // for (auto i{0}; i < 3; ++i) {
        //   LOGP(info, "vertex: {} {}", (*file1_ClusterLines_pre)[iClusLine].getVertex()[i], (*file2_ClusterLines_pre)[iClusLine].getVertex()[i]);
        // }
        // return retval && this->mAvgDistance2 == rhs.mAvgDistance2;
        // LOGP(fatal, "[ClusterLines]: not equal!");
      }
    }
    if (file1_ClusterLines_post->size() != file2_ClusterLines_post->size()) {
      LOGP(fatal, "[Post ClusterLines, rof: {}]: Mismatch {} {}", iEntry, file1_ClusterLines_post->size(), file2_ClusterLines_post->size());
    }
    // LOGP(info, "Rof {}: ClusterLines: {} <=> {}", iEntry, file1_ClusterLines_post->size(), file2_ClusterLines_post->size());
    for (size_t iClusLine{0}; iClusLine < file2_ClusterLines_post->size(); ++iClusLine) {
      if (!((*file1_ClusterLines_post)[iClusLine] == (*file2_ClusterLines_post)[iClusLine])) {
        // for (auto i{0}; i < 6; ++i) {
        //   LOGP(info, "rms2: {} {}", (*file1_ClusterLines_post)[iClusLine].getRMS2()[i], (*file2_ClusterLines_post)[iClusLine].getRMS2()[i]);
        // }
        // for (auto i{0}; i < 3; ++i) {
        //   LOGP(info, "vertex: {} {}", (*file1_ClusterLines_post)[iClusLine].getVertex()[i], (*file2_ClusterLines_post)[iClusLine].getVertex()[i]);
        // }
        // return retval && this->mAvgDistance2 == rhs.mAvgDistance2;
        // LOGP(fatal, "[ClusterLines]: not equal!");
      }
    }
  }
  LOG(info) << "done.";
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