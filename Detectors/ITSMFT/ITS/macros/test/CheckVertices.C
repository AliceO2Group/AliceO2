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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <vector>
#include <unordered_map>

#include <TFile.h>
#include <TTree.h>

// #include "CommonUtils/RootSerializableKeyValueStore.h"
#include "Framework/Logger.h"
#include "ITSBase/GeometryTGeo.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"
#endif
#include "DataFormatsITSMFT/CompCluster.h"

o2::MCCompLabel getMainLabel(std::vector<o2::MCCompLabel>& labs);

struct ParticleInfo {
  int event;
  int pdg;
  float pt;
  float eta;
  float phi;
  int mother;
  int first;
  unsigned short clusters = 0u;
  unsigned char isReco = 0u;
  unsigned char isFake = 0u;
  bool isPrimary = 0u;
  unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
  bool canContribToVertex = false;
  std::array<int, 7> rofs = {-1, -1, -1, -1, -1, -1, -1}; /// readout frames of corresponding clusters
  o2::its::TrackITS track;
  o2::MCCompLabel lab;
};

struct RofInfo {
  void print();
  void uniqeff();
  int id = 0;
  std::vector<int> eventIds;                                                       // ID of events in rof
  std::vector<bool> usedIds;                                                       // EvtID used to calculate actual efficiency
  std::vector<ParticleInfo> parts;                                                 // Particle usable for vertexing
  std::vector<std::vector<o2::MCCompLabel>> vertLabels;                            // Labels associated to contributors to vertex
  std::unordered_map<int, std::array<double, 3>> simVerts;                         // Simulated vertices of events that can be spot in current rof <evtId, pos[3]>
  std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>> recoVerts; // Vertices found in current ROF
  float recoeff = 0.f;                                                             // Vertexing efficiency
};

void RofInfo::print()
{
  std::cout << "\n=================================== ROF " << id << " ============================================ \n";
  // Simulated vertices
  for (auto& sV : simVerts) {
    std::cout << "\tSimulated vertex for event: " << sV.first << " vertex:"
              << " x= " << sV.second[0]
              << " y= " << sV.second[1]
              << " z= " << sV.second[2]
              << std::endl;
    std::cout << "\t\tPotentially contributing tracks:\n";
    for (auto& part : parts) {
      if (part.lab.getEventID() == sV.first && part.canContribToVertex) {
        std::cout << "\t\t\t" << part.lab << "\t" << part.pt << " [GeV]\t" << part.pdg << std::endl;
      }
    }
    std::cout << std::endl;
  }

  // Reconstructed vertices
  for (size_t iV{0}; iV < recoVerts.size(); ++iV) {
    auto l = getMainLabel(vertLabels[iV]);
    auto eventID = l.isSet() ? l.getEventID() : -1;
    std::cout << "\tReconstructed vertex for event: " << eventID << " (-1: fake):"
              << " x= " << recoVerts[iV].getX()
              << " y= " << recoVerts[iV].getY()
              << " z= " << recoVerts[iV].getZ()
              << std::endl;
    std::cout << "\t\tContributor labels:\n";
    for (auto& l : vertLabels[iV]) {
      std::cout << "\t\t\t" << l << std::endl;
    }
  }

  // Efficiency
  if (simVerts.size() || recoVerts.size()) {
    std::cout << "\n\tEfficiency: " << recoeff * 100 << " %\n";
  }
}

void RofInfo::uniqeff()
{
  auto c{0};
  int current{-42};
  std::sort(parts.begin(), parts.end(), [](ParticleInfo& lp, ParticleInfo& rp) { return lp.lab.getEventID() > rp.lab.getEventID(); }); // sorting at this point should be harmless.
  for (auto& p : parts) {
    if (p.lab.getEventID() != current) {
      eventIds.push_back(p.lab.getEventID());
      current = p.lab.getEventID();
    }
  }

  usedIds.resize(eventIds.size(), false);
  for (size_t iV{0}; iV < vertLabels.size(); ++iV) {
    auto label = getMainLabel(vertLabels[iV]);
    for (size_t evId{0}; evId < eventIds.size(); ++evId) {
      if (eventIds[evId] == label.getEventID() && !usedIds[evId]) {
        usedIds[evId] = true;
        ++c;
      }
    }
  }
  recoeff = (float)c / (float)eventIds.size();
}

#pragma link C++ class ParticleInfo + ;
#pragma link C++ class RofInfo + ;

o2::MCCompLabel getMainLabel(std::vector<o2::MCCompLabel>& labs)
{
  o2::MCCompLabel lab;
  size_t max_count = 0;
  for (size_t i = 0; i < labs.size(); i++) {
    size_t count = 1;
    for (size_t j = i + 1; j < labs.size(); j++) {
      if (labs[i] == labs[j] && (labs[i].isSet() && labs[j].isSet()))
        count++;
    }
    if (count > max_count)
      max_count = count;
  }

  if (max_count == 1) { // pick first valid label in case of no majority
    for (size_t i = 0; i < labs.size(); i++) {
      if (labs[i].isSet())
        return labs[i];
    }
  }

  for (size_t i = 0; i < labs.size(); i++) {
    size_t count = 1;
    for (size_t j = i + 1; j < labs.size(); j++)
      if (labs[i] == labs[j])
        count++;
    if (count == max_count)
      lab = labs[i];
  }
  return lab;
}

void CheckVertices(const int dumprof = -1, std::string path = "tf1/", std::string tracfile = "o2trac_its.root", std::string clusfile = "o2clus_its.root", std::string kinefile = "sgn_1_Kine.root")
{
  using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
  using namespace o2::dataformats;
  using namespace o2::itsmft;
  using namespace o2::its;

  // Geometry
  o2::base::GeometryManager::loadGeometry(path.data());
  auto gman = o2::its::GeometryTGeo::Instance();

  // MC tracks and event header
  TFile* file0 = TFile::Open((path + kinefile).data());
  TTree* mcTree = (TTree*)gFile->Get("o2sim");
  mcTree->SetBranchStatus("*", 0); // disable all branches
  mcTree->SetBranchStatus("MCEventHeader*", 1);
  mcTree->SetBranchStatus("MCTrack*", 1);

  std::vector<o2::MCTrack>* mcArr = nullptr;
  mcTree->SetBranchAddress("MCTrack", &mcArr);
  MCEventHeader* eventHeader = nullptr;
  mcTree->SetBranchAddress("MCEventHeader.", &eventHeader);

  // Clusters
  TFile::Open((path + clusfile).data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);
  std::vector<o2::itsmft::ROFRecord>* clusROFRecords = nullptr;
  clusTree->SetBranchAddress("ITSClustersROF", &clusROFRecords);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  // Reconstructed vertices
  TFile* recFile = TFile::Open((path + tracfile).data());
  TTree* recTree = (TTree*)recFile->Get("o2sim");

  std::vector<Vertex>* recVerArr = nullptr;
  recTree->SetBranchAddress("Vertices", &recVerArr);
  std::vector<ROFRecord>* recVerROFArr = nullptr;
  recTree->SetBranchAddress("VerticesROF", &recVerROFArr);
  std::vector<o2::MCCompLabel>* recLabelsArr = nullptr;
  recTree->SetBranchAddress("ITSVertexMCTruth", &recLabelsArr);

  // Process
  // Fill MC info
  auto nev{mcTree->GetEntriesFast()};
  std::vector<std::vector<ParticleInfo>> info(nev);
  std::vector<std::array<double, 3>> simVerts;
  for (auto n{0}; n < nev; ++n) {
    mcTree->GetEvent(n);
    info[n].resize(mcArr->size());
    // Event header
    for (unsigned int mcI{0}; mcI < mcArr->size(); ++mcI) {
      auto part = mcArr->at(mcI);
      info[n][mcI].event = n;
      info[n][mcI].pdg = part.GetPdgCode();
      info[n][mcI].pt = part.GetPt();
      info[n][mcI].phi = part.GetPhi();
      info[n][mcI].eta = part.GetEta();
      info[n][mcI].isPrimary = part.isPrimary();
    }
    simVerts.push_back({eventHeader->GetX(), eventHeader->GetY(), eventHeader->GetZ()});
  }

  // Fill ROF info and complement MC info with cluster info
  std::vector<RofInfo> rofinfo;
  for (int frame = 0; frame < clusTree->GetEntriesFast(); frame++) { // Cluster frames
    if (!clusTree->GetEvent(frame))
      continue;
    rofinfo.resize(clusROFRecords->size());
    for (size_t rof{0}; rof < clusROFRecords->size(); ++rof) {
      for (int iClus{clusROFRecords->at(rof).getFirstEntry()}; iClus < clusROFRecords->at(rof).getFirstEntry() + clusROFRecords->at(rof).getNEntries(); ++iClus) {
        auto lab = (clusLabArr->getLabels(iClus))[0];
        if (!lab.isValid() || lab.getSourceID() != 0 || !lab.isCorrect())
          continue;

        int trackID, evID, srcID;
        bool fake;
        lab.get(trackID, evID, srcID, fake);
        if (evID < 0 || evID >= (int)info.size()) {
          std::cout << "Cluster MC label eventID out of range" << std::endl;
          continue;
        }
        if (trackID < 0 || trackID >= (int)info[evID].size()) {
          std::cout << "Cluster MC label trackID out of range" << std::endl;
          continue;
        }
        info[evID][trackID].lab = lab; // seems redundant but we are going to copy these info and loosing the nice evt/tr_id ordering
        const CompClusterExt& c = (*clusArr)[iClus];
        auto layer = gman->getLayer(c.getSensorID());
        info[evID][trackID].clusters |= 1 << layer;
        info[evID][trackID].rofs[layer] = rof;
      }
    }
  }

  for (size_t evt{0}; evt < info.size(); ++evt) {
    auto& evInfo = info[evt];
    int ntrackable{0};
    int nusable{0};
    for (auto& part : evInfo) {
      if (part.clusters & (1 << 0) && part.clusters & (1 << 1) && part.clusters & (1 << 2)) {
        ++ntrackable;
        if (part.rofs[0] > -1 && part.rofs[0] == part.rofs[1] && part.rofs[1] == part.rofs[2]) {
          ++nusable;
          part.canContribToVertex = true;
          rofinfo[part.rofs[0]].parts.push_back(part);
          int trackID, evID, srcID;
          bool fake;
          part.lab.get(trackID, evID, srcID, fake);
          rofinfo[part.rofs[0]].simVerts[evID] = simVerts[evID];
        }
      }
    }
  }

  // Reco vertices processing
  for (int frame = 0; frame < recTree->GetEntriesFast(); frame++) { // Vertices frames
    if (!recTree->GetEvent(frame)) {
      continue;
    }
    // loop on rof records
    int contLabIdx{0};
    for (size_t iRecord{0}; iRecord < recVerROFArr->size(); ++iRecord) {
      auto& rec = recVerROFArr->at(iRecord);
      auto verStartIdx = rec.getFirstEntry(), verSize = rec.getNEntries();
      int totContrib{0}, nVerts{0};
      rofinfo[iRecord].id = iRecord;
      rofinfo[iRecord].vertLabels.resize(verSize);
      int vertCounter{0};
      for (int iVertex{verStartIdx}; iVertex < verStartIdx + verSize; ++iVertex, ++vertCounter) {
        auto vert = recVerArr->at(iVertex);
        rofinfo[iRecord].recoVerts.push_back(vert);
        totContrib += vert.getNContributors();
        nVerts += 1;
        for (int ic{0}; ic < vert.getNContributors(); ++ic, ++contLabIdx) {
          rofinfo[iRecord].vertLabels[vertCounter].push_back(recLabelsArr->at(contLabIdx));
          // std::cout << "Pushed " << rofinfo[iRecord].vertLabels[vertCounter].back() << " at position " << rofinfo[iRecord].vertLabels[vertCounter].size() << std::endl;
        }
      }
    }
  }
  // Epilog
  LOGP(info, "ROF inspection summary");
  size_t nvt{0}, nevts{0}, nroffilled{0};
  float addeff{0};
  if (dumprof < 0) {
    for (size_t iROF{0}; iROF < rofinfo.size(); ++iROF) {
      auto& rof = rofinfo[iROF];
      nvt += rof.recoVerts.size();
      nevts += rof.simVerts.size();
      rof.uniqeff();
      if (rof.eventIds.size()) {
        addeff += rof.recoeff;
        nroffilled++;
      }
      rof.print();
    }
  } else {
    rofinfo[dumprof].uniqeff();
    rofinfo[dumprof].print();
    addeff += rofinfo[dumprof].recoeff;
    nvt += rofinfo[dumprof].recoVerts.size();
    nevts += rofinfo[dumprof].simVerts.size();
  }
  LOGP(info, "Summary:");
  LOGP(info, "Found {} vertices in {} usable out of {} simulated", nvt, nevts, simVerts.size());
  LOGP(info, "Average good vertexing efficiency: {}%", (addeff / (float)nroffilled) * 100);
}
