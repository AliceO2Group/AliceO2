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
#include <TH2.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TGraphErrors.h>

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

o2::MCCompLabel getMainLabel(std::vector<o2::MCCompLabel>& labs);

float range_for_vertex_fit(int NCont){return (0.03+TMath::Exp(1.11-float(NCont)*0.28))*1e+04;}
void pretty_TGraphErrors( TGraphErrors * g, int color , TString xlabel = "", TString ylabel = "", TString title = "");

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
    void access_vertex( TH2F * h2x, TH2F * h2y, TH2F * h2z );
    int id = 0;
    std::vector<int> eventIds;                                                       // ID of events in rof
    std::vector<bool> usedIds;                                                       // EvtID used to calculate actual efficiency
    std::vector<ParticleInfo> parts;                                                 // Particle usable for vertexing
    std::vector<std::vector<o2::MCCompLabel>> vertLabels;                            // Labels associated to contributors to vertex
    std::unordered_map<int, std::array<double, 3>> simVerts;                         // Simulated vertices of events that can be spot in current rof <evtId, pos[3]>
    std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>> recoVerts; // Vertices found in current ROF
    float recoeff = 0.f;                                                             // Vertexing efficiency
};

// ============================================================================================================================
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
// ============================================================================================================================
void RofInfo::access_vertex( TH2F * h2x, TH2F * h2y, TH2F * h2z )
{
    //cout << "************************* entering access_vertex function *************************" << endl;

    // Loop over simulated vertices
    for (auto& sV : simVerts) {
        double xgen = sV.second[0];
        double ygen = sV.second[1];
        double zgen = sV.second[2];

        // Loop over reconstructed vertices
        for (size_t iV{0}; iV < recoVerts.size(); ++iV) {
            auto l = getMainLabel(vertLabels[iV]);
            auto eventID = l.isSet() ? l.getEventID() : -1;
            std::cout << ">>>>>>>>> " << l << std::endl;

            // Only look at reco vertices that match event number of MC vertex
            if(eventID!=sV.first) continue;

            double xrec = recoVerts[iV].getX();
            double yrec = recoVerts[iV].getY();
            double zrec = recoVerts[iV].getZ();

            int contrib_rec = 0;
            int contrib_rec_fake = 0;

            for (auto& l : vertLabels[iV]) {
                contrib_rec++;	
                if(l.isFake())
                    contrib_rec_fake++;
            }

            // For the time being I am only looking at events without fake contributors
            if(contrib_rec_fake==0){
                double delta_vx = xrec - xgen;
                double delta_vy = yrec - ygen;
                double delta_vz = zrec - zgen;

                h2x -> Fill(contrib_rec, delta_vx*1e+04);
                h2y -> Fill(contrib_rec, delta_vy*1e+04);
                h2z -> Fill(contrib_rec, delta_vz*1e+04);
            }
        }
    }
}
// ============================================================================================================================
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

// ============================================================================================================================
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

// ============================================================================================================================
// Main function
void CheckVertices(
        const int dumprof = -1, 
        std::string path = "./",
        std::string tracfile = "o2trac_its.root",
        std::string clusfile = "o2clus_its.root",
        std::string kinefile = "o2sim_Kine.root")
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

    // -----------------------------------------------------------------
    // Declaring histograms, functions, and variables for vertex resolution determination
    const int max_num_contrib = 40;

    TH2F * h2_Ncontrib_v_delta_vx = new TH2F("h2_Ncontrib_v_delta_vx",";# contributors;delta vx (um)",max_num_contrib,0,max_num_contrib,2500,-1.5*1e+04,1.5*1e+04);
    TH2F * h2_Ncontrib_v_delta_vy = new TH2F("h2_Ncontrib_v_delta_vy",";# contributors;delta vy (um)",max_num_contrib,0,max_num_contrib,2500,-1.5*1e+04,1.5*1e+04);
    TH2F * h2_Ncontrib_v_delta_vz = new TH2F("h2_Ncontrib_v_delta_vz",";# contributors;delta vz (um)",max_num_contrib,0,max_num_contrib,2500,-1.5*1e+04,1.5*1e+04);

    // 1D Projections
    TH1F ** h1_delta_vx = new TH1F*[max_num_contrib];
    TH1F ** h1_delta_vy = new TH1F*[max_num_contrib];
    TH1F ** h1_delta_vz = new TH1F*[max_num_contrib];

    // Fit functions
    TF1 ** f1_delta_vx = new TF1*[max_num_contrib];
    TF1 ** f1_delta_vy = new TF1*[max_num_contrib];
    TF1 ** f1_delta_vz = new TF1*[max_num_contrib];

    float delta_vx_fit[max_num_contrib] = {0};
    float delta_vy_fit[max_num_contrib] = {0};
    float delta_vz_fit[max_num_contrib] = {0};

    float uncer_vx_fit[max_num_contrib] = {0};
    float uncer_vy_fit[max_num_contrib] = {0};
    float uncer_vz_fit[max_num_contrib] = {0};

    // -----------------------------------------------------------------
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

            //rof.print();
            // Fill vertex resolution histograms
            rof.access_vertex(h2_Ncontrib_v_delta_vx, h2_Ncontrib_v_delta_vy, h2_Ncontrib_v_delta_vz);
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

    // -----------------------------------------------------------------
    // Project 2D histograms for vertex resolutions
    float nContributors[max_num_contrib] = {0};

    for(int n = 0 ; n < max_num_contrib ; n++){
        h1_delta_vx[n] = (TH1F*) h2_Ncontrib_v_delta_vx -> ProjectionY(Form("h1_delta_vx_%i",n),n,n);
        h1_delta_vy[n] = (TH1F*) h2_Ncontrib_v_delta_vy -> ProjectionY(Form("h1_delta_vy_%i",n),n,n);
        h1_delta_vz[n] = (TH1F*) h2_Ncontrib_v_delta_vz -> ProjectionY(Form("h1_delta_vz_%i",n),n,n);

        nContributors[n] = float(n);
    }

    // -----------------------------------------------------------------
    // Plotting distributions related to vertex resolution
    TCanvas * c1 = new TCanvas("c1");
    c1 -> Divide(2,2);
    c1 -> cd(1); h2_Ncontrib_v_delta_vx -> Draw("COLZ");
    c1 -> cd(2); h2_Ncontrib_v_delta_vy -> Draw("COLZ");
    c1 -> cd(3); h2_Ncontrib_v_delta_vz -> Draw("COLZ");

    // ---
    TCanvas * c2 = new TCanvas("c2","c2",2500,1500);
    c2 -> Divide(8,5);
    for(int n = 0 ; n < max_num_contrib ; n++){
        c2 -> cd(n+1);
        float max_val = range_for_vertex_fit(n);
        h1_delta_vx[n] -> GetXaxis() -> SetRangeUser(-max_val,max_val);
        f1_delta_vx[n] = new TF1(Form("f1_delta_vx_%i",n),"gaus",-max_val,max_val);
        h1_delta_vx[n] -> SetTitle(Form("# contributors = %i",n));
        h1_delta_vx[n] -> Fit(Form("f1_delta_vx_%i",n));
        h1_delta_vx[n] -> Draw();
        delta_vx_fit[n] = f1_delta_vx[n] -> GetParameter(2);
        uncer_vx_fit[n] = f1_delta_vx[n] -> GetParError(2);
    }

    // ---
    TCanvas * c3 = new TCanvas("c3","c3",2500,1500);
    c3 -> Divide(8,5);
    for(int n = 0 ; n < max_num_contrib ; n++){
        c3 -> cd(n+1);
        float max_val = range_for_vertex_fit(n);
        h1_delta_vy[n] -> GetXaxis() -> SetRangeUser(-max_val,max_val);
        f1_delta_vy[n] = new TF1(Form("f1_delta_vy_%i",n),"gaus",-max_val,max_val);
        h1_delta_vy[n] -> SetTitle(Form("# contributors = %i",n));
        h1_delta_vy[n] -> Fit(Form("f1_delta_vy_%i",n));
        h1_delta_vy[n] -> Draw();
        delta_vy_fit[n] = f1_delta_vy[n] -> GetParameter(2);
        uncer_vy_fit[n] = f1_delta_vy[n] -> GetParError(2);
    }

    // ---
    TCanvas * c4 = new TCanvas("c4","c4",2500,1500);
    c4 -> Divide(8,5);
    for(int n = 0 ; n < max_num_contrib ; n++){
        c4 -> cd(n+1);
        float max_val = range_for_vertex_fit(n);
        h1_delta_vz[n] -> GetXaxis() -> SetRangeUser(-max_val,max_val);
        f1_delta_vz[n] = new TF1(Form("f1_delta_vz_%i",n),"gaus",-max_val,max_val);
        h1_delta_vz[n] -> SetTitle(Form("# contributors = %i",n));
        h1_delta_vz[n] -> Fit(Form("f1_delta_vz_%i",n));
        h1_delta_vz[n] -> Draw();
        delta_vz_fit[n] = f1_delta_vz[n] -> GetParameter(2);
        uncer_vz_fit[n] = f1_delta_vz[n] -> GetParError(2);
    }

    // -----------------------------------------------------------------

    TCanvas * c5 = new TCanvas("c5","c5",2000,1000);

    TGraphErrors * g_vtx_res_x = new TGraphErrors(max_num_contrib, nContributors, delta_vx_fit, 0, uncer_vx_fit);
    TGraphErrors * g_vtx_res_y = new TGraphErrors(max_num_contrib, nContributors, delta_vy_fit, 0, uncer_vy_fit);
    TGraphErrors * g_vtx_res_z = new TGraphErrors(max_num_contrib, nContributors, delta_vz_fit, 0, uncer_vz_fit);

    pretty_TGraphErrors(g_vtx_res_x, 64, "# contributors", "#sigma(#Deltav_{x}) [um]", "");
    pretty_TGraphErrors(g_vtx_res_y, 64, "# contributors", "#sigma(#Deltav_{y}) [um]", "");
    pretty_TGraphErrors(g_vtx_res_z, 64, "# contributors", "#sigma(#Deltav_{z}) [um]", "");

    g_vtx_res_x -> GetXaxis() -> SetRangeUser(3,max_num_contrib-1);
    g_vtx_res_y -> GetXaxis() -> SetRangeUser(3,max_num_contrib-1);
    g_vtx_res_z -> GetXaxis() -> SetRangeUser(3,max_num_contrib-1);

    g_vtx_res_x -> SetTitle("");
    g_vtx_res_y -> SetTitle("");
    g_vtx_res_z -> SetTitle("");

    c5 -> Divide(2,2);
    c5 -> cd(1); gPad -> SetLeftMargin(0.15); gPad -> SetBottomMargin(0.15);
    g_vtx_res_x -> Draw();
    c5 -> cd(2); gPad -> SetLeftMargin(0.15); gPad -> SetBottomMargin(0.15);
    g_vtx_res_y -> Draw();
    c5 -> cd(3); gPad -> SetLeftMargin(0.15); gPad -> SetBottomMargin(0.15);
    g_vtx_res_z -> Draw();

}
// ============================================================================================================================
void pretty_TGraphErrors( TGraphErrors * g, int color , TString xlabel, TString ylabel, TString title){
    g -> SetLineColor(color);
    g -> SetLineWidth(2);

    g -> GetXaxis() -> CenterTitle();
    g -> GetXaxis() -> SetTitle(xlabel);
    g -> GetXaxis() -> SetNdivisions(508);
    g -> GetXaxis() -> SetLabelSize(0.05);
    g -> GetXaxis() -> SetTitleSize(0.05);

    g -> GetYaxis() -> CenterTitle();
    g -> GetYaxis() -> SetTitle(ylabel);
    g -> GetYaxis() -> SetNdivisions(508);
    g -> GetYaxis() -> SetLabelSize(0.05);
    g -> GetYaxis() -> SetTitleSize(0.05);

}
