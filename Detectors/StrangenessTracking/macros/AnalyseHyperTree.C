#if !defined(CLING) || defined(ROOTCLING)
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include <TLorentzVector.h>
#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TMath.h"
#include "TString.h"
#include "TTree.h"
#endif

using GIndex = o2::dataformats::VtxTrackIndex;
using V0 = o2::dataformats::V0;
using MCTrack = o2::MCTrack;
using Cascade = o2::dataformats::Cascade;
using RRef = o2::dataformats::RangeReference<int, int>;
using VBracket = o2::math_utils::Bracket<int>;
using namespace o2::itsmft;
using Vec3 = ROOT::Math::SVector<double, 3>;

const int motherPDG = 1010010030;
const int firstDaughterPDG = 1000020030;
const int secondDaughterPDG = -211;

// const int motherPDG = 3122;
// const int firstDaughterPDG = 2212;
// const int secondDaughterPDG = -211;

o2::its::TrackITS *getITSTrack(int motherEvID, int motherTrackID, TTree *ITStree, std::vector<o2::MCCompLabel> *ITSlabel, std::vector<o2::its::TrackITS> *ITStrack);
void doMatching(const std::vector<std::vector<o2::MCTrack>> &mcTracksMatrix, TTree *treeDetectors, std::vector<o2::MCCompLabel> *labDetectors, TH1D *histo);
double calcMass(const V0 &v0, double dauMass[2], int dauCharges[2]);

void AnalyseHyperTree()
{
    auto fMCTracks = TFile::Open("sgn_1_Kine.root");
    auto fSecondaries = TFile::Open("o2_hypertrack.root");
    auto fITS = TFile::Open("o2trac_its.root");
    auto fTPC = TFile::Open("tpctracks.root");

    auto fITSTPC = TFile::Open("o2match_itstpc.root");
    auto fTPCTOF = TFile::Open("o2match_tof_tpc.root");
    auto fITSTPCTOF = TFile::Open("o2match_tof_itstpc.root");

    // Trees
    auto treeMCTracks = (TTree *)fMCTracks->Get("o2sim");
    auto treeSecondaries = (TTree *)fSecondaries->Get("o2sim");
    auto treeITS = (TTree *)fITS->Get("o2sim");
    auto treeTPC = (TTree *)fTPC->Get("tpcrec");

    auto treeITSTPC = (TTree *)fITSTPC->Get("matchTPCITS");
    auto treeITSTPCTOF = (TTree *)fITSTPCTOF->Get("matchTOF");
    auto treeTPCTOF = (TTree *)fTPCTOF->Get("matchTOF");

    // Tracks
    std::vector<o2::MCTrack> *MCtracks = nullptr;
    std::vector<V0> *v0vec = nullptr;
    std::vector<int> *ITSref = nullptr;
    std::vector<float> *chi2vec = nullptr;


    std::vector<o2::its::TrackITS> *ITStracks = nullptr;
    std::vector<o2::itsmft::ROFRecord> *rofArr = nullptr;

    // Labels
    std::vector<o2::MCCompLabel> *labITSvec = nullptr;
    std::vector<o2::MCCompLabel> *labITSTPCvec = nullptr;
    std::vector<o2::MCCompLabel> *labITSTPCTOFvec = nullptr;
    std::vector<o2::MCCompLabel> *labTPCTOFvec = nullptr;
    std::vector<o2::MCCompLabel> *labTPCvec = nullptr;

    treeSecondaries->SetBranchAddress("V0s", &v0vec);
    treeSecondaries->SetBranchAddress("ITSTrackRefs", &ITSref);
    treeSecondaries->SetBranchAddress("ITSV0Chi2", &chi2vec);


    treeMCTracks->SetBranchAddress("MCTrack", &MCtracks);
    treeITSTPC->SetBranchAddress("MatchMCTruth", &labITSTPCvec);
    treeTPCTOF->SetBranchAddress("MatchTOFMCTruth", &labTPCTOFvec);
    treeITSTPCTOF->SetBranchAddress("MatchTOFMCTruth", &labITSTPCTOFvec);
    treeITS->SetBranchAddress("ITSTrackMCTruth", &labITSvec);
    treeITS->SetBranchAddress("ITSTrack", &ITStracks);
    treeITS->SetBranchAddress("ITSTracksROF", &rofArr);
    treeTPC->SetBranchAddress("TPCTracksMCTruth", &labTPCvec);

    std::map<std::string, std::vector<o2::MCCompLabel> *>
        map{{"TPC", labTPCvec}, {"ITS", labITSvec}, {"ITS-TPC", labITSTPCvec}, {"TPC-TOF", labTPCTOFvec}, {"ITS-TPC-TOF", labITSTPCTOFvec}};

    // fill MC matrix
    int injectedParticles = 0;
    std::vector<std::vector<o2::MCTrack>> mcTracksMatrix;
    auto nev = treeMCTracks->GetEntriesFast();

    mcTracksMatrix.resize(nev);
    for (int n = 0; n < nev; n++)
    { // loop over MC events
        treeMCTracks->GetEvent(n);

        mcTracksMatrix[n].resize(MCtracks->size());
        for (unsigned int mcI{0}; mcI < MCtracks->size(); ++mcI)
        {
            mcTracksMatrix[n][mcI] = MCtracks->at(mcI);
            if (MCtracks->at(mcI).GetPdgCode() == motherPDG)
            {
                injectedParticles++;
            }
        }
    }

    treeSecondaries->GetEntry();
    treeITS->GetEntry();
    treeTPC->GetEntry();

    treeITSTPC->GetEntry();
    treeTPCTOF->GetEntry();
    treeITSTPCTOF->GetEntry();

    int counter = 0;
    for (unsigned int hTrack{0}; hTrack < v0vec->size(); hTrack++)
    {
        auto &v0 = v0vec->at(hTrack);
        auto &chi2 = chi2vec->at(hTrack);
        
        std::vector<int> motherIDvec;
        std::vector<int> daughterIDvec;
        std::vector<int> evIDvec;

        for (int iV0 = 0; iV0 < 2; iV0++)
        {
            std::cout << "---------------------------------" << std::endl;
            LOG(INFO) << "Daughter 0, Rec Pt: " << v0.getProng(0).getPt() << ", Track type: " << v0.getProngID(0).getSourceName();
            LOG(INFO) << "Daughter 1, Rec Pt: " << v0.getProng(1).getPt() << ", Track type: " << v0.getProngID(1).getSourceName();

            


            if (map[v0.getProngID(iV0).getSourceName()])
            {
                auto labTrackType = map[v0.getProngID(iV0).getSourceName()];
                auto lab = labTrackType->at(v0.getProngID(iV0).getIndex());
                // LOG(INFO) << v0.getProngID(iV0);

                int trackID, evID, srcID;
                bool fake;
                lab.get(trackID, evID, srcID, fake);
                if (!lab.isNoise() && lab.isValid() && lab.isCorrect() && srcID)
                {
                    auto motherID = mcTracksMatrix[evID][trackID].getMotherTrackId();
                    motherIDvec.push_back(motherID);
                    daughterIDvec.push_back(trackID);
                    evIDvec.push_back(evID);
                }
            }
        }



        if (motherIDvec.size() < 2)
            continue;
        if (motherIDvec[0] != motherIDvec[1] || evIDvec[0] != evIDvec[1])
            continue;
        if (motherIDvec[0] <= 0 || motherIDvec[0] > 10000)
            continue;

        int pdg0 = mcTracksMatrix[evIDvec[0]][daughterIDvec[0]].GetPdgCode();
        int pdg1 = mcTracksMatrix[evIDvec[0]][daughterIDvec[1]].GetPdgCode();

        if (pdg0 != firstDaughterPDG && pdg0 != secondDaughterPDG)
            continue;
        if (pdg1 != firstDaughterPDG && pdg1 != secondDaughterPDG)
            continue;

        auto ITSlabel = labITSvec->at(ITSref->at(hTrack));
        int ITStrackID, ITSevID, ITSsrcID;
        bool ITSfake;
        ITSlabel.get(ITStrackID, ITSevID, ITSsrcID, ITSfake);

        if(ITStrackID == daughterIDvec[0]) LOG(INFO) << "ITS TRACK == He3 TRACK";
        if(ITStrackID == daughterIDvec[1]) LOG(INFO) << "ITS TRACK == PI TRACK";
        LOG(INFO) << "Chi2: " << chi2;



        if (ITStrackID != motherIDvec[1] || ITSevID != evIDvec[0])
            continue;

        counter++;
        LOG(INFO) << "Counter: " << counter;
        LOG(INFO) << evIDvec[0] << ", " << motherIDvec[0] << ", " << motherIDvec[1];
        LOG(INFO) << "Common mother found, PDG: " << mcTracksMatrix[evIDvec[0]][motherIDvec[0]].GetPdgCode();
        LOG(INFO) << "Daughter 0, PDG: " << pdg0 << ", Pt: " << mcTracksMatrix[evIDvec[0]][daughterIDvec[0]].GetPt();
        LOG(INFO) << "Daughter 0, Rec Pt: " << v0.getProng(0).getPt() << ", Track type: " << v0.getProngID(0).getSourceName();
        LOG(INFO) << "Daughter 1, PDG: " << pdg1 << ", Pt: " << mcTracksMatrix[evIDvec[0]][daughterIDvec[1]].GetPt();
        LOG(INFO) << "Daughter 1, Rec Pt: " << v0.getProng(1).getPt() << ", Track type: " << v0.getProngID(1).getSourceName();
        auto motherTrack = mcTracksMatrix[evIDvec[0]][motherIDvec[0]];
    }
}
