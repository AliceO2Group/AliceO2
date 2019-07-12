//
// Created by jmy on 09.07.19.
//
//#include "EventVisualisationView/MultiView.h"
#include "EventVisualisationBase/DataSourceOfflineITS.h"


//


#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "ITSMFTReconstruction/RawPixelReader.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTBase/Digit.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "TGenericClassInfo.h"

#include <TEveElement.h>




#include <iostream>
#include <array>
#include <algorithm>
#include <fstream>

#include <TFile.h>
#include <TTree.h>
#include <TEveManager.h>
#include <TEveBrowser.h>
#include <TGButton.h>
#include <TGNumberEntry.h>
#include <TGFrame.h>
#include <TGTab.h>
#include <TGLCameraOverlay.h>
#include <TEveFrameBox.h>
#include <TEveQuadSet.h>
#include <TEveTrans.h>
#include <TEvePointSet.h>
#include <TEveTrackPropagator.h>
#include <TEveTrack.h>
#include <Rtypes.h>
#include <gsl/span>

#include "ITSMFTReconstruction/DigitPixelReader.h"



class ITSData {
    o2::itsmft::PixelReader* mPixelReader = nullptr;
    o2::InteractionRecord mIR;                          // interraction record
    o2::itsmft::ChipPixelData mChipData;                // chip data

    TTree* mClusTree = nullptr;
    TTree* mTracTree = nullptr;
    std::vector<o2::itsmft::ROFRecord> mClustersROF;
    std::vector<o2::itsmft::Cluster>* mClusterBuffer = nullptr;

    std::vector<o2::itsmft::ROFRecord> mTracksROF;
    std::vector<o2::its::TrackITS>* mTrackBuffer = nullptr;

    int mLastEvent = 0;
public:
    std::vector<o2::itsmft::Digit> mDigits;
    gsl::span<o2::itsmft::Cluster> mClusters;
    gsl::span<o2::its::TrackITS> mTracks;

    ITSData() {}
    int getLastEvent() const { return mLastEvent; }
    void setDigitPixelReader(TString input)
    {
        auto reader = new o2::itsmft::DigitPixelReader();
        reader->openInput(input.Data(), o2::detectors::DetID("ITS"));
        reader->init();
        reader->readNextEntry();
        mPixelReader = dynamic_cast<o2::itsmft::PixelReader*>(reader);                    // incompatible class DigitPixelReader : public PixelReader
        assert(mPixelReader != nullptr);
        mPixelReader->getNextChipData(mChipData);
        mIR = mChipData.getInteractionRecord();
    }
    void setClusTree(TTree* tree)
    {
        if (tree == nullptr) {
            std::cerr << "No tree for clusters !\n";
            return;
        }
        tree->SetBranchAddress("ITSCluster", &mClusterBuffer);
        mClusTree = tree;

        TTree* roft = (TTree*)gFile->Get("ITSClustersROF");
        if (roft != nullptr) {
            std::vector<o2::itsmft::ROFRecord>* roFrames = &mClustersROF;
            roft->SetBranchAddress("ITSClustersROF", &roFrames);
            roft->GetEntry(0);
        }
    }
    void setTracTree(TTree* tree)
    {
        if (tree == nullptr) {
            std::cerr << "No tree for tracks !\n";
            return;
        }
        tree->SetBranchAddress("ITSTrack", &mTrackBuffer);
        mTracTree = tree;

        TTree* roft = (TTree*)gFile->Get("ITSTracksROF");
        if (roft != nullptr) {
            std::vector<o2::itsmft::ROFRecord>* roFrames = &mTracksROF;
            roft->SetBranchAddress("ITSTracksROF", &roFrames);
            roft->GetEntry(0);
        }
    }
    void loadData(int entry)
    {
        loadDigits(entry);
        loadClusters(entry);
        loadTracks(entry);
    }
    void loadDigits()
    {
        auto ir = mChipData.getInteractionRecord();
        std::cout << "orbit/crossing: " << ' ' << ir.orbit << '/' << ir.bc << '\n';

        mDigits.clear();

        do {
            auto chipID = mChipData.getChipID();
            auto pixels = mChipData.getData();
            for (auto& pixel : pixels) {
                auto col = pixel.getCol();
                auto row = pixel.getRow();
                mDigits.emplace_back(chipID, 0, row, col);
            }
            if (!mPixelReader->getNextChipData(mChipData))
                return;
            ir = mChipData.getInteractionRecord();
        } while (mIR == ir);
        mIR = ir;

        std::cout << "Number of ITSDigits: " << mDigits.size() << '\n';
    }

    void loadDigits(int entry)
    {
        if (mPixelReader == nullptr)
            return;

        for (; mLastEvent < entry; mLastEvent++) {
            auto ir = mChipData.getInteractionRecord();
            do {
                if (!mPixelReader->getNextChipData(mChipData))
                    return;
                ir = mChipData.getInteractionRecord();
            } while (mIR == ir);
            mIR = ir;
        }
        mLastEvent++;
        loadDigits();
    }
    void loadClusters(int entry)
    {
        static int lastLoaded = -1;

        if (mClusTree == nullptr)
            return;

        auto event = entry; // If no RO frame informaton available, assume one entry per a RO frame.
        if (!mClustersROF.empty()) {
            if ((event < 0) || (event >= (int)mClustersROF.size())) {
                std::cerr << "Clusters: Out of event range ! " << event << '\n';
                return;
            }
            auto rof = mClustersROF[entry];
            event = rof.getROFEntry().getEvent();
        }
        if ((event < 0) || (event >= mClusTree->GetEntries())) {
            std::cerr << "Clusters: Out of event range ! " << event << '\n';
            return;
        }
        if (event != lastLoaded) {
            mClusterBuffer->clear();
            mClusTree->GetEntry(event);
            lastLoaded = event;
        }

        int first = 0, last = mClusterBuffer->size();
        if (!mClustersROF.empty()) {
            auto rof = mClustersROF[entry];
            first = rof.getROFEntry().getIndex();
            last = first + rof.getNROFEntries();
        }
        mClusters = gsl::make_span(&(*mClusterBuffer)[first], last - first);

        std::cout << "Number of ITSClusters: " << mClusters.size() << '\n';
    }
    void loadTracks(int entry)
    {
        static int lastLoaded = -1;

        if (mTracTree == nullptr)
            return;

        auto event = entry; // If no RO frame informaton available, assume one entry per a RO frame.
        if (!mTracksROF.empty()) {
            if ((event < 0) || (event >= (int)mTracksROF.size())) {
                std::cerr << "Clusters: Out of event range ! " << event << '\n';
                return;
            }
            auto rof = mTracksROF[entry];
            event = rof.getROFEntry().getEvent();
        }
        if ((event < 0) || (event >= mTracTree->GetEntries())) {
            std::cerr << "Tracks: Out of event range ! " << event << '\n';
            return;
        }
        if (event != lastLoaded) {
            mTrackBuffer->clear();
            mTracTree->GetEntry(event);
            lastLoaded = event;
        }

        int first = 0, last = mTrackBuffer->size();
        if (!mTracksROF.empty()) {
            auto rof = mTracksROF[entry];
            first = rof.getROFEntry().getIndex();
            last = first + rof.getNROFEntries();
        }
        mTracks = gsl::make_span(&(*mTrackBuffer)[first], last - first);

        std::cout << "Number of ITSTracks: " << mTracks.size() << '\n';
    }

} its_data;


class ITS_Display {
    static constexpr float sizey = o2::itsmft::SegmentationAlpide::ActiveMatrixSizeRows;
    static constexpr float sizex = o2::itsmft::SegmentationAlpide::ActiveMatrixSizeCols;
    static constexpr float dy = o2::itsmft::SegmentationAlpide::PitchRow;
    static constexpr float dx = o2::itsmft::SegmentationAlpide::PitchCol;
    static constexpr float gap = 1e-4; // For a better visualization of pix

    // TEve-related members
    TEveElementList* mEvent = nullptr;
    TEveElementList* mChip = nullptr;

    TEveElement* getEveChipDigits(int chip, std::vector<o2::itsmft::Digit>& mDigits)
    {
        static TEveFrameBox* box = new TEveFrameBox();
        box->SetAAQuadXY(0, 0, 0, sizex, sizey);
        box->SetFrameColor(kGray);

        // Digits
        TEveQuadSet* qdigi = new TEveQuadSet("digits");
        qdigi->SetOwnIds(kTRUE);
        qdigi->SetFrame(box);
        qdigi->Reset(TEveQuadSet::kQT_RectangleXY, kFALSE, 32);
        auto gman = o2::its::GeometryTGeo::Instance();
        std::vector<int> occup(gman->getNumberOfChips());
        for (const auto& d : mDigits) {
            auto id = d.getChipIndex();
            occup[id]++;
            if (id != chip)
                continue;

            int row = d.getRow();
            int col = d.getColumn();
            int charge = d.getCharge();
            qdigi->AddQuad(col * dx + gap, row * dy + gap, 0., dx - 2 * gap, dy - 2 * gap);
            qdigi->QuadValue(charge);
        }
        qdigi->RefitPlex();
        TEveTrans& t = qdigi->RefMainTrans();
        t.RotateLF(1, 3, 0.5 * TMath::Pi());
        t.SetPos(0, 0, 0);

        auto most = std::distance(occup.begin(), std::max_element(occup.begin(), occup.end()));
        std::cout << "Most occupied chip: " << most << " (" << occup[most] << " digits)\n";
        std::cout << "Chip " << chip << " number of digits " << occup[chip] << '\n';

        return qdigi;
    }

    TEveElement* getEveChipClusters(int chip, gsl::span<o2::itsmft::Cluster>& mClusters)
    {
        static TEveFrameBox* box = new TEveFrameBox();
        box->SetAAQuadXY(0, 0, 0, sizex, sizey);
        box->SetFrameColor(kGray);

        // Clusters
        TEveQuadSet* qclus = new TEveQuadSet("clusters");
        qclus->SetOwnIds(kTRUE);
        qclus->SetFrame(box);
        int ncl = 0;
        qclus->Reset(TEveQuadSet::kQT_LineXYFixedZ, kFALSE, 32);
        for (const auto& c : mClusters) {
            auto id = c.getSensorID();
            if (id != chip)
                continue;
            ncl++;

            int row = c.getPatternRowMin();
            int col = c.getPatternColMin();
            int len = c.getPatternColSpan();
            int wid = c.getPatternRowSpan();
            qclus->AddLine(col * dx, row * dy, len * dx, 0.);
            qclus->AddLine(col * dx, row * dy, 0., wid * dy);
            qclus->AddLine((col + len) * dx, row * dy, 0., wid * dy);
            qclus->AddLine(col * dx, (row + wid) * dy, len * dx, 0.);
        }
        qclus->RefitPlex();
        TEveTrans& ct = qclus->RefMainTrans();
        ct.RotateLF(1, 3, 0.5 * TMath::Pi());
        ct.SetPos(0, 0, 0);

        std::cout << "Chip " << chip << " number of clusters " << ncl << '\n';

        return qclus;
    }

    TEveElement* getEveClusters(gsl::span<o2::itsmft::Cluster>& mClusters)
    {
        auto gman = o2::its::GeometryTGeo::Instance();
        TEvePointSet* clusters = new TEvePointSet("clusters");
        clusters->SetMarkerColor(kBlue);
        for (const auto& c : mClusters) {
            const auto& gloC = c.getXYZGloRot(*gman);
            clusters->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
        }
        return clusters;
    }

    TEveElement* getEveTracks(gsl::span<o2::its::TrackITS>& mTracks, gsl::span<o2::itsmft::Cluster>& mClusters)
    {
        auto gman = o2::its::GeometryTGeo::Instance();
        TEveTrackList* tracks = new TEveTrackList("tracks");
        auto prop = tracks->GetPropagator();
        prop->SetMagField(0.5);
        prop->SetMaxR(50.);
        for (const auto& rec : mTracks) {
            std::array<float, 3> p;
            rec.getPxPyPzGlo(p);
            TEveRecTrackD t;
            t.fP = { p[0], p[1], p[2] };
            t.fSign = (rec.getSign() < 0) ? -1 : 1;
            TEveTrack* track = new TEveTrack(&t, prop);
            track->SetLineColor(kMagenta);
            tracks->AddElement(track);

            if (mClusters.empty())
                continue;
            TEvePointSet* tpoints = new TEvePointSet("tclusters");
            tpoints->SetMarkerColor(kGreen);
            int nc = rec.getNumberOfClusters();
            while (nc--) {
                //Int_t idx = rec.getClusterIndex(nc);
                Int_t idx = 0;
                const o2::itsmft::Cluster& c = mClusters[idx];
                const auto& gloC = c.getXYZGloRot(*gman);
                tpoints->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
            }
            track->AddElement(tpoints);
        }
        tracks->MakeTracks();

        return tracks;
    }


public:
    TEveElementList* displayData(int entry, int chip,
            std::vector<o2::itsmft::Digit>& mDigits,
            gsl::span<o2::its::TrackITS>& mTracks,
            gsl::span<o2::itsmft::Cluster>& mClusters)
    {
        //mEvent = new TEveElementList("balbnka");
        //return mEvent;


        std::string ename("Event #");
        ename += std::to_string(entry);

        // Chip display
        auto chipDigits = getEveChipDigits(chip, mDigits);
        auto chipClusters = getEveChipClusters(chip, mClusters);
        delete mChip;
        std::string cname(ename + "  ALPIDE chip #");
        cname += std::to_string(chip);
        mChip = new TEveElementList(cname.c_str());
        mChip->AddElement(chipDigits);
        mChip->AddElement(chipClusters);
        gEve->AddElement(mChip);

        // Event display
        auto clusters = getEveClusters(mClusters);
        auto tracks = getEveTracks(mTracks,mClusters);
        delete mEvent;
        mEvent = new TEveElementList(ename.c_str());
        mEvent->AddElement(clusters);
        mEvent->AddElement(tracks);
        return mEvent;
        //auto multi = o2::event_visualisation::MultiView::getInstance();
        //multi->registerEvent(mEvent);

        //gEve->Redraw3D(kFALSE);
    }
} its_eve;



o2::event_visualisation::DataSourceOfflineITS::DataSourceOfflineITS():DataSourceOffline(),
    digifile("itsdigits.root"),
    clusfile("o2clus_its.root"),
    tracfile("o2trac_its.root"){

    std::string inputGeom = "O2geometry.root";

    o2::base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
    auto gman = o2::its::GeometryTGeo::Instance();
    gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                              o2::TransformType::L2G));


}


bool o2::event_visualisation::DataSourceOfflineITS::open() {
    TFile* file;
    file = TFile::Open(digifile);
    if (file && gFile->IsOpen()) {
        file->Close();
        std::cout << "Running with MC digits...\n";
        its_data.setDigitPixelReader(digifile);
    }

    file = TFile::Open(clusfile);
    if (file && gFile->IsOpen())
        its_data.setClusTree((TTree*)gFile->Get("o2sim"));
    else
        std::cerr << "ERROR: Cannot open file: " << clusfile << "\n\n";

    file = TFile::Open(tracfile);
    if (file && gFile->IsOpen())
        its_data.setTracTree((TTree*)gFile->Get("o2sim"));
    else
        std::cerr << "\nERROR: Cannot open file: " << tracfile << "\n\n";

}

TEveElementList* o2::event_visualisation::DataSourceOfflineITS::gotoEvent(Int_t event) {
    std::cout << "o2::event_visualisation::DataSourceOfflineITS::gotoEvent "<< event <<std::endl;
    its_data.loadData(event);
    TEveElementList* mEvent = its_eve.displayData(event,0, its_data.mDigits, its_data.mTracks, its_data.mClusters);
    return mEvent;

}

