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
    std::vector<o2::itsmft::Digit> mDigits;
    gsl::span<o2::itsmft::Cluster> mClusters;
    gsl::span<o2::its::TrackITS> mTracks;
    int mLastEvent = 0;
public:
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




o2::event_visualisation::DataSourceOfflineITS::DataSourceOfflineITS():DataSourceOffline(),
    digifile("itsdigits.root"),
    clusfile("o2clus_its.root"),
    tracfile("o2trac_its.root"){
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

int o2::event_visualisation::DataSourceOfflineITS::gotoEvent(Int_t event) {
    its_data.loadData(0);
}

