// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   DataReaderITS.cxx
/// \brief  ITS Detector-specific reading from file(s)
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch
/*
#include "EventVisualisationBase/DataReaderITS.h"
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


using namespace o2::itsmft;
extern TEveManager* gEve;

static TGNumberEntry* gEntry;
static TGNumberEntry* gChipID;

class ITSData
{
public:
    void loadData(int entry);
    TEveElementList *displayData(int entry, int chip);
    int getLastEvent() const { return mLastEvent; }
    void setRawPixelReader(std::string input)
    {
        auto reader = new RawPixelReader<ChipMappingITS>();
        reader->openInput(input);
        mPixelReader = reader;
        mPixelReader->getNextChipData(mChipData);
        mIR = mChipData.getInteractionRecord();
    }
    void setDigitPixelReader(std::string input)
    {
        auto reader = new DigitPixelReader();
        reader->openInput(input, o2::detectors::DetID("ITS"));
        reader->init();
        reader->readNextEntry();
        mPixelReader = dynamic_cast<PixelReader*>(reader);                    // incompatible class DigitPixelReader : public PixelReader
        assert(mPixelReader != nullptr);
        mPixelReader->getNextChipData(mChipData);
        mIR = mChipData.getInteractionRecord();
    }
    void setDigiTree(TTree* t) { mDigiTree = t; }
    void setClusTree(TTree* t);
    void setTracTree(TTree* t);

private:
    // Data loading members
    int mLastEvent = 0;
    PixelReader* mPixelReader = nullptr;
    ChipPixelData mChipData;
    o2::InteractionRecord mIR;
    std::vector<Digit> mDigits;
    std::vector<Cluster>* mClusterBuffer = nullptr;
    gsl::span<Cluster> mClusters;
    std::vector<o2::itsmft::ROFRecord> mClustersROF;
    std::vector<o2::its::TrackITS>* mTrackBuffer = nullptr;
    gsl::span<o2::its::TrackITS> mTracks;
    std::vector<o2::itsmft::ROFRecord> mTracksROF;
    void loadDigits();
    void loadDigits(int entry);
    void loadClusters(int entry);
    void loadTracks(int entry);

    TTree* mDigiTree = nullptr;
    TTree* mClusTree = nullptr;
    TTree* mTracTree = nullptr;

    // TEve-related members
    TEveElementList* mEvent = nullptr;
    TEveElementList* mChip = nullptr;
    TEveElement* getEveChipDigits(int chip);
    TEveElement* getEveChipClusters(int chip);
    TEveElement* getEveClusters();
    TEveElement* getEveTracks();
} its_data;

void ITSData::loadDigits()
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

void ITSData::loadDigits(int entry)
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

void ITSData::setClusTree(TTree* tree)
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

void ITSData::loadClusters(int entry)
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

void ITSData::setTracTree(TTree* tree)
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

void ITSData::loadTracks(int entry)
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

void ITSData::loadData(int entry)
{
    loadDigits(entry);
    loadClusters(entry);
    loadTracks(entry);
}



constexpr float sizey = SegmentationAlpide::ActiveMatrixSizeRows;
constexpr float sizex = SegmentationAlpide::ActiveMatrixSizeCols;
constexpr float dy = SegmentationAlpide::PitchRow;
constexpr float dx = SegmentationAlpide::PitchCol;
constexpr float gap = 1e-4; // For a better visualization of pixels

TEveElement* ITSData::getEveChipDigits(int chip)
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

TEveElement* ITSData::getEveChipClusters(int chip)
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

TEveElement* ITSData::getEveClusters()
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

TEveElement* ITSData::getEveTracks()
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
            const Cluster& c = mClusters[idx];
            const auto& gloC = c.getXYZGloRot(*gman);
            tpoints->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
        }
        track->AddElement(tpoints);
    }
    tracks->MakeTracks();

    return tracks;
}

TEveElementList* ITSData::displayData(int entry, int chip)
{
    std::string ename("Event #");
    ename += std::to_string(entry);

    // Chip display
    auto chipDigits = getEveChipDigits(chip);
    auto chipClusters = getEveChipClusters(chip);
    delete mChip;
    std::string cname(ename + "  ALPIDE chip #");
    cname += std::to_string(chip);
    mChip = new TEveElementList(cname.c_str());
    mChip->AddElement(chipDigits);
    mChip->AddElement(chipClusters);
    gEve->AddElement(mChip);

    // Event display
    auto clusters = getEveClusters();
    auto tracks = getEveTracks();
    delete mEvent;
    mEvent = new TEveElementList(ename.c_str());
    mEvent->AddElement(clusters);
    mEvent->AddElement(tracks);
    return mEvent;
}



TEveElement* ITSload(int entry, int chip)
{
    int lastEvent = its_data.getLastEvent();
    if (lastEvent > entry) {
        std::cerr << "\nERROR: Cannot stay or go back over events. Please increase the event number !\n\n";
        gEntry->SetIntNumber(lastEvent - 1);
        return nullptr;
    }

    gEntry->SetIntNumber(entry);
    gChipID->SetIntNumber(chip);

    std::cout << "\n*** Event #" << entry << " ***\n";
    its_data.loadData(entry);
    return its_data.displayData(entry, chip);
}


void ITSload()
{
    auto event = gEntry->GetNumberEntry()->GetIntNumber();
    auto chip = gChipID->GetNumberEntry()->GetIntNumber();

    ITSload(event, chip);
}

void ITSnext()
{
    auto event = gEntry->GetNumberEntry()->GetIntNumber();
    event++;
    auto chip = gChipID->GetNumberEntry()->GetIntNumber();
    ITSload(event, chip);
}

void ITSprev()
{
    auto event = gEntry->GetNumberEntry()->GetIntNumber();
    event--;
    auto chip = gChipID->GetNumberEntry()->GetIntNumber();
    ITSload(event, chip);
}

void ITSloadChip()
{
    auto event = gEntry->GetNumberEntry()->GetIntNumber();
    auto chip = gChipID->GetNumberEntry()->GetIntNumber();
    its_data.displayData(event, chip);
}

void ITSloadChip(int chip)
{
    gChipID->SetIntNumber(chip);
    ITSloadChip();
}

void ITSnextChip()
{
    auto chip = gChipID->GetNumberEntry()->GetIntNumber();
    chip++;
    ITSloadChip(chip);
}

void ITSprevChip()
{
    auto chip = gChipID->GetNumberEntry()->GetIntNumber();
    chip--;
    ITSloadChip(chip);
}

void ITSDisplayEvents()
{
    // A dummy function with the same name as this macro
}



o2::event_visualisation::DataReaderITS::DataReaderITS() {

}

void o2::event_visualisation::DataReaderITS::open() {

}

Bool_t o2::event_visualisation::DataReaderITS::GotoEvent(Int_t ev) {
    return 0;
}
*/
