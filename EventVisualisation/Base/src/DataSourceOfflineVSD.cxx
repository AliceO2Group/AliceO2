//
// Created by jmy on 26.02.19.
//



#include <EventVisualisationBase/DataSourceOfflineVSD.h>
#include <TSystem.h>
#include <TEveManager.h>
#include <TFile.h>
#include <TPRegexp.h>
#include <TEveTrackPropagator.h>
#include <TEveEventManager.h>



namespace o2  {
namespace event_visualisation {




    DataSourceOfflineVSD::DataSourceOfflineVSD()
            : DataSourceOffline(), fTrackList(0), fITSClusters(0), fTPCClusters(0), fTRDClusters(0), fTOFClusters(0),
              fVSD(0), fDirectory(0),
              fFile(0), fEvDirKeys(0),
              fMaxEv(-1), fCurEv(-1) {}
    //---------------------------------------------------------------------------
    // Cluster loading
    //---------------------------------------------------------------------------

    void DataSourceOfflineVSD::LoadClusters(TEvePointSet *&ps, const TString &det_name, Int_t det_id) {
        if (ps == 0) {
            ps = new TEvePointSet(det_name);
            ps->SetMainColor((Color_t) (det_id + 2));
            ps->SetMarkerSize(0.5);
            ps->SetMarkerStyle(2);
            ps->IncDenyDestroy();
        } else {
            ps->Reset();
        }

        //TEvePointSelector ss(fVSD->fTreeC, ps, "fV.fX:fV.fY:fV.fZ", TString::Format("fDetId==%d", det_id));
        //ss.Select();
        ps->SetTitle(TString::Format("N=%d", ps->Size()));

        gEve->AddElement(ps);
    }

DataSourceOfflineVSD::~DataSourceOfflineVSD()  {
        if (fVSD) {
            delete fVSD;
            fVSD = nullptr;
        }
        if (fEvDirKeys) {
            DropEvent();
            delete fEvDirKeys;
            fEvDirKeys = nullptr;

            fFile->Close();
            delete fFile;
            fFile == nullptr;
        }
    }

    void DataSourceOfflineVSD::AttachEvent() {
        // Attach event data from current directory.

        fVSD->LoadTrees();
        fVSD->SetBranchAddresses();
    }


    void DataSourceOfflineVSD::DropEvent() {
        // Drup currently held event data, release current directory.

        // Drop old visualization structures.

        this->viewers = gEve->GetViewers();
        this->viewers->DeleteAnnotations();
        TEveEventManager *manager = gEve->GetCurrentEvent();
        assert(manager != nullptr);
        manager->DestroyElements();

        // Drop old event-data.

        fVSD->DeleteTrees();
        delete fDirectory;
        fDirectory = 0;
    }

    void DataSourceOfflineVSD::LoadEsdTracks() {
        // Read reconstructed tracks from current event.

        if (fTrackList == 0) {
            fTrackList = new TEveTrackList("ESD Tracks");
            fTrackList->SetMainColor(6);
            fTrackList->SetMarkerColor(kYellow);
            fTrackList->SetMarkerStyle(4);
            fTrackList->SetMarkerSize(0.5);

            fTrackList->IncDenyDestroy();
        } else {
            fTrackList->DestroyElements();
        }

        TEveTrackPropagator *trkProp = fTrackList->GetPropagator();
        // !!!! Need to store field on file !!!!
        // Can store TEveMagField ?
        trkProp->SetMagField(0.5);
        trkProp->SetStepper(TEveTrackPropagator::kRungeKutta);

        Int_t nTracks = fVSD->fTreeR->GetEntries();
        for (Int_t n = 0; n < nTracks; ++n) {
            fVSD->fTreeR->GetEntry(n);

            TEveTrack *track = new TEveTrack(&fVSD->fR, trkProp);
            track->SetName(Form("ESD Track %d", fVSD->fR.fIndex));
            track->SetStdTitle();
            track->SetAttLineAttMarker(fTrackList);
            fTrackList->AddElement(track);
        }

        fTrackList->MakeTracks();

        gEve->AddElement(fTrackList);
    }

    bool DataSourceOfflineVSD::open()  {
        TString ESDFileName = "balbinka.root";
        fMaxEv = -1;
        fCurEv = -1;
        fFile = TFile::Open(ESDFileName);
        if (!fFile) {
            Error("VSD_Reader", "Can not open file '%s' ... terminating.",
                  ESDFileName.Data());
            gSystem->Exit(1);
        }

        fEvDirKeys = new TObjArray;
        TPMERegexp name_re("Event\\d+");
        TObjLink *lnk = fFile->GetListOfKeys()->FirstLink();
        while (lnk) {
            if (name_re.Match(lnk->GetObject()->GetName())) {
                fEvDirKeys->Add(lnk->GetObject());
            }
            lnk = lnk->Next();
        }

        fMaxEv = fEvDirKeys->GetEntriesFast();
        if (fMaxEv == 0) {
            Error("VSD_Reader", "No events to show ... terminating.");
            gSystem->Exit(1);
        }

        fVSD = new TEveVSD;
    }

    Bool_t DataSourceOfflineVSD::GotoEvent(Int_t ev) {
        if (ev < 0 || ev >= this->fMaxEv) {
            Warning("GotoEvent", "Invalid event id %d.", ev);
            return kFALSE;
        }

        this->DropEvent();

        // Connect to new event-data.

        this->fCurEv = ev;
        this->fDirectory = (TDirectory *) ((TKey *) this->fEvDirKeys->At(this->fCurEv))->ReadObj();
        this->fVSD->SetDirectory(this->fDirectory);

        this->AttachEvent();

        // Load event data into visualization structures.

        this->LoadClusters(this->fITSClusters, "ITS", 0);
        this->LoadClusters(this->fTPCClusters, "TPC", 1);
        this->LoadClusters(this->fTRDClusters, "TRD", 2);
        this->LoadClusters(this->fTOFClusters, "TOF", 3);

        this->LoadEsdTracks();
        return kTRUE;
    }

}


}

