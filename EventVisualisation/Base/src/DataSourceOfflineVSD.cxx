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
#include <EventVisualisationBase/EventManager.h>
#include <EventVisualisationBase/EventRegistration.h>


namespace o2  {
namespace event_visualisation {

DataSourceOfflineVSD::DataSourceOfflineVSD()
        : DataSourceOffline(),
          fFile(0), fEvDirKeys(0),
          fMaxEv(-1), fCurEv(-1) {
}


DataSourceOfflineVSD::~DataSourceOfflineVSD()  {
  if (fEvDirKeys) {
    //dynamic_cast<DataInterpreterVSD*>(this->dataInterpreter)->DropEvent();
    delete fEvDirKeys;
    fEvDirKeys = nullptr;

    fFile->Close();
    delete fFile;
    fFile = nullptr;
  }
}


void DataSourceOfflineVSD::open(TString ESDFileName)  {
    Warning("GotoEvent", "OPEN");
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
}



TObject *DataSourceOfflineVSD::getEventData(int ev) {
  Warning("GotoEvent", "GOTOEVENT");
  if (ev < 0 || ev >= this->fMaxEv) {
    Warning("GotoEvent", "Invalid event id %d.", ev);
    return nullptr;
  }
  this->fCurEv = ev;
  return ((TKey *) this->fEvDirKeys->At(this->fCurEv))->ReadObj();
}

}
}

