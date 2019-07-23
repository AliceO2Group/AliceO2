// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataReaderVSD.cxx
/// \brief VSD specific reading from file(s) (Visualisation Summary Data)
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch


#include <EventVisualisationBase/DataReaderVSD.h>
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

DataReaderVSD::DataReaderVSD()
        : DataReader(),
          fFile(0), fEvDirKeys(0),
          fMaxEv(-1), fCurEv(-1) {
}


DataReaderVSD::~DataReaderVSD()  {
  if (fEvDirKeys) {
    //dynamic_cast<DataInterpreterVSD*>(this->dataInterpreter)->DropEvent();
    delete fEvDirKeys;
    fEvDirKeys = nullptr;

    fFile->Close();
    delete fFile;
    fFile = nullptr;
  }
}


void DataReaderVSD::open()  {
    TString ESDFileName = "events_0.root";
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



TObject *DataReaderVSD::getEventData(int ev) {
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

