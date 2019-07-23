// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataReaderITS.h
/// \brief VSD specific reading from file(s) (Visualisation Summary Data)
/// \author julian.myrcha@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINEVSD_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINEVSD_H

#include <EventVisualisationBase/DataReader.h>
#include <EventVisualisationBase/VisualisationConstants.h>
#include <TString.h>
#include <TEveTrack.h>
#include <TEveViewer.h>
#include <TEveVSD.h>

namespace o2  {
namespace event_visualisation {

class DataReaderVSD : public DataReader {
    TFile *fFile;
    TObjArray *fEvDirKeys;
    Int_t fMaxEv, fCurEv;

public:
    Int_t GetEventCount() override { return fEvDirKeys->GetEntriesFast(); };
    DataReaderVSD();
    ~DataReaderVSD() override;
    void open() override;
    TObject* getEventData(int no) override;
};


}
}
#endif