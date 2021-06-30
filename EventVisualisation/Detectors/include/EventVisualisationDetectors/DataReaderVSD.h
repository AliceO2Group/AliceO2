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

/// \file DataReaderITS.h
/// \brief VSD specific reading from file(s) (Visualisation Summary Data)
/// \author julian.myrcha@cern.ch

#ifndef O2EVE_EVENTVISUALISATION_DETECTORS_DATAREADERVSD_H
#define O2EVE_EVENTVISUALISATION_DETECTORS_DATAREADERVSD_H

#include <EventVisualisationBase/DataReader.h>
#include <EventVisualisationBase/VisualisationConstants.h>
#include <TString.h>
#include <TEveTrack.h>
#include <TEveViewer.h>
#include <TEveVSD.h>

class TKey;

namespace o2
{
namespace event_visualisation
{

class DataReaderVSD : public DataReader
{
  TFile* mFile;
  std::vector<TKey*> mEvDirKeys;
  Int_t mMaxEv, mCurEv;

 public:
  //Int_t GetEventCount() override { return mEvDirKeys->GetEntriesFast(); };
  int GetEventCount() const override { return mEvDirKeys.size(); };
  DataReaderVSD(DataInterpreter* interpreter);
  ~DataReaderVSD() override;
  void open() override;
  TObject* getEventData(int no) override;
};

} // namespace event_visualisation
} // namespace o2
#endif // O2EVE_EVENTVISUALISATION_DETECTORS_DATAREADERVSD_H