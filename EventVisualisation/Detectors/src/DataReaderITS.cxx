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

#include "EventVisualisationDetectors/DataReaderITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include <TTree.h>
#include <TVector2.h>
#include <TError.h>

namespace o2
{
namespace event_visualisation
{

void DataReaderITS::open()
{
  TString clusterFile = "o2clus_its.root";
  TString trackFile = "o2trac_its.root";

  this->mTracFile = TFile::Open(trackFile);
  this->mClusFile = TFile::Open(clusterFile);

  TTree* roft = (TTree*)this->mTracFile->Get("ITSTracksROF");
  TTree* rofc = (TTree*)this->mClusFile->Get("ITSClustersROF");

  if (roft != nullptr && rofc != nullptr) {
    // TTree* tracks = (TTree*)this->mTracFile->Get("o2sim");
    TTree* tracksRof = (TTree*)this->mTracFile->Get("ITSTracksROF");

    // TTree* clusters = (TTree*)this->mClusFile->Get("o2sim");
    TTree* clustersRof = (TTree*)this->mClusFile->Get("ITSClustersROF");

    //Read all track RO frames to a buffer
    std::vector<o2::itsmft::ROFRecord>* trackROFrames = nullptr;
    tracksRof->SetBranchAddress("ITSTracksROF", &trackROFrames);
    tracksRof->GetEntry(0);

    //Read all cluster RO frames to a buffer
    std::vector<o2::itsmft::ROFRecord>* clusterROFrames = nullptr;
    clustersRof->SetBranchAddress("ITSClustersROF", &clusterROFrames);
    clustersRof->GetEntry(0);

    if (trackROFrames->size() == clusterROFrames->size()) {
      mMaxEv = trackROFrames->size();
    } else {
      Error("DataReaderITS", "Inconsistent number of readout frames in files");
      exit(1);
    }
  }
}

TObject* DataReaderITS::getEventData(int no)
{
  /// FIXME: Redesign the data reader class
  TList* list = new TList();
  list->Add(this->mTracFile);
  list->Add(this->mClusFile);
  TVector2* v = new TVector2(no, 0);
  list->Add(v);
  return list;
}
} // namespace event_visualisation
} // namespace o2
