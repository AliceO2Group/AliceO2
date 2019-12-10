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
/// \file   DataReaderTPC.cxx
/// \brief  TPC Detector-specific reading from file(s)
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#include "EventVisualisationDetectors/DataReaderTPC.h"
#include <TTree.h>
#include <TVector2.h>
#include <TError.h>
#include "DataFormatsTPC/TrackTPC.h"

namespace o2
{
namespace event_visualisation
{

DataReaderTPC::DataReaderTPC() = default;

void DataReaderTPC::open()
{
  TString clusterFile = "tpc-native-clusters.root";
  TString trackFile = "tpctracks.root";

  this->mTracFile = TFile::Open(trackFile);
  this->mClusFile = TFile::Open(clusterFile);

  TTree* trec = static_cast<TTree*>(this->mTracFile->Get("tpcrec"));
  std::vector<tpc::TrackTPC>* trackBuffer = nullptr;

  trec->SetBranchAddress("TPCTracks", &trackBuffer);
  trec->GetEntry(0);

  mMaxEv = trackBuffer->size();
}

TObject* DataReaderTPC::getEventData(int no)
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
