// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataReaderTPC.h
/// \brief TPC Detector-specific reading from file(s)
/// \author julian.myrcha@cern.ch

#ifndef O2EVE_EVENTVISUALISATION_DETECTORS_DATAREADERTPC_H
#define O2EVE_EVENTVISUALISATION_DETECTORS_DATAREADERTPC_H

#include <TFile.h>
#include "EventVisualisationBase/DataReader.h"

namespace o2
{
namespace event_visualisation
{

class DataReaderTPC : public DataReader
{
 private:
  Int_t mMaxEv;
  TFile* mClusFile;
  TFile* mTracFile;

 public:
  DataReaderTPC();
  void open() override;
  Int_t GetEventCount() const override { return mMaxEv; };
  TObject* getEventData(int no) override;
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_EVENTVISUALISATION_DETECTORS_DATAREADERTPC_H
