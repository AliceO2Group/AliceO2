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
/// \brief ITS Detector-specific reading from file(s)
/// \author julian.myrcha@cern.ch

#ifndef O2EVE_EVENTVISUALISATION_DETECTORS_DATAREADERITS_H
#define O2EVE_EVENTVISUALISATION_DETECTORS_DATAREADERITS_H

#include <TFile.h>
#include "EventVisualisationBase/DataReader.h"

namespace o2
{
namespace event_visualisation
{

class DataReaderITS : public DataReader
{
 private:
  Int_t mMaxEv;
  TFile* mClusFile;
  TFile* mTracFile;

 public:
  DataReaderITS(DataInterpreter* interpreter) : DataReader(interpreter) {}

  void open() override;
  int GetEventCount() const override { return mMaxEv; }

  TObject* getEventData(int no) override;
};

} // namespace event_visualisation
} // namespace o2

#endif //O2EVE_DATAREADERITS_H
