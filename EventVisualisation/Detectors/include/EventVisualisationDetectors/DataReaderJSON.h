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
/// \brief JSON specific reading from file(s)
/// \author julian.myrcha@cern.ch

#ifndef O2EVE_EVENTVISUALISATION_DETECTORS_DATAREADERJSON_H
#define O2EVE_EVENTVISUALISATION_DETECTORS_DATAREADERJSON_H

#include <TFile.h>
#include "EventVisualisationBase/DataReader.h"

namespace o2
{
namespace event_visualisation
{

class DataReaderJSON : public DataReader
{
 private:
  Int_t mMaxEv;
  std::string mFileName;

 public:
  DataReaderJSON(DataInterpreter* interpreter) : DataReader(interpreter) {}

  void open() override;
  int GetEventCount() const override { return mMaxEv; }
  VisualisationEvent getEvent(int no, EVisualisationDataType dataType) override;
};

} // namespace event_visualisation
} // namespace o2

#endif //O2EVE_DATAREADERJSON_H
