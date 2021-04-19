// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataReader.h
/// \brief Abstract base class for Detector-specific reading from file(s)
/// \author julian.myrcha@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATAREADER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATAREADER_H

#include "EventVisualisationDataConverter/VisualisationEvent.h"
#include "EventVisualisationBase/DataInterpreter.h"

class TObject;

namespace o2
{
namespace event_visualisation
{

class DataReader
{
  DataInterpreter* mInterpreter;
  virtual TObject* getEventData(int no){};

 public:
  DataReader(DataInterpreter* interpreter);
  virtual int GetEventCount() const = 0;
  virtual ~DataReader() = default;
  virtual void open() = 0;
  virtual VisualisationEvent getEvent(int no, EVisualisationDataType dataType);
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATAREADER_H
