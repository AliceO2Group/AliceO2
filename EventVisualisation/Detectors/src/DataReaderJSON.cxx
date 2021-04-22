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
/// \file   DataReaderJSON.cxx
/// \brief  JSON specific reading from file(s)
/// \author julian.myrcha@cern.ch

#include "EventVisualisationDetectors/DataReaderJSON.h"

#include <TTree.h>

namespace o2
{
namespace event_visualisation
{

void DataReaderJSON::open()
{
  this->mFileName = "/home/jmy/CERN/event";
  this->mMaxEv = 0;
  while (true) {
    FILE* file = fopen(VisualisationEvent::fileNameIndexed(this->mFileName, this->mMaxEv).c_str(), "r");
    if (file == nullptr) {
      break;
    }
    fclose(file);
    this->mMaxEv++;
  }
}

VisualisationEvent DataReaderJSON::getEvent(int no, EVisualisationDataType /*dataType*/)
{
  VisualisationEvent vEvent;
  vEvent.fromFile(VisualisationEvent::fileNameIndexed(this->mFileName, no));
  return vEvent;
}

} // namespace event_visualisation
} // namespace o2
