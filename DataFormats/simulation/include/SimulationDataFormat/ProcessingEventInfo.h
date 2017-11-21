// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ProcessingEventInfo.h
/// \brief Encapsulated meta information about current event being processed by FairRoot (analysis) tasks
/// \author Sandro Wenzel

#ifndef ALICEO2_DATA_EVENTINFO_H_
#define ALICEO2_DATA_EVENTINFO_H_

namespace o2 {

// A class encapsulating meta information about events being process 
// and the data being sent by run classes such as FairRunAna.
// Can be send to processing tasks for usage so that they do no longer
// need to access the FairRootManager directly.
struct ProcessingEventInfo {
  double eventTime; //! time of the current event
  int    eventNumber; //! the current entry
  int    sourceNumber; //! the current source number
  bool   numberSources; //! number of sources
  // can be extended further
};

}

#endif
