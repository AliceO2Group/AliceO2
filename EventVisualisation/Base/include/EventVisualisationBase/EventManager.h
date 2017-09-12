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
/// \file    EventManager.h
/// \author  Jeremi Niedziela

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_EVENTMANAGER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_EVENTMANAGER_H

#include "CCDB/Manager.h"

#include <TEveElement.h>
#include <TEveEventManager.h>
#include <TQObject.h>

#include <string>

namespace o2  {
namespace EventVisualisation {

/// EventManager is a singleton class managing event loading.
///
/// This class is a hub for data visualisation classes, providing them with objects of requested type
/// (Raw data, hits, digits, clusters, ESDs, AODs...). It is a role of detector-specific data macros to
/// interpret data from different formats as visualisation objects (points, lines...) and register them
/// for drawing in the MultiView.
  
class EventManager : public TEveEventManager, public TQObject
{
  public:
    enum EDataSource{
      SourceOnline,   ///< Online reconstruction is a source of events
      SourceOffline,  ///< Local files are the source of events
      SourceHLT       ///< HLT reconstruction is a source of events
    };

    /// Returns an instance of EventManager
    static EventManager& getInstance();
  
    /// Setter of the current data source
    inline void setDataSourceType(EDataSource source){mCurrentDataSourceType = source;}
    /// Sets the CDB path in CCDB Manager
    inline void setCdbPath(std::string path){ o2::CDB::Manager::Instance()->setDefaultStorage(path.c_str()); }
  
  private:
    EDataSource mCurrentDataSourceType; ///< enum type of the current data source
  
    /// Default constructor
    EventManager();
    /// Default destructor
    ~EventManager() final;
    /// Deleted copy constructor
    EventManager(EventManager const&) = delete;
    /// Deleted assignemt operator
    void operator=(EventManager const&) = delete;
};

}
}

#endif


