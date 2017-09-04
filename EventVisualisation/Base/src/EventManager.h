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
/// \file    EveInitializer.h
/// \author  Jeremi Niedziela

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_EVENTMANAGER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_EVENTMANAGER_H

#include "CCDB/Manager.h"

#include <TEveElement.h>
#include <TEveEventManager.h>
#include <TQObject.h>

namespace o2  {
namespace EventVisualisation {

/// EventManager is a singleton class managing event loading and drawing.
///
/// This class is a hub for data macros, providing them with objects of requested type
/// (Raw data, hits, digits, clusters, ESDs, AODs...) and drawing registered shapes representing
/// the data. It is a role of detector-specific data macros to interpret data from different formats
/// as visualisation objects (points, lines...) and register them for drawing in the EventManager.
  
class EventManager : public TEveEventManager, public TQObject
{
  public:
    enum EDataSource{
      SourceOnline,   ///< Online reconstruction is a source of events
      SourceOffline,  ///< Local files are the source of events
      SourceHLT       ///< HLT reconstruction is a source of events
    };
    enum EDataType{
      Raw,      ///< Raw data
      Hits,     ///< Hits
      Digits,   ///< Digits
      Clusters, ///< Reconstructed clusters (RecPoints)
      ESD,      ///< Event Summary Data
      AOD       ///< Analysis Object Data
    };
    
    /// Returns an instance of EventManager
    static EventManager* getInstance();
    
    /// Registers an event to be drawn
    void registerEvent(TEveElement* event);
    /// Removes all shapes representing current event
    void restroyAllEvents();
    
    /// Setter of the current data source
    inline void setDataSourceType(EDataSource source){mCurrentDataSourceType = source;}
    /// Sets the CDB path in CCDB Manager
    inline void setCdbPath(std::string path){ o2::CDB::Manager::Instance()->setDefaultStorage(path.c_str()); }
    
  private:
    /// Default constructor
    EventManager();
    /// Default destructor
    ~EventManager() final;
    
    static EventManager* sMaster;       ///< Singleton instance of EventManager
    EDataSource mCurrentDataSourceType; ///< enum type of the current data source
};

}
}

#endif


