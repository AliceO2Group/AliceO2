// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file    EveInitializer.h
/// @author  Jeremi Niedziela

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_EVEEVENTMANAGER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_EVEEVENTMANAGER_H

//#include <AliEventInfo.h>
//#include <AliESDEvent.h>
//#include <AliEveSaveViews.h>
//#include <AliEveESDTracks.h>
//#include <AliEveESDKinks.h>
//#include <AliEveESDCascades.h>
//#include <AliEveESDV0s.h>
//#include <AliEveESDMuonTracks.h>
//#include <AliEveESDSPDTracklets.h>
//#include <AliEveAODTracks.h>
//#include <AliEveDataSource.h>
//#include <AliEveMomentumHistograms.h>
//#include <AliEvePrimaryVertex.h>
//#include <AliEveKineTracks.h>
//
//#include <TEveEventManager.h>
//#include <TQObject.h>

//class AliEveMacroExecutor;
//class AliEveEventSelector;
//class AliMagF;

//==============================================================================
//
// EveEventManager
//
// Interface to ALICE event-data (RunLoader, ESD), magnetic field and
// geometry.
//

class EveEventManager : public TEveEventManager, public TQObject
{
public:
  enum EDataSource { kSourceHLT, kSourceOnline, kSourceOffline };
  enum EDataType { kRaw,kHits,kDigits,kClusters,kESD,kAOD };
  
  static EveEventManager& Instance();
  
  // getters for data from current data source:
//  inline AliESDEvent* GetESD(){ return fCurrentData->fESD; }
//  inline AliAODEvent* GetAOD(){ return fCurrentData->fAOD; }
  
//  AliRunLoader*  GetRunLoader(){ return fCurrentData->fRunLoader;}
//  AliRawReader*  GetRawReader(){ return fCurrentData->fRawReader;}
//  TFile*         GetESDFile()  { return fCurrentData->fESDFile;  }
//  TFile*         GetAODFile()  { return fCurrentData->fAODFile;  }
//  TTree*         GetESDTree()  { return fCurrentData->fESDTree;  }
//  TTree*         GetAODTree()  { return fCurrentData->fAODTree;  }
//  AliESDfriend*  GetESDfriend(){ return fCurrentData->fESDfriend;}
  
  
//  void AddElement(TEveElement *element, TEveElement *parent=0);
//  
//  inline void Redraw3D(){gEve->Redraw3D();}
//  inline void EnableRedraw(){gEve->EnableRedraw();}
//  inline void DisableRedraw(){gEve->DisableRedraw();}
  
  // autoload timer getters and setters
//  Double_t      GetAutoLoadTime()        const { return fAutoLoadTime; }
//  Bool_t        GetAutoLoad()            const { return fAutoLoad;     }
//  bool          GetAutoLoadRunning()     const { return fAutoLoadTimerRunning;}
//  
//  void          SetAutoLoadTime(Float_t time){fAutoLoadTime = time;}
//  void          SetAutoLoad(Bool_t autoLoad);
//  void          StartAutoLoadTimer();
//  void          StopAutoLoadTimer();
//  
//  // global and transient elements:
//  void          RegisterTransient(TEveElement* element);
//  void          DestroyTransients();
//  
//  // data sources:
//  void ChangeDataSource(EDataSource newSource);
//  AliEveDataSource* GetCurrentDataSource(){return fCurrentDataSource;}
//  AliEveDataSource* GetDataSourceOnline(){return fDataSourceOnline;}
//  AliEveDataSource* GetDataSourceOffline(){return fDataSourceOffline;}
//  AliEveDataSource* GetDataSourceHLTZMQ(){return fDataSourceHLTZMQ;}
  
  
//  inline void SetCdbUri(TString path){ AliCDBManager::Instance()->SetDefaultStorage(path); }
  
  // getters and setters for info about events:
//  Int_t          GetEventId() const {return fEventId;}
//  Int_t          GetMaxEventId();
//  int            GetCurrentRun() {return fCurrentRun;}
//  std::string    GetSelectedTrigger() {return fSelectedTrigger;}
//  
//  void           SetEventId(int eventId)    { fEventId=eventId;}
//  void           SetCurrentRun(int run);
//  void           SetSelectedTrigger(std::string selectedTrigger) {fSelectedTrigger = selectedTrigger;}
//  void           SetHasEvent(bool hasEvent){fHasEvent=hasEvent;}
  
  // other public methods:
//  void                        ResetMagneticField(){fgMagField=0;}
//  virtual void                AfterNewEventLoaded();
//  AliEveMacroExecutor*        GetExecutor() const { return fExecutor; }
//  AliEveEventSelector*        GetEventSelector() const { return fPEventSelector; }
//  AliEveMomentumHistograms*   GetMomentumHistogramsDrawer(){return fMomentumHistogramsDrawer;}
//  
//  Bool_t InitOCDB(int runNo);
//  
//  // signals:
//  void Timeout();             // *SIGNAL*
//  void NewEventDataLoaded();  // *SIGNAL*
//  void NewEventLoaded();      // *SIGNAL*
//  
//  void AutoLoadNextEvent();
private:
  EveEventManager();
  ~EveEventManager();
  static EveEventManager *mgMaster; // singleton instance of EveEventManager
  
//  void   InitInternals();
//  Bool_t InitGRP();
//  
//  Int_t         fEventId;         // Id of current event.
//  AliEventInfo  fEventInfo;       // Current Event Info
//  Bool_t        fHasEvent;        // Is an event available.
//  int           fCurrentRun;      // Current run number
//  std::string   fSelectedTrigger; // Selected trigger class for events filtering
//  
//  AliEveData        fEmptyData;          //just a place holder in case we have no sources
//  const AliEveData* fCurrentData;        //current data struct from one of the data sources
//  AliEveDataSource* fCurrentDataSource;  // data source in use at the moment
//  AliEveDataSource* fDataSourceOnline;   // pointer to online data source
//  AliEveDataSource* fDataSourceOffline;  // pointer to offline data source
//  AliEveDataSource* fDataSourceHLTZMQ;   // pointer to HLT ZMQ data source
//  EDataSource fCurrentDataSourceType;    // enum type of the current data source
//  
//  
//  Bool_t   fAutoLoad;              // Automatic loading of events (online)
//  Float_t  fAutoLoadTime;          // Auto-load time in seconds
//  TTimer*  fAutoLoadTimer;         // Timer for automatic event loading
//  Bool_t   fAutoLoadTimerRunning;  // State of auto-load timer.
//  
//  TEveElementList*  fTransients;      // Container for additional transient (per event) elements.
//  
//  AliEveMacroExecutor*        fExecutor;                  // Executor for std macros
//  AliEveSaveViews*            fViewsSaver;                // views saver
//  AliEveESDTracks*            fESDTracksDrawer;           // drawer of ESD tracks
//  AliEveAODTracks*            fAODTracksDrawer;           // drawer of AOD tracks
//  AliEveMomentumHistograms*   fMomentumHistogramsDrawer;  // drawer of momentum histograms
//  AliEvePrimaryVertex*        fPrimaryVertexDrawer;       // drawer of primary vertex
//  AliEveESDKinks*             fKinksDrawer;               // drawer of ESD kinks
//  AliEveESDCascades*          fCascadesDrawer;            // drawer of ESD cascades
//  AliEveESDV0s*               fV0sDrawer;                 // drawer of ESD v0s
//  AliEveESDMuonTracks*        fMuonTracksDrawer;          // drawer of ESD muon tracks
//  AliEveESDSPDTracklets*      fSPDTracklersDrawer;        // drawer of ESD SPD tracklets
//  AliEveKineTracks*           fKineTracksDrawer;          // drawer of tracks from Kinematics.root
//  AliEveEventSelector*        fPEventSelector;            // Event filter
//  
//  Bool_t    fgGRPLoaded;     // Global run parameters loaded?
//  AliMagF*  fgMagField;      // Global pointer to magnetic field.
  
  ClassDef(EveEventManager, 0); // Interface for getting all event components in a uniform way.
};

#endif
