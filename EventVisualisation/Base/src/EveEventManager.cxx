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
/// @file    EveInitializer.cxx
/// @author  Jeremi Niedziela


ClassImp(EveEventManager)

EveEventManager* EveEventManager::mgMaster  = nullptr;

EveEventManager::EveEventManager()
//:
//TEveEventManager("Event", ""),
//fEventId(-1),fEventInfo(),fHasEvent(kFALSE),fCurrentRun(-1),fSelectedTrigger(""),
//fCurrentData(&fEmptyData),fCurrentDataSource(NULL),fDataSourceOnline(NULL),fDataSourceOffline(NULL),fDataSourceHLTZMQ(NULL), fCurrentDataSourceType(defaultDataSource),
//fAutoLoad(kFALSE), fAutoLoadTime(5),fAutoLoadTimer(0),fAutoLoadTimerRunning(kFALSE),
//fTransients(0),
//fExecutor(0),fViewsSaver(0),fESDTracksDrawer(0),fAODTracksDrawer(0),fPrimaryVertexDrawer(0),fKinksDrawer(0),fCascadesDrawer(0),fV0sDrawer(0),fMuonTracksDrawer(0),fSPDTracklersDrawer(0),fKineTracksDrawer(0),  fMomentumHistogramsDrawer(0),fPEventSelector(0),
//fgGRPLoaded(false),
//fgMagField(0)
{
  mgMaster = this;
  ChangeDataSource(kSourceOffline);
}

EveEventManager* EveEventManager::Instance()
{
  if(!mgMaster){
    new EveEventManager();
  }
  return mgMaster;
}
