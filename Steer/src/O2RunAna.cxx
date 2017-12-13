// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Steer/O2RunAna.h"

#include "FairBaseParSet.h"             // for FairBaseParSet
#include "FairGeoParSet.h"              // for FairGeoParSet
#include "FairEventHeader.h"            // for FairEventHeader
#include "FairField.h"                  // for FairField
#include "FairFieldFactory.h"           // for FairFieldFactory
#include "FairFileHeader.h"             // for FairFileHeader
#include "FairLogger.h"                 // for FairLogger, MESSAGE_ORIGIN
#include "FairParIo.h"                  // for FairParIo
#include "FairParSet.h"                 // for FairParSet
#include "FairRootManager.h"            // for FairRootManager
#include "FairRunIdGenerator.h"         // for FairRunIdGenerator
#include "FairRuntimeDb.h"              // for FairRuntimeDb
#include "FairTask.h"                   // for FairTask
#include "FairTrajFilter.h"             // for FairTrajFilter
#include "FairSystemInfo.h"

#include "SimulationDataFormat/ProcessingEventInfo.h"

#include "FairFileSource.h"             // ONLY TEMPORARILY, FOR COMPABILITY
#include "FairMixedSource.h"            // ONLY TEMPORARILY, FOR COMPABILITY

#include <iosfwd>                       // for ostream
#include "TChain.h"                     // for TChain
#include "TCollection.h"                // for TIter
#include "TDirectory.h"                 // for TDirectory, gDirectory
#include "TFile.h"                      // for TFile, gFile
#include "TGeoManager.h"                // for gGeoManager, TGeoManager
#include "TKey.h"                       // for TKey
#include "TList.h"                      // for TList
#include "TNamed.h"                     // for TNamed
#include "TObjArray.h"                  // for TObjArray
#include "TObject.h"                    // for TObject
#include "TROOT.h"                      // for TROOT, gROOT
#include "TSeqCollection.h"             // for TSeqCollection
#include "TSystem.h"                    // for TSystem, gSystem
#include "TTree.h"                      // for TTree

#include <cstdlib>                     // for NULL, exit
#include <csignal>
#include <cstring>                     // for strcmp
#include <iostream>                     // for operator<<, basic_ostream, etc
#include <list>                         // for list

using std::cout;
using std::endl;
using std::list;
using namespace o2::steer;

#define LAZY

Bool_t gFRAIsInterrupted;

//_____________________________________________________________________________
void FRA_handler_ctrlc(int)
{
  LOG(INFO) << "*********** CTRL C PRESSED *************" << FairLogger::endl;
  gFRAIsInterrupted = kTRUE;
}
//_____________________________________________________________________________

//_____________________________________________________________________________
O2RunAna* O2RunAna::Instance()
{
  static O2RunAna instance;
  return &instance;
}
//_____________________________________________________________________________
O2RunAna::O2RunAna()
  :FairRun(),
   mRunInfo(),
   mIsInitialized(kFALSE),
   mInputGeoFile(nullptr),
   mLoadGeo( kFALSE),
   mStatic(kFALSE),
   mField(nullptr),
   mInFileIsOpen(kFALSE),
   mEventTimeMin(0),
   mEventTimeMax(0),
   mEventTime(0),
   mEventMeanTime(0),
   mTimeProb(nullptr),
   mFileSource(nullptr),
   mMixedSource(nullptr),
   mStoreEventHeader(kTRUE),
   mEventInfo(new o2::ProcessingEventInfo)
{
  // defined in base class:
  fAna=kTRUE;
}
//_____________________________________________________________________________

//_____________________________________________________________________________
O2RunAna::~O2RunAna()
{
  // we are not owning the field!!
  // delete fField;
  if (gGeoManager) {
    if (gROOT->GetVersionInt() >= 60602) {
      gGeoManager->GetListOfVolumes()->Delete();
      gGeoManager->GetListOfShapes()->Delete();
    }
    delete gGeoManager;
  }
}

//_____________________________________________________________________________

void O2RunAna::SetGeomFile(const char* GeoFileName)
{
  if (mIsInitialized) {
    LOG(FATAL) << "Geometry file has to be set before Run::Init !"
               << FairLogger::endl;
    exit(-1);
  } else {

    TFile* CurrentFile=gFile;
    mInputGeoFile= TFile::Open(GeoFileName);
    if (mInputGeoFile->IsZombie()) {
      LOG(ERROR) << "Error opening Geometry Input file"
                 << FairLogger::endl;
      mInputGeoFile=nullptr;
    }
    LOG(INFO) << "Opening Geometry input file: " << GeoFileName
              << FairLogger::endl;
    mLoadGeo=kTRUE;
    gFile=CurrentFile;
  }
}

//_____________________________________________________________________________

void O2RunAna::Init()
{
  if (mIsInitialized) {
    LOG(FATAL) << "Error Init is already called before!" << FairLogger::endl;
    exit(-1);
  } else {
    mIsInitialized=kTRUE;
  }

  // register important data output (as memory branch)
  auto iomgr = FairRootManager::Instance();
  iomgr->RegisterAny("EventInfo", mEventInfo, false);

  // Check if we have an input file to be used
  mInFileIsOpen = fRootManager->InitSource();

 //Load Geometry from user file
  if (mLoadGeo) {
    if (mInputGeoFile!=nullptr) { //First check if the user has a separate Geo file!
      TIter next(mInputGeoFile->GetListOfKeys());
      TKey* key;
      while ((key =dynamic_cast< TKey*>(next()))) {
        if (strcmp(key->GetClassName(),"TGeoManager") != 0) {
          continue;
        }
        gGeoManager = dynamic_cast<TGeoManager*>(key->ReadObj());
        break;
      }
    }
  } else {
    /*** Get the container that normly has the geometry and all the basic stuff from simulation*/
    fRtdb->getContainer("FairGeoParSet");
  }

  if (mInFileIsOpen) {
    //check that the geometry was loaded if not try all connected files!
    if (mLoadGeo && gGeoManager==nullptr) {
      LOG(INFO) << "Geometry was not found in the input file we will look in the friends if any!" << FairLogger::endl;
      TFile* currentfile= gFile;
      TFile* nextfile=nullptr;
      TSeqCollection* fileList=gROOT->GetListOfFiles();
      for (Int_t k=0; k<fileList->GetEntries(); k++) {
        nextfile=dynamic_cast<TFile*>(fileList->At(k));
        if (nextfile) {
          nextfile->Get("FAIRGeom");
        }
        if (gGeoManager) {
          break;
        }
      }
      gFile=currentfile;
    }
  } else { //  if(fInputFile )
    // NO input file but there is a geometry file
    if (mLoadGeo) {
      if (mInputGeoFile!=nullptr) { //First check if the user has a separate Geo file!
        TIter next(mInputGeoFile->GetListOfKeys());
        TKey* key;
        while ((key = dynamic_cast<TKey*>(next()))) {
          if (strcmp(key->GetClassName(),"TGeoManager") != 0) {
            continue;
          }
          gGeoManager = dynamic_cast<TGeoManager*>(key->ReadObj());
          break;
        }
      }
    }
  }
  gROOT->GetListOfBrowsables()->Add(fTask);

  // Init the RTDB containers
  //  auto par=dynamic_cast<FairBaseParSet*>(fRtdb->getContainer("FairBaseParSet"));
  if (/*par &&*/ mInFileIsOpen) {

    LOG(INFO) << "Parameter and input file are available, Assure that basic info is there for the run!" << FairLogger::endl;
    FairSystemInfo sysInfo;
    LOG(DEBUG) << "MEMORY BEFORE READEVENT(0) " << sysInfo.GetMaxMemory() << FairLogger::endl;      
    //    fRootManager->ReadEvent(0);
    LOG(DEBUG) << "MEMORY AFTER READEVENT(0) " << sysInfo.GetMaxMemory() << FairLogger::endl;
    
    // This is *CREATING* an instance of an event header (pointed to by fEvtHeader)
    // (or returning an existing instance)
    // happening in the base class
    GetEventHeader();

    // we fill the event header
    fRootManager->FillEventHeader(fEvtHeader);

    fRunId = fEvtHeader->GetRunId();

    // Copy the Event Header Info to Output
    fEvtHeader->Register(mStoreEventHeader);

    // Init the containers in Tasks
    fRtdb->initContainers(fRunId);

    // not sure what this is doing
    fTask->SetParTask();

  }
  else
  { 
    LOG(INFO) << "Initializing without input file or Mixed input"
              << FairLogger::endl;
    FairEventHeader* evt = GetEventHeader();
    evt->Register(mStoreEventHeader);
    FairRunIdGenerator genid;
    fRunId = genid.generateId();
    fRtdb->addRun(fRunId);
    evt->SetRunId( fRunId);
    fTask->SetParTask();
    fRtdb->initContainers( fRunId );
  }
  FairFieldFactory* fieldfact= FairFieldFactory::Instance();
  if (fieldfact) {
    fieldfact->SetParm();
  }

  fRtdb->initContainers(fRunId);
  fFileHeader->SetRunId(fRunId);

  // create a field if the user did not give one
  // <DB>
  // Add test for external FairField settings
  if (fieldfact && !mField) {
    mField = fieldfact->createFairField();
  }

  // Now call the init function on attach tasks
  auto beforeinit = fRootManager->GetReqBranchNames();
  auto oldsize = beforeinit.size();
  fTask->InitTask();
  auto afterinit = fRootManager->GetReqBranchNames();
  mRequestedBranches = afterinit;
  auto newsize = afterinit.size();
  //  LOG(INFO) << "The following branches were requestes as input by task\n";
  //  for(auto& s : afterinit) {
  //    LOG(INFO) << s << "\n";
  //}
  
  // if the vis manager is available then initialize it!
  FairTrajFilter* fTrajFilter = FairTrajFilter::Instance();
  if (fTrajFilter) {
    fTrajFilter->Init();
  }

  // create the output tree after tasks initialisation
  fOutFile->cd();
  TTree* outTree =new TTree(FairRootManager::GetTreeName(), "/cbmout", 99);
  fRootManager->TruncateBranchNames(outTree, "cbmout");
  // FIXME: Do not forget to create external branches
  fRootManager->SetOutTree(outTree);
  fRootManager->WriteFolder();
  fRootManager->WriteFileHeader(fFileHeader);
}
//_____________________________________________________________________________

//_____________________________________________________________________________
void O2RunAna::Run(Int_t Ev_start, Int_t Ev_end)
{
  gFRAIsInterrupted = kFALSE;

  if (false /*fTimeStamps*/) {
    RunTSBuffers();
  }
  else
  {
    UInt_t tmpId =0;
    //  if (fInputFile==0) {
    if (!mInFileIsOpen) {
      DummyRun(Ev_start,Ev_end);
      return;
    }

   Int_t MaxAllowed=fRootManager->CheckMaxEventNo(Ev_end);
    if ( MaxAllowed != -1 ) {
      if (Ev_end==0) {
        if (Ev_start==0) {
          Ev_end=MaxAllowed;
        } else {
          Ev_end =  Ev_start;
          if ( Ev_end > MaxAllowed ) {
            Ev_end = MaxAllowed;
          }
          Ev_start=0;
        }
      } else {
        if (Ev_end > MaxAllowed) {
          cout << "-------------------Warning---------------------------" << endl;
          cout << " -W O2RunAna : File has less events than requested!!" << endl;
          cout << " File contains : " << MaxAllowed  << " Events" << endl;
          cout << " Requested number of events = " <<  Ev_end <<  " Events"<< endl;
          cout << " The number of events is set to " << MaxAllowed << " Events"<< endl;
          cout << "-----------------------------------------------------" << endl;
          Ev_end = MaxAllowed;
        }
      }
      LOG(INFO) << "O2RunAna::Run() After checking, the run will run from event " << Ev_start << " to " << Ev_end << "." << FairLogger::endl;
    }
    else {
      LOG(INFO) << "O2RunAna::Run() continue running without stop" << FairLogger::endl;
    }

    if (fGenerateRunInfo) {
      mRunInfo.Reset();
    }

    Int_t readEventReturn = 0;

    for (int i=Ev_start; i< Ev_end || MaxAllowed==-1 ; i++) {

      gSystem->IgnoreInterrupt();
      //  gFRAIsInterrupted = kFALSE;
      signal(SIGINT, FRA_handler_ctrlc);

      if ( gFRAIsInterrupted ) {
        LOG(WARNING) << "O2RunAna::Run() Event loop was interrupted by the user!" << FairLogger::endl;
        break;
      }
 
#ifdef LAZY
      // FIXME: fix event timings (should be the same as in nonlazy case)
      for(auto& name : mRequestedBranches) {
        std::cerr << "READING " << name << "\n";
        fRootManager->ReadBranchEvent(name.c_str(), i);
      }
#else
      readEventReturn = fRootManager->ReadEvent(i);
      if ( readEventReturn != 0 ) {
        LOG(WARNING) << "O2RunAna::Run() fRootManager->ReadEvent(" << i << ") returned " << readEventReturn << ". Breaking the event loop" << FairLogger::endl;
        break;
      }
#endif
      fRootManager->FillEventHeader(fEvtHeader);

      // fill the event info structure
      mEventInfo->eventNumber = i;
      mEventInfo->eventTime = fRootManager->GetEventTime();


      tmpId = fEvtHeader->GetRunId();
      if ( tmpId != fRunId ) {
        fRunId = tmpId;
        if ( !mStatic ) {
          Reinit( fRunId );
          fTask->ReInitTask();
        }
      }
      //std::cout << "WriteoutBufferData with time: " << fRootManager->GetEventTime();
      fRootManager->StoreWriteoutBufferData(fRootManager->GetEventTime());
      fTask->ExecuteTask("");
      Fill();
      fRootManager->DeleteOldWriteoutBufferData();
      fTask->FinishEvent();

      if (fGenerateRunInfo) {
        mRunInfo.StoreInfo();
      }
      if (nullptr !=  FairTrajFilter::Instance()) {
        FairTrajFilter::Instance()->Reset();
      }

    }

    fRootManager->StoreAllWriteoutBufferData();
    fTask->FinishTask();
    if (fGenerateRunInfo) {
      mRunInfo.WriteInfo();
    }
    fRootManager->LastFill();
    fRootManager->Write();
  }
}
//_____________________________________________________________________________

//_____________________________________________________________________________
void O2RunAna::RunEventReco(Int_t Ev_start, Int_t Ev_end)
{
  UInt_t tmpId =0;

  Int_t MaxAllowed=fRootManager->CheckMaxEventNo(Ev_end);
  if ( MaxAllowed != -1 ) {
    if (Ev_end==0) {
      if (Ev_start==0) {
        Ev_end=MaxAllowed;
      } else {
        Ev_end =  Ev_start;
        if ( Ev_end > MaxAllowed ) {
          Ev_end = MaxAllowed;
        }
        Ev_start=0;
      }
    } else {
      if (Ev_end > MaxAllowed) {
        cout << "-------------------Warning---------------------------" << endl;
        cout << " -W O2RunAna : File has less events than requested!!" << endl;
        cout << " File contains : " << MaxAllowed  << " Events" << endl;
        cout << " Requested number of events = " <<  Ev_end <<  " Events"<< endl;
        cout << " The number of events is set to " << MaxAllowed << " Events"<< endl;
        cout << "-----------------------------------------------------" << endl;
        Ev_end = MaxAllowed;
      }
    }
    LOG(INFO) << "O2RunAna::Run() After checking, the run will run from event " << Ev_start << " to " << Ev_end << "." << FairLogger::endl;
  }
  else {
    LOG(INFO) << "O2RunAna::Run() continue running without stop" << FairLogger::endl;
  }

  if (fGenerateRunInfo) {
    mRunInfo.Reset();
  }

  for (int i=Ev_start; i< Ev_end; i++) {

    // fRootManager->ReadEvent(i);
    for(auto& name : mRequestedBranches) {
      std::cerr << "READING " << name << "\n";
      fRootManager->ReadBranchEvent(name.c_str(), i);
    }

    /**
     * if we have simulation files then they have MC Event Header and the Run Id is in it, any way it
     * would be better to make FairMCEventHeader a subclass of FairEvtHeader.
     */
    if ( tmpId != fRunId ) {
      fRunId = tmpId;
      if ( !mStatic ) {
        Reinit( fRunId );
        fTask->ReInitTask();
      }
    }
    //FairMCEventHeader* header = dynamic_cast<FairMCEventHeader*>(fRootManager->GetObject("MCEventHeader.");
    //    std::cout << "WriteoutBufferData with time: " << fRootManager->GetEventTime();
    fRootManager->StoreWriteoutBufferData(fRootManager->GetEventTime());
    fTask->ExecuteTask("");

    fRootManager->FillEventHeader(fEvtHeader);
    // Fill();
    fTask->FinishEvent();

    if (fGenerateRunInfo) {
      mRunInfo.StoreInfo();
    }
    if (nullptr !=  FairTrajFilter::Instance()) {
      FairTrajFilter::Instance()->Reset();
    }

  }

  fTask->FinishTask();
  if (fGenerateRunInfo) {
    mRunInfo.WriteInfo();
  }
  fRootManager->LastFill();
  fRootManager->Write();
}
//_____________________________________________________________________________

//_____________________________________________________________________________
void O2RunAna::Run(Double_t delta_t)
{
  while (fRootManager->ReadNextEvent(delta_t)==kTRUE) {
    fTask->ExecuteTask("");
    fRootManager->FillEventHeader(fEvtHeader);
    Fill();
    fRootManager->DeleteOldWriteoutBufferData();
    fTask->FinishEvent();
    if (nullptr !=  FairTrajFilter::Instance()) {
      FairTrajFilter::Instance()->Reset();
    }
  }

  fRootManager->StoreAllWriteoutBufferData();
  fTask->FinishTask();
  fRootManager->LastFill();
  fRootManager->Write();

}
//_____________________________________________________________________________


//_____________________________________________________________________________
void O2RunAna::RunMQ(Long64_t entry)
{
  /**
   This methode is only needed and used with ZeroMQ
   it read a certain event and call the task exec, but no output is written
   */
  UInt_t tmpId =0;
  fRootManager->ReadEvent(entry);
  tmpId = fEvtHeader->GetRunId();
  if ( tmpId != fRunId ) {
    fRunId = tmpId;
    if ( !mStatic ) {
      Reinit( fRunId );
      fTask->ReInitTask();
    }
  }
  fTask->ExecuteTask("");
  fRootManager->FillEventHeader(fEvtHeader);
  fTask->FinishTask();
}
//_____________________________________________________________________________


//_____________________________________________________________________________
void O2RunAna::Run(Long64_t entry)
{
  UInt_t tmpId =0;
  fRootManager->ReadEvent(entry);
  tmpId = fEvtHeader->GetRunId();
  if ( tmpId != fRunId ) {
    fRunId = tmpId;
    if ( !mStatic ) {
      Reinit( fRunId );
      fTask->ReInitTask();
    }
  }
  fTask->ExecuteTask("");
  fRootManager->FillEventHeader(fEvtHeader);
  fTask->FinishTask();
  Fill();
  fRootManager->DeleteOldWriteoutBufferData();
  fRootManager->LastFill();
  fRootManager->Write();
}
//_____________________________________________________________________________

//_____________________________________________________________________________
void O2RunAna::RunTSBuffers()
{
  Int_t globalEvent = 0;

  bool firstRun = true;
  while (firstRun || fRootManager->AllDataProcessed() == kFALSE) {
    firstRun = false;
    if (globalEvent < fRootManager->CheckMaxEventNo(0) ) { //this step is necessary to load in all data which is not read in via TSBuffers
      fRootManager->ReadNonTimeBasedEventFromBranches(globalEvent++);
    }
    fTask->ExecuteTask("");
    fRootManager->FillEventHeader(fEvtHeader);
    Fill();
    fRootManager->DeleteOldWriteoutBufferData();
    fTask->FinishEvent();
    if (nullptr !=  FairTrajFilter::Instance()) {
      FairTrajFilter::Instance()->Reset();
    }
  }
  fRootManager->StoreAllWriteoutBufferData();
  fTask->FinishTask();
  fRootManager->LastFill();
  fRootManager->Write();
}
//_____________________________________________________________________________

//_____________________________________________________________________________
void O2RunAna::DummyRun(Int_t Ev_start, Int_t Ev_end)
{
  LOG(INFO) << " DUMMY RUN \n";
  /** This methode is just for testing, if you are not sure about what you do, don't use it */
  for (int i=Ev_start; i< Ev_end; i++) {
    fTask->ExecuteTask("");
    fRootManager->FillEventHeader(fEvtHeader);
    Fill();
  }
  fTask->FinishTask();
  fRootManager->Write();

}
//_____________________________________________________________________________

//_____________________________________________________________________________
void O2RunAna::TerminateRun()
{
  fRootManager->StoreAllWriteoutBufferData();
  fTask->FinishTask();
  gDirectory->SetName(fRootManager->GetOutFile()->GetName());
  //  fRunInfo.WriteInfo(); // CRASHES due to file ownership i guess...
  //   cout << ">>> SlaveTerminate fRootManager->GetInChain()->Print()" << endl;
  //   fRootManager->GetInChain()->Print();
  //   cout << ">>>------------------------------------------------<<<" << endl;
  fRootManager->LastFill();
  fRootManager->Write();
  fRootManager->CloseOutFile();
}
//_____________________________________________________________________________

void O2RunAna::Reinit(UInt_t runId)
{
  // reinit procedure
  fRtdb->initContainers( runId );
}
//_____________________________________________________________________________


//_____________________________________________________________________________
void  O2RunAna::SetContainerStatic(Bool_t tempBool)
{
  mStatic=tempBool;
  if ( mStatic ) {
    LOG(INFO) << "Parameter Cont. initialisation is static" << FairLogger::endl;
  } else {
    LOG(INFO) << "Parameter Cont. initialisation is NOT static" << FairLogger::endl;
  }
}

// BELOW FUNCTIONS SHOULD BE DELETED AND MOVED TO FairFileSource ONLY
//_____________________________________________________________________________
void O2RunAna::SetInputFile(TString name)
{
  LOG(WARNING) << "O2RunAna::SetInputFile is obsolete. Set it by FairFileSource" << FairLogger::endl;
  if ( mMixedSource )
    {
      LOG(ERROR) << "Mixed input already set!" << FairLogger::endl;
      return;
    }
  if ( !mFileSource )
    {
      mFileSource = new FairFileSource(name);
      SetSource(mFileSource);
      return;
    }
  mFileSource->SetInputFile(name);
}
//_____________________________________________________________________________
void O2RunAna::AddFriend (TString name)
{
  LOG(WARNING) << "O2RunAna::AddFriend is obsolete. Set it by FairFileSource" << FairLogger::endl;
  if ( mMixedSource )
    {
      LOG(ERROR) << "Mixed input already set!" << FairLogger::endl;
      return;
    }
  if ( !mFileSource )
    {
      LOG(ERROR) << "Input file not yet set!" << FairLogger::endl;
      return;
    }
  mFileSource->AddFriend(name);
}
//_____________________________________________________________________________
void O2RunAna::AddFile(TString name)
{
  LOG(WARNING) << "O2RunAna::AddFile is obsolete. Set it by FairFileSource" << FairLogger::endl;
  if ( mMixedSource )
    {
      LOG(ERROR) << "Mixed input already set!" << FairLogger::endl;
      return;
    }
  if ( !mFileSource )
    {
      LOG(ERROR) << "Input file not yet set!" << FairLogger::endl;
      return;
    }
  mFileSource->AddFile(name);
}
//_____________________________________________________________________________
// ABOVE FUNCTIONS SHOULD BE DELETED AND MOVED TO FairFileSource ONLY

// BELOW FUNCTIONS SHOULD BE DELETED AND MOVED TO FairMixedSource ONLY
//_____________________________________________________________________________
void O2RunAna::SetSignalFile(TString name, UInt_t identifier )
{
  LOG(WARNING) << "O2RunAna::SetSignalFile is obsolete. Set it by FairMixedSource" << FairLogger::endl;
  if (identifier==0) {
    LOG(FATAL) << " ----- Identifier 0 is reserved for background files! please use other value ------ " << FairLogger::endl;
  }
  if ( mFileSource )
    {
      LOG(ERROR) << "Standard input already set!" << FairLogger::endl;
      return;
    }
  if ( !mMixedSource )
    {
      mMixedSource = new FairMixedSource(name,identifier);
      SetSource(mMixedSource);
      return;
    }
  mMixedSource->AddSignalFile(name, identifier);
}
//_____________________________________________________________________________
void O2RunAna::AddSignalFile(TString name, UInt_t identifier )
{
  LOG(WARNING) << "O2RunAna::AddSignalFile is obsolete. Set it by FairMixedSource" << FairLogger::endl;
  if (identifier==0) {
    LOG(FATAL) << " ----- Identifier 0 is reserved for background files! please use other value ------ " << FairLogger::endl;
  }
  if ( mFileSource )
    {
      LOG(ERROR) << "Standard input already set!" << FairLogger::endl;
      return;
    }
  if ( !mMixedSource )
    {
      mMixedSource = new FairMixedSource(name,identifier);
      SetSource(mMixedSource);
      return;
    }
  mMixedSource->AddSignalFile(name, identifier);
}
//_____________________________________________________________________________
void O2RunAna::SetBackgroundFile(TString name)
{
  LOG(WARNING) << "O2RunAna::SetBackgroundFile is obsolete. Set it by FairMixedSource" << FairLogger::endl;
  if ( mFileSource )
    {
      LOG(ERROR) << "Standard input already set!" << FairLogger::endl;
      return;
    }
  if ( !mMixedSource )
    {
      mMixedSource = new FairMixedSource(name,0);
      SetSource(mMixedSource);
      return;
    }
  mMixedSource->SetBackgroundFile(name);
}
//_____________________________________________________________________________
void O2RunAna::AddBackgroundFile(TString name)
{
  LOG(WARNING) << "O2RunAna::AddBackgroundFile is obsolete. Set it by FairMixedSource" << FairLogger::endl;
  if ( mFileSource )
    {
      LOG(ERROR) << "Standard input already set!" << FairLogger::endl;
      return;
    }
  if ( !mMixedSource )
    {
      LOG(ERROR) << "Background file not yet set!" << FairLogger::endl;
      return;
    }
  mMixedSource->AddBackgroundFile(name);
}
//_____________________________________________________________________________
void  O2RunAna::BGWindowWidthNo(UInt_t background, UInt_t Signalid)
{
  LOG(WARNING) << "O2RunAna::BGWindowWidthNo is obsolete. Set it by FairMixedSource" << FairLogger::endl;
  if ( mFileSource )
    {
      LOG(ERROR) << "Standard input already set!" << FairLogger::endl;
      return;
    }
  if ( !mMixedSource )
    {
      LOG(ERROR) << "Background file not yet set!" << FairLogger::endl;
      return;
    }
  mMixedSource->BGWindowWidthNo(background, Signalid);
}
//_____________________________________________________________________________
void  O2RunAna::BGWindowWidthTime(Double_t background, UInt_t Signalid)
{
  LOG(WARNING) << "O2RunAna::BGWindowWidthTime is obsolete. Set it by FairMixedSource" << FairLogger::endl;
  if ( mFileSource )
    {
      LOG(ERROR) << "Standard input already set!" << FairLogger::endl;
      return;
    }
  if ( !mMixedSource )
    {
      LOG(ERROR) << "Background file not yet set!" << FairLogger::endl;
      return;
    }
  mMixedSource->BGWindowWidthTime(background, Signalid);
}
//_____________________________________________________________________________
// ABOVE FUNCTIONS SHOULD BE DELETED AND MOVED TO FairMixedSource ONLY

// BELOW FUNCTIONS SHOULD BE DELETED AND MOVED TO FairFileSource AND FairMixedSource ONLY
//_____________________________________________________________________________
void O2RunAna::SetEventTimeInterval(Double_t min, Double_t max)
{
  LOG(WARNING) << "O2RunAna::SetEventTimeInterval is obsolete. Set it by FairSource" << FairLogger::endl;
  if ( mFileSource )
    {
      mFileSource->SetEventTimeInterval(min,max);
      return;
    }
  if ( mMixedSource )
    {
      mMixedSource->SetEventTimeInterval(min,max);
      return;
    }
  LOG(ERROR) << "SetEventTimeInterval only by input source!" << FairLogger::endl;
}
//_____________________________________________________________________________
void  O2RunAna::SetEventMeanTime(Double_t mean)
{
  LOG(WARNING) << "O2RunAna::SetEventMeanTime is obsolete. Set it by FairSource" << FairLogger::endl;
  if ( mFileSource )
    {
      mFileSource->SetEventMeanTime(mean);
      return;
    }
  if ( mMixedSource )
    {
      mMixedSource->SetEventMeanTime(mean);
      return;
    }
  LOG(ERROR) << "SetEventMeanTime only by input source!" << FairLogger::endl;
}
//_____________________________________________________________________________
void O2RunAna::SetBeamTime(Double_t beamTime, Double_t gapTime)
{
  LOG(WARNING) << "O2RunAna::SetBeamTime is obsolete. Set it by FairSource" << FairLogger::endl;
  if ( mFileSource )
    {
      mFileSource->SetBeamTime(beamTime, gapTime);
      return;
    }
  if ( mMixedSource )
    {
      mMixedSource->SetBeamTime(beamTime, gapTime);
      return;
    }
  LOG(ERROR) << "SetBeamTime only by input source!" << FairLogger::endl;
}
//_____________________________________________________________________________


//_____________________________________________________________________________
void O2RunAna::Fill()
{
  if(fMarkFill)
  {
    fRootManager->Fill();
  }
  else
  {
    fMarkFill = kTRUE;
  }
}
//_____________________________________________________________________________


// void  O2RunAna::SetMixAllInputs(Bool_t Status)
// {
//    fLogger->Info(MESSAGE_ORIGIN, "Mixing for all input is choosed, in this mode one event per input file is read per step");
//    fRootManager->SetMixAllInputs(Status);
// }
//_____________________________________________________________________________
// ABOVE FUNCTIONS SHOULD BE DELETED AND MOVED TO FairFileSource AND FairMixedSource ONLY


ClassImp(O2RunAna);
