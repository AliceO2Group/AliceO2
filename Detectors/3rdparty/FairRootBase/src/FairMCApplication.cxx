/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----               FairMCApplication source file                   -----
// -----            Created 06/01/04  by M. Al-Turany                  -----
// -------------------------------------------------------------------------
#include "FairMCApplication.h"

#include "FairDetector.h"           // for FairDetector
#include "FairField.h"              // for FairField
#include "FairGenericStack.h"       // for FairGenericStack
#include "FairGeoInterface.h"       // for FairGeoInterface
#include "FairGeoLoader.h"          // for FairGeoLoader
#include "FairGeoMedia.h"           // for FairGeoMedia
#include "FairGeoMedium.h"          // for FairGeoMedium
#include "FairIon.h"                // for FairIon
#include "FairLogger.h"             // for FairLogger, MESSAGE_ORIGIN
#include "FairMCEventHeader.h"      // for FairMCEventHeader
#include "FairMesh.h"               // for FairMesh
#include "FairModule.h"             // for FairModule, etc
#include "FairParticle.h"           // for FairParticle
#include "FairPrimaryGenerator.h"   // for FairPrimaryGenerator
#include "FairRadGridManager.h"     // for FairRadGridManager
#include "FairRadLenManager.h"      // for FairRadLenManager
#include "FairRadMapManager.h"      // for FairRadMapManager
#include "FairRootFileSink.h"       // for CloneForWorker (in MT mode only)
#include "FairRootManager.h"        // for FairRootManager
#include "FairRun.h"                // for FairRun
#include "FairRunInfo.h"            // for FairRunInfo
#include "FairRunSim.h"             // for FairRunSim
#include "FairTask.h"               // for FairTask
#include "FairTrajFilter.h"         // for FairTrajFilter
#include "FairVolume.h"             // for FairVolume

#include <TDatabasePDG.h>       // for TDatabasePDG
#include <TFile.h>              // for TFile
#include <TGeoManager.h>        // for gGeoManager, TGeoManager
#include <TGeoMedium.h>         // for TGeoMedium
#include <TGeoNode.h>           // for TGeoNode
#include <TGeoPhysicalNode.h>   // for TGeoPhysicalNode
#include <TGeoTrack.h>          // for TGeoTrack
#include <TGeoVolume.h>         // for TGeoVolume
#include <TH2.h>                // for TH2D
#include <THashList.h>          // for THashList
#include <TInterpreter.h>       // for TInterpreter, gInterpreter
#include <TIterator.h>          // for TIterator
#include <TList.h>              // for TList, TListIter
#include <TObjArray.h>          // for TObjArray
#include <TObject.h>            // for TObject
#include <TParticlePDG.h>       // for TParticlePDG
#include <TROOT.h>              // for TROOT, gROOT
#include <TRefArray.h>          // for TRefArray
#include <TSystem.h>            // for TSystem, gSystem
#include <TVirtualMC.h>         // for TVirtualMC
#include <TVirtualMCStack.h>    // for TVirtualMCStack
class TParticle;

#include <float.h>    // for DBL_MAX
#include <mutex>      // std::mutex
#include <stdlib.h>   // for getenv, exit
#include <utility>    // for pair
std::mutex mtx;       // mutex for critical section

using std::pair;

FairMCApplication* FairMCApplication::fgMasterInstance = 0;

//_____________________________________________________________________________
FairMCApplication::FairMCApplication(const char* name, const char* title, TObjArray* ModList, const char*)
    : TVirtualMCApplication(name, title)
    , fActiveDetectors(nullptr)
    , fFairTaskList(nullptr)
    , fDetectors(nullptr)
    , fDetMap(nullptr)
    , fModIter(nullptr)
    , fModules(nullptr)
    , fNoSenVolumes(0)
    , fPythiaDecayer(kFALSE)
    , fPythiaDecayerConfig("")
    , fStack(nullptr)
    , fRootManager(nullptr)
    , fSenVolumes(nullptr)
    , fxField(nullptr)
    , fEvGen(nullptr)
    , fMcVersion(-1)
    , fTrajFilter(nullptr)
    , fTrajAccepted(kFALSE)
    , fUserDecay(kFALSE)
    , fUserDecayConfig("")
    , fDebug(kFALSE)
    , fDisVol(nullptr)
    , fDisDet(nullptr)
    , fVolMap()
    , fVolIter()
    , fModVolMap()
    , fModVolIter()
    , fTrkPos(TLorentzVector(0, 0, 0, 0))
    , fRadLength(kFALSE)
    , fRadLenMan(nullptr)
    , fRadMap(kFALSE)
    , fRadMapMan(nullptr)
    , fRadGridMan(nullptr)
    , fEventHeader(nullptr)
    , fMCEventHeader(nullptr)
    , listActiveDetectors()
    , listDetectors()
    , fMC(nullptr)
    , fRun(nullptr)
    , fSaveCurrentEvent(kTRUE)
    , fState(FairMCApplicationState::kUnknownState)
    , fRunInfo()
    , fGeometryIsInitialized(kFALSE)
{
    // Standard Simulation constructor

    // Create an ObjArray of Modules and its iterator

    LOG(debug) << "FairMCApplication-ctor " << this;

    fRun = FairRunSim::Instance();
    fModules = ModList;
    fModIter = fModules->MakeIterator();
    // Create and fill a list of active detectors
    fDetectors = new TRefArray;
    fActiveDetectors = new TRefArray();
    fModIter->Reset();
    FairDetector* detector;
    TObject* obj;

    while ((obj = fModIter->Next())) {
        detector = dynamic_cast<FairDetector*>(obj);
        if (detector) {
            fDetectors->Add(detector);
            listDetectors.push_back(detector);
            if (detector->IsActive()) {
                fActiveDetectors->Add(detector);
                listActiveDetectors.push_back(detector);
            }
        } else if (!dynamic_cast<FairModule*>(obj)) {
            LOG(error) << "Dynamic cast fails. Object neither FairDetector nor FairModule in module list";
        }
    }

    // Create a Task list
    fFairTaskList = new FairTask("Task List", 1);
    gROOT->GetListOfBrowsables()->Add(fFairTaskList);
    fMcVersion = -1;
    // Initialise fTrajFilter pointer
    fTrajFilter = nullptr;
    fDetMap = new TRefArray(1000);
    fDisVol = 0;
    fDisDet = 0;

    // This ctor is used to construct the application on master
    fgMasterInstance = this;
}

//_____________________________________________________________________________
FairMCApplication::FairMCApplication(const FairMCApplication& rhs)
    : TVirtualMCApplication(rhs.GetName(), rhs.GetTitle())
    , fActiveDetectors(nullptr)
    , fFairTaskList(nullptr)
    , fDetectors(nullptr)
    , fDetMap(nullptr)
    , fModIter(nullptr)
    , fModules(nullptr)
    , fNoSenVolumes(0)
    , fPythiaDecayer(kFALSE)
    , fPythiaDecayerConfig(rhs.fPythiaDecayerConfig)
    , fStack(nullptr)
    , fRootManager(nullptr)
    , fSenVolumes(nullptr)
    , fxField(rhs.fxField)
    , fEvGen(nullptr)
    , fMcVersion(rhs.fMcVersion)
    , fTrajFilter(nullptr)
    , fTrajAccepted(kFALSE)
    , fUserDecay(kFALSE)
    , fUserDecayConfig(rhs.fUserDecayConfig)
    , fDebug(rhs.fDebug)
    , fDisVol(nullptr)
    , fDisDet(nullptr)
    , fVolMap()
    , fVolIter()
    , fModVolMap()
    , fModVolIter()
    , fTrkPos(rhs.fTrkPos)
    , fRadLength(kFALSE)
    , fRadLenMan(nullptr)
    , fRadMap(kFALSE)
    , fRadMapMan(nullptr)
    , fRadGridMan(nullptr)
    , fEventHeader(nullptr)
    , fMCEventHeader(nullptr)
    , listActiveDetectors()
    , listDetectors()
    , fMC(nullptr)
    , fRun(nullptr)
    , fSaveCurrentEvent(kTRUE)
    , fState(FairMCApplicationState::kUnknownState)
    , fRunInfo()
    , fGeometryIsInitialized(kFALSE)
{
    // Copy constructor
    // Do not create Root manager

    LOG(debug) << "FairMCApplication-copy-ctor " << this;

    // Create an ObjArray of Modules and its iterator
    fModules = new TObjArray();
    fModIter = fModules->MakeIterator();
    // Clone modules
    TObject* obj;
    rhs.fModIter->Reset();
    while ((obj = rhs.fModIter->Next())) {
        LOG(debug) << "cloning " << (static_cast<FairModule*>(obj))->GetName();
        fModules->Add(static_cast<FairModule*>(obj)->CloneModule());
    }

    // Create and fill a list of active detectors
    fDetectors = new TRefArray;
    fActiveDetectors = new TRefArray();
    fModIter->Reset();
    FairDetector* detector;
    while ((obj = fModIter->Next())) {
        if (obj->InheritsFrom("FairDetector")) {
            detector = dynamic_cast<FairDetector*>(obj);
            if (detector) {
                fDetectors->Add(detector);
                listDetectors.push_back(detector);
                if (detector->IsActive()) {
                    fActiveDetectors->Add(detector);
                    listActiveDetectors.push_back(detector);
                }
            } else {
                LOG(error) << "Dynamic cast fails.";
            }
        }
    }

    // Clone stack
    fStack = rhs.fStack->CloneStack();

    // Create a Task list
    // Let's try without it
    // fFairTaskList= new FairTask("Task List", 1);
    // gROOT->GetListOfBrowsables()->Add(fFairTaskList);

    fDetMap = new TRefArray(1000);
}
//_____________________________________________________________________________
FairMCApplication::FairMCApplication()
    : TVirtualMCApplication()
    , fActiveDetectors(0)
    , fFairTaskList(0)
    , fDetectors(0)
    , fDetMap(0)
    , fModIter(0)
    , fModules(0)
    , fNoSenVolumes(0)
    , fPythiaDecayer(kFALSE)
    , fPythiaDecayerConfig("")
    , fStack(0)
    , fRootManager(0)
    , fSenVolumes(0)
    , fxField(0)
    , fEvGen(0)
    , fMcVersion(-1)
    , fTrajFilter(nullptr)
    , fTrajAccepted(kFALSE)
    , fUserDecay(kFALSE)
    , fUserDecayConfig("")
    , fDebug(kFALSE)
    , fDisVol(0)
    , fDisDet(0)
    , fVolMap()
    , fVolIter()
    , fModVolMap()
    , fModVolIter()
    , fTrkPos(TLorentzVector(0, 0, 0, 0))
    , fRadLength(kFALSE)
    , fRadLenMan(nullptr)
    , fRadMap(kFALSE)
    , fRadMapMan(nullptr)
    , fRadGridMan(nullptr)
    , fEventHeader(nullptr)
    , fMCEventHeader(nullptr)
    , listActiveDetectors()
    , listDetectors()
    , fMC(nullptr)
    , fRun(nullptr)
    , fSaveCurrentEvent(kTRUE)
    , fState(FairMCApplicationState::kUnknownState)
    , fRunInfo()
    , fGeometryIsInitialized(kFALSE)
{
    // Default constructor
}
//_____________________________________________________________________________
FairMCApplication::~FairMCApplication()
{
    // Destructor
    //   LOG(debug3) << "Enter Destructor of FairMCApplication";
    delete fStack;
    delete fActiveDetectors;   // don't do fActiveDetectors->Delete() here
    // the modules are already deleted in FairRunSim
    delete fDetectors;
    delete fModIter;
    //  LOG(debug3) << "Leave Destructor of FairMCApplication";
    delete fMC;
}

//_____________________________________________________________________________
FairMCApplication& FairMCApplication::operator=(const FairMCApplication& rhs)
{
    // Assignment operator

    // check assignment to self
    if (this != &rhs) {

        // base class assignment
        TVirtualMCApplication::operator=(rhs);

        fActiveDetectors = nullptr;
        fFairTaskList = nullptr;
        fDetectors = nullptr;
        fDetMap = nullptr;
        fModIter = nullptr;
        fModules = nullptr;
        fNoSenVolumes = 0;
        fPythiaDecayer = kFALSE;
        fPythiaDecayerConfig = rhs.fPythiaDecayerConfig;
        fStack = nullptr;
        fRootManager = nullptr;
        fSenVolumes = nullptr;
        fxField = rhs.fxField;
        fEvGen = nullptr;
        fMcVersion = rhs.fMcVersion;
        fTrajFilter = nullptr;
        fTrajAccepted = kFALSE;
        fUserDecay = kFALSE;
        fUserDecayConfig = rhs.fUserDecayConfig;
        fDebug = rhs.fDebug;
        fDisVol = nullptr;
        fDisDet = nullptr;
        fTrkPos = rhs.fTrkPos;
        fRadLength = kFALSE;
        fRadLenMan = nullptr;
        fRadMap = kFALSE;
        fRadMapMan = nullptr;
        fRadGridMan = nullptr;
        fEventHeader = nullptr;
        fMCEventHeader = nullptr;
        fGeometryIsInitialized = kFALSE;

        // Do not create Root manager

        // Create an ObjArray of Modules and its iterator
        fModules = new TObjArray();
        fModIter = fModules->MakeIterator();
        // Clone modules
        TObject* obj;
        while ((obj = rhs.fModIter->Next())) {
            fModules->Add(static_cast<FairModule*>(obj)->CloneModule());
        }

        // Create and fill a list of active detectors
        fDetectors = new TRefArray;
        fActiveDetectors = new TRefArray();
        fModIter->Reset();
        FairDetector* detector;
        while ((obj = fModIter->Next())) {
            if (obj->InheritsFrom("FairDetector")) {
                detector = dynamic_cast<FairDetector*>(obj);
                if (detector) {
                    fDetectors->Add(detector);
                    listDetectors.push_back(detector);
                    if (detector->IsActive()) {
                        fActiveDetectors->Add(detector);
                        listActiveDetectors.push_back(detector);
                    }
                } else {
                    LOG(error) << "Dynamic cast fails.";
                }
            }
        }

        // Clone stack
        fStack = rhs.fStack->CloneStack();

        // Create a Task list
        // Let's try without it
        // fFairTaskList= new FairTask("Task List", 1);
        // gROOT->GetListOfBrowsables()->Add(fFairTaskList);

        fDetMap = new TRefArray(1000);

        fState = rhs.fState;
    }

    return *this;
}

//_____________________________________________________________________________
void FairMCApplication::InitMC(const char*, const char*)
{
    // Initialize MC.
    // ---
    // This methode is called from FairRunSim::SetMCConfig which excucute first the gconfig
    // macro that creates the MC instance (G3 or G4)

    fMC = TVirtualMC::GetMC();

    if (fMC == 0) {
        LOG(fatal) << "No MC engine defined";
    }

    fStack = dynamic_cast<FairGenericStack*>(fMC->GetStack());
    if (fStack == nullptr) {
        LOG(fatal) << "No Stack defined.";
    }
    fMC->SetMagField(fxField);

    fRootManager = FairRootManager::Instance();
    // fRootManager->SetDebug(true);

    fMC->Init();
    fMC->BuildPhysics();
    TString MCName = fMC->GetName();
    if (MCName == "TGeant3" || MCName == "TGeant3TGeo") {
        fMcVersion = 0;
    } else if (MCName == "TGeant4") {
        fMcVersion = 1;
    } else if (MCName == "TFluka") {
        fMcVersion = 2;
    } else {
        fMcVersion = 3;   // Geane
    }
    fTrajFilter = FairTrajFilter::Instance();

    LOG(info) << "Monte Carlo Engine Initialisation with: " << MCName.Data();
}

//_____________________________________________________________________________
void FairMCApplication::RunMC(Int_t nofEvents)
{
    // Reset the time for FairRunInfo. Otherwise the time of the
    // first event will include the time needed for initilization.
    fRunInfo.Reset();

    /** Set the list of active detectors to the stack*/
    fStack->SetDetArrayList(fActiveDetectors);

    // MC run.
    fMC->ProcessRun(nofEvents);
    // finish run
    FinishRun();
}

//____________________________________________________________________________
void FairMCApplication::FinishRun()
{
    // Finish MC run.
    // ---

    LOG(debug) << "FairMCMCApplication::FinishRun() start";

    for (auto detectorPtr : listActiveDetectors) {
        detectorPtr->FinishRun();
    }

    // TO DO: clone this in MT mode ?
    if (fFairTaskList) {
        fFairTaskList->FinishTask();
    }
    // fRootManager->Fill();

    FairPrimaryGenerator* gen = FairRunSim::Instance()->GetPrimaryGenerator();
    // FairMCEventHeader* header = gen->GetEvent();
    Int_t nprimary = gen->GetTotPrimary();
    TObjArray* meshlist = nullptr;

    // Only in sequential mode
    if (fRadGridMan) {

        meshlist = fRadGridMan->GetMeshList();

        TH2D* tid = nullptr;
        TH2D* flu = nullptr;
        TH2D* seu = nullptr;

        LOG(info) << "=======================================================";
        LOG(info) << "   Dosimetry  histos saving in \"" << fRadGridMan->GetOutputFileName() << "\"";
        LOG(info) << "=======================================================";

        TFile* radGridFile = TFile::Open(fRadGridMan->GetOutputFileName(), "recreate");
        radGridFile->mkdir("Dosimetry");
        radGridFile->cd("Dosimetry");

        for (Int_t i = 0; i < meshlist->GetEntriesFast(); i++) {
            FairMesh* aMesh = dynamic_cast<FairMesh*>(meshlist->At(i));
            if (aMesh) {
                aMesh->Scale(1. / nprimary);
                tid = aMesh->GetMeshTid();
                flu = aMesh->GetMeshFlu();
                seu = aMesh->GetMeshSEU();
                tid->Write();
                flu->Write();
                seu->Write();
            }
        }
        radGridFile->Write();
        radGridFile->Close();
        delete radGridFile;
    }

    // Save histograms with memory and runtime information in the output file
    if (FairRunSim::Instance()->IsRunInfoGenerated()) {
        fRunInfo.WriteInfo();
    }

    if (!fRadGridMan && fRootManager) {
        fRootManager->Write();
        fRootManager->CloseSink();
    }

    if (!fMC->IsMT()) {
        UndoGeometryModifications();
    }

    LOG(debug) << "Done FairMCMCApplication::FinishRun()";
}

//_____________________________________________________________________________
void FairMCApplication::BeginEvent()
{
    // User actions at beginning of event
    // ---

    for (auto detectorPtr : listActiveDetectors) {
        detectorPtr->BeginEvent();
    }
}

//_____________________________________________________________________________
void FairMCApplication::BeginPrimary()
{
    // User actions at beginning of a primary track
    // ---
    for (auto detectorPtr : listActiveDetectors) {
        detectorPtr->BeginPrimary();
    }
}

//_____________________________________________________________________________
void FairMCApplication::PreTrack()
{
    // User actions at beginning of each track
    // ---

    for (auto detectorPtr : listActiveDetectors) {
        detectorPtr->PreTrack();
    }

    fTrajAccepted = kFALSE;
    if (nullptr != fTrajFilter) {
        // Get the pointer to current track
        TParticle* particle = fStack->GetCurrentTrack();
        //  LOG(debug) << " FairMCApplication::PreTrack(): " << particle;
        // Apply cuts
        fTrajAccepted = fTrajFilter->IsAccepted(particle);
        if (fTrajAccepted) {
            // Add trajectory to geo manager
            //    Int_t trackId = fStack->GetCurrentTrackNumber();
            TGeoTrack* fTrack = fTrajFilter->CheckAddTrack(fMC->GetStack()->GetCurrentTrackNumber(), particle);
            // TLorentzVector pos;
            fMC->TrackPosition(fTrkPos);
            fTrack->AddPoint(fTrkPos.X(), fTrkPos.Y(), fTrkPos.Z(), fTrkPos.T());
        }
    }
}

//_____________________________________________________________________________
TVirtualMCApplication* FairMCApplication::CloneForWorker() const
{
    mtx.lock();
    LOG(info) << "FairMCApplication::CloneForWorker ";

    // Create new FairRunSim object on worker
    // and pass some data from master FairRunSim object
    FairRunSim* workerRun = new FairRunSim(kFALSE);
    workerRun->SetName(fRun->GetName());   // Transport engine
    workerRun->SetSink(fRun->GetSink()->CloneSink());

    // Trajectories filter is created explicitly as we do not call
    // FairRunSim::Init on workers
    if (fRun->GetStoreTraj()) {
        new FairTrajFilter();
    }

    // Create new FairMCApplication object on worker
    FairMCApplication* workerApplication = new FairMCApplication(*this);
    workerApplication->SetGenerator(fEvGen->ClonePrimaryGenerator());

    mtx.unlock();

    return workerApplication;
}

//_____________________________________________________________________________
void FairMCApplication::InitOnWorker()
{
    // Create Root manager
    fRootManager = FairRootManager::Instance();

    LOG(info) << "FairMCApplication::InitForWorker " << fRootManager->GetInstanceId() << " " << this;

    // Set FairRunSim worker(just for consistency, not needed on worker)
    fRun = FairRunSim::Instance();

    // Generate per-thread file name
    // and create a new sink on worker
    // moved to CloneForWorker and specific sink->CloneSink();

    fRootManager->InitSink();

    // Cache thread-local gMC
    fMC = gMC;

    // Set data to MC
    fMC->SetStack(fStack);
    fMC->SetMagField(fxField);

    LOG(info) << "Monte Carlo Engine Worker Initialisation  with: " << fMC->GetName();
}

//_____________________________________________________________________________
void FairMCApplication::FinishRunOnWorker()
{
    LOG(debug) << "FairMCApplication::FinishRunOnWorker: ";

    FinishRun();
}

//_____________________________________________________________________________
void FairMCApplication::Stepping()
{
    // User actions at each step
    // ---

    // Work around for Fluka VMC, which does not call
    // MCApplication::PreTrack()
    static Int_t TrackId = 0;
    if (fMcVersion == 2 && fMC->GetStack()->GetCurrentTrackNumber() != TrackId) {
        PreTrack();
        TrackId = fMC->GetStack()->GetCurrentTrackNumber();
    }

    // Check if the volume with id is in the volume multimap.
    // If it is not in the map the volume is not a sensitive volume
    // and we do not call nay of our ProcessHits functions.

    // If the volume is in the multimap, check in second step if the current
    // copy is alredy inside the multimap.
    // If the volume is not in the multimap add the copy of the volume to the
    // multimap.
    // In any case call the ProcessHits function for this specific detector.
    Int_t copyNo;
    Int_t id = fMC->CurrentVolID(copyNo);
    Bool_t InMap = kFALSE;
    fDisVol = 0;
    fDisDet = 0;
    Int_t fCopyNo = 0;
    fVolIter = fVolMap.find(id);

    if (fVolIter != fVolMap.end()) {

        // Call Process hits for FairVolume with this id, copyNo
        do {
            fDisVol = fVolIter->second;
            fCopyNo = fDisVol->getCopyNo();
            if (copyNo == fCopyNo) {
                fDisDet = fDisVol->GetDetector();
                if (fDisDet) {
                    fDisDet->ProcessHits(fDisVol);
                }
                InMap = kTRUE;
                break;
            }
            ++fVolIter;
        } while (fVolIter != fVolMap.upper_bound(id));

        //    if (fDisVol && !InMap) { // fDisVolume is set previously, no check needed

        // Create new FairVolume with this id, copyNo.
        // Use the FairVolume with the same id found in the map to get
        // the link to the detector.
        // Seems that this never happens (?)
        if (!InMap) {
            // cout << "Volume not in map; fDisVol ? " << fDisVol << endl
            FairVolume* fNewV = new FairVolume(fMC->CurrentVolName(), id);
            fNewV->setMCid(id);
            fNewV->setModId(fDisVol->getModId());
            fNewV->SetModule(fDisVol->GetModule());
            fNewV->setCopyNo(copyNo);
            fVolMap.insert(pair<Int_t, FairVolume*>(id, fNewV));
            fDisDet = fDisVol->GetDetector();

            // LOG(info) << "FairMCApplication::Stepping: new fair volume"
            //    << id << " " << copyNo << " " <<  fDisDet;
            if (fDisDet) {
                fDisDet->ProcessHits(fNewV);
            }
        }
    }

    // If information about the tracks should be stored the information as to be
    // stored for any step.
    // Information about each single step has also to be stored for the other
    // special run modes of the simulation which are used to store information
    // about
    // 1.) Radiation length in each volume
    // 2.) Energy deposition in each volume
    // 3.) Fluence of particles through a defined plane which can be anywhere
    //     in the geometry. This plane has not to be correlated with any real
    //     volume
    if (fTrajAccepted) {
        if (fMC->TrackStep() > fTrajFilter->GetStepSizeCut()) {
            fMC->TrackPosition(fTrkPos);
            fTrajFilter->GetCurrentTrk()->AddPoint(fTrkPos.X(), fTrkPos.Y(), fTrkPos.Z(), fTrkPos.T());
        }
    }
    if (fRadLenMan) {
        id = fMC->CurrentVolID(copyNo);
        fModVolIter = fgMasterInstance->fModVolMap.find(id);
        fRadLenMan->AddPoint(fModVolIter->second);
    }
    if (fRadMapMan) {
        id = fMC->CurrentVolID(copyNo);
        fModVolIter = fgMasterInstance->fModVolMap.find(id);
        fRadMapMan->AddPoint(fModVolIter->second);
    }
    if (fRadGridMan) {
        fRadGridMan->FillMeshList();
    }
}

//_____________________________________________________________________________
void FairMCApplication::PostTrack()
{
    // User actions after finishing of each track
    // ---
    for (auto detectorPtr : listActiveDetectors) {
        detectorPtr->PostTrack();
    }
}

//_____________________________________________________________________________
void FairMCApplication::FinishPrimary()
{
    // User actions after finishing of a primary track
    // ---
    for (auto detectorPtr : listActiveDetectors) {
        detectorPtr->FinishPrimary();
    }

    fStack->FinishPrimary();
}

//_____________________________________________________________________________
void FairMCApplication::StopRun()
{
    FinishEvent();
    FinishRun();
    if (fRootManager) {
        fRootManager->Write();
        fRootManager->CloseSink();
    }
    LOG(warn) << "StopRun() exiting not safetly oopps !!!@@@!!!";
    exit(0);
}

//_____________________________________________________________________________
void FairMCApplication::StopMCRun()
{
    if (fMC)
        fMC->StopRun();
}

//_____________________________________________________________________________
void FairMCApplication::FinishEvent()
{
    // User actions after finishing of an event
    // ---
    LOG(debug) << "[" << fRootManager->GetInstanceId()
               << " FairMCMCApplication::FinishEvent: " << fMCEventHeader->GetEventID() << " (MC "
               << gMC->CurrentEvent() << ")";
    if (gMC->IsMT()
        && fRun->GetSink()->GetSinkType() == kONLINESINK) {   // fix the rare case when running G4 multithreaded on MQ
        fMCEventHeader->SetEventID(gMC->CurrentEvent() + 1);
    }

    // --> Fill the stack output array
    fStack->FillTrackArray();
    // --> Update track indizes in MCTracks and MCPoints
    fStack->UpdateTrackIndex(fActiveDetectors);
    // --> Screen output of stack
    // fStack->Print();

    if (fFairTaskList) {
        fFairTaskList->ExecuteTask("");
        fFairTaskList->FinishEvent();
    }

    for (auto detectorPtr : listActiveDetectors) {
        detectorPtr->FinishEvent();
    }

    if (fRootManager && fSaveCurrentEvent) {
        fRootManager->Fill();
    } else {
        fSaveCurrentEvent = kTRUE;
    }

    for (auto detectorPtr : listActiveDetectors) {
        detectorPtr->EndOfEvent();
    }

    fStack->Reset();
    if (nullptr != fTrajFilter) {
        fTrajFilter->Reset();
        //    TObjArray* fListOfTracks=gGeoManager->GetListOfTracks();
        //    fListOfTracks->Delete();
        gGeoManager->GetListOfTracks()->Delete();
    }
    if (nullptr != fRadLenMan) {
        fRadLenMan->Reset();
    }
    if (nullptr != fRadMapMan) {
        fRadMapMan->Reset();
    }

    // Store information about runtime for one event and memory consuption
    // for later usage.
    if ((FairRunSim::Instance()->IsRunInfoGenerated()) && !gMC->IsMT()) {
        fRunInfo.StoreInfo();
    }
}
//_____________________________________________________________________________
Double_t FairMCApplication::TrackingRmax() const
{
    // No limit
    // ---
    return DBL_MAX;
}
//_____________________________________________________________________________
Double_t FairMCApplication::TrackingZmax() const
{
    // No limit
    // ---
    return DBL_MAX;
}
//_____________________________________________________________________________
void FairMCApplication::SetField(FairField* field) { fxField = field; }
//_____________________________________________________________________________
void FairMCApplication::ConstructOpGeometry()
{
    // The optical properties of ASCII-defined media used to be pushed into the
    // MC here (TVirtualMC::SetCerenkov) by walking FairGeoLoader's media list.
    // ALICE never loads an ASCII media file, so that list was always empty and
    // the loop was a no-op - but it dereferenced FairGeoLoader::Instance()
    // unconditionally, which meant the singleton had to be constructed even
    // though it did nothing. Detectors that need optical media (HMPID) define
    // them in their own ConstructOpGeometry() override, dispatched below.
    fModIter->Reset();
    FairModule* Mod = nullptr;
    while ((Mod = dynamic_cast<FairModule*>(fModIter->Next()))) {
        Mod->ConstructOpGeometry();
    }
}

//_____________________________________________________________________________
void FairMCApplication::ConstructGeometry()
{
    // Construct geometry and also fill following member data:
    // - fModVolMap: (volId,moduleId)
    // - fSenVolumes: list of sensitive volumes
    if (!gGeoManager) {
        LOG(fatal) << "gGeoManager not initialized at FairMCApplication::ConstructGeometry\n";
    }

    fState = FairMCApplicationState::kConstructGeometry;

    fModIter->Reset();
    FairModule* Mod = nullptr;
    Int_t NoOfVolumes = 0;
    Int_t NoOfVolumesBefore = 0;
    Int_t ModId = 0;

    TObjArray* tgeovolumelist = gGeoManager->GetListOfVolumes();

    while ((Mod = dynamic_cast<FairModule*>(fModIter->Next()))) {
        NoOfVolumesBefore = tgeovolumelist->GetEntriesFast();
        Mod->ConstructGeometry();
        ModId = Mod->GetModId();
        NoOfVolumes = tgeovolumelist->GetEntriesFast();
        for (Int_t n = NoOfVolumesBefore; n < NoOfVolumes; n++) {
            TGeoVolume* v = (TGeoVolume*)tgeovolumelist->At(n);
            fModVolMap.insert(pair<Int_t, Int_t>(v->GetNumber(), ModId));
        }
    }

    //  LOG(debug) << "FairMCApplication::ConstructGeometry() : Now closing the geometry";
    int NoOfVolumesBeforeClose = tgeovolumelist->GetEntries();
    gGeoManager->CloseGeometry();   // close geometry
    int NoOfVolumesAfterClose = tgeovolumelist->GetEntries();

    // Check if CloseGeometry has modified the volume list which might happen for
    // runtime shapes (parametrizations,etc). If this is the case, our lookup structures
    // built above are likely out of date. Issue at least a warning here.
    if (NoOfVolumesBeforeClose != NoOfVolumesAfterClose) {
        LOG(error) << "TGeoManager::CloseGeometry() modified the volume list from " << NoOfVolumesBeforeClose << " to "
                   << NoOfVolumesAfterClose << "\n"
                   << "This almost certainly means inconsistent lookup structures used in simulation/stepping.\n";
    }

    if (fRun->IsImportTGeoToVMC()) {
        TVirtualMC::GetMC()->SetRootGeometry();   // notify VMC about Root geometry
        LOG(info) << "TGeometry will be imported to VMC"
                  << "\n";
    } else {
        LOG(info) << "TGeometry will not be imported to VMC"
                  << "\n";
    }
    Int_t Counter = 0;
    TDatabasePDG* pdgDatabase = TDatabasePDG::Instance();
    const THashList* list = pdgDatabase->ParticleList();
    if (list == 0)
        pdgDatabase->ReadPDGTable();
    list = pdgDatabase->ParticleList();
    if (list != 0) {
        TIterator* particleIter = list->MakeIterator();
        TParticlePDG* Particle = 0;
        while ((Particle = dynamic_cast<TParticlePDG*>(particleIter->Next())) && (Counter <= 256)) {
            TString Name = gGeoManager->GetPdgName(Particle->PdgCode());
            //    LOG(info) << Counter <<" : Particle name: "<< Name.Data() << " PDG " << Particle->PdgCode();
            if (Name == "XXX")
                gGeoManager->SetPdgName(Particle->PdgCode(), Particle->GetName());
            Counter++;
        }
        delete particleIter;
    }
    fModIter->Reset();
    while ((Mod = dynamic_cast<FairModule*>(fModIter->Next()))) {
        Mod->RegisterAlignmentMatrices();
    }

    // dont use this here anymore, use FairMCApplication::MisalignGeometry
    // fRun->AlignGeometry()

    gGeoManager->RefreshPhysicalNodes(kFALSE);

    fState = FairMCApplicationState::kUnknownState;
}

// ____________________________________________________________________________
Bool_t FairMCApplication::MisalignGeometry()
{
    // call this only here
    fRun->AlignGeometry();
    return true;
}

//_____________________________________________________________________________
void FairMCApplication::InitGeometry()
{
    fState = FairMCApplicationState::kInitGeometry;

    LOG(info) << "FairMCApplication::InitGeometry: " << fRootManager->GetInstanceId();

    // ToBeDone
    // Recently the InitGeometry is called twice from the G4VMC, This is a work around tell the problem get fixed in
    // G4VMC
    if (fGeometryIsInitialized)
        return;

    /// Initialize geometry

    /** Register stack and detector collections*/
    FairVolume* fv = 0;
    Int_t id = 0;
    fModIter->Reset();

    // Register stack
    if (fEvGen && fStack && fRootManager) {
        fStack->Register();
    } else {
        LOG(warn) << "Stack is not registered ";
    }

    /** SetSpecialPhysicsCuts for FairDetector objects and all passive modules inheriting from FairModule */
    // initialize and register FairDetector objects in addition
    // Note: listActiveDetectors or fActiveDetectors not used to include passive modules in the same loop.
    FairModule* module;
    FairDetector* detector;
    TObject* obj;
    fModIter->Reset();
    while ((obj = fModIter->Next())) {
        detector = dynamic_cast<FairDetector*>(obj);
        module = dynamic_cast<FairModule*>(obj);
        if (module) {
            module->SetSpecialPhysicsCuts();
        }
        if (detector) {
            // check whether detector is active
            if (detector->IsActive()) {
                detector->Initialize();
                detector->Register();
            }
        }
    }
    fModIter->Reset();

    /**Tasks has to be initialized here, they have access to the detector branches and still can create objects in the
     * tree*/
    /// There is always a Main Task  !
    /// so .. always a InitTasks() is called <D.B>
    if (fFairTaskList) {
        InitTasks();
    }

    // store the EventHeader Info
    // Get and register EventHeader
    UInt_t runId = FairRunSim::Instance()->GetRunId();

    LOG(info) << "Simulation RunID: " << runId;

    // Get and register the MCEventHeader
    fMCEventHeader = FairRunSim::Instance()->GetMCEventHeader();
    fMCEventHeader->SetRunID(runId);
    if (fRootManager) {
        fMCEventHeader->Register();
    }

    if (fEvGen) {
        fEvGen->SetEvent(fMCEventHeader);
    }
    fTrajFilter = FairTrajFilter::Instance();
    if (nullptr != fTrajFilter) {
        fTrajFilter->Init();
    }
    if (nullptr != fRadLenMan) {
        fRadLenMan->Init();
    }
    if (nullptr != fRadMapMan) {
        fRadMapMan->Init();
    }
    if (nullptr != fRadGridMan) {
        fRadGridMan->Init();
    }

    /// save Geo Params in Output file
    if (fRootManager) {
        fRootManager->WriteFolder();
    }

    // Get static thread local svList
    fSenVolumes = FairModule::svList;
    if (fSenVolumes) {
        fNoSenVolumes = fSenVolumes->GetEntries();
    }

    // Fill sensitive volumes in fVolMap
    for (Int_t i = 0; i < fNoSenVolumes; i++) {

        fv = dynamic_cast<FairVolume*>(fSenVolumes->At(i));
        if (!fv) {
            LOG(error) << "No FairVolume in fSenVolumes at position " << i;
            continue;
        }
        id = fv->getMCid();
        if (fv->getGeoNode() == 0) {
            TGeoNode* fN = 0;
            TGeoVolume* v = gGeoManager->GetVolume(fv->GetName());
            TObjArray* fNs = 0;
            if (v) {
                fNs = v->GetNodes();
            }
            if (fNs) {
                for (Int_t k = 0; k < fNs->GetEntriesFast(); k++) {
                    fN = dynamic_cast<TGeoNode*>(fNs->At(k));
                    if (!fN) {
                        LOG(error) << "No TGeoNode in fNs at position " << k;
                        continue;
                    }
                    FairVolume* fNewV = new FairVolume(fv->GetName(), id);
                    fNewV->setModId(fv->getModId());
                    fNewV->SetModule(fv->GetModule());
                    fNewV->setCopyNo(fN->GetNumber());
                    fNewV->setMCid(id);
                    fVolMap.insert(pair<Int_t, FairVolume*>(id, fNewV));
                }
            } else {
                FairVolume* fNewV = new FairVolume(fv->GetName(), id);
                fNewV->setModId(fv->getModId());
                fNewV->SetModule(fv->GetModule());
                fNewV->setCopyNo(1);
                fNewV->setMCid(id);
                fVolMap.insert(pair<Int_t, FairVolume*>(id, fNewV));
            }
        } else {
            fVolMap.insert(pair<Int_t, FairVolume*>(id, fv));
        }
    }   // end off loop Fill sensitive volumes

    fGeometryIsInitialized = kTRUE;

    fState = FairMCApplicationState::kUnknownState;
}

//_____________________________________________________________________________
void FairMCApplication::GeneratePrimaries()
{
    // Fill the user stack (derived from TVirtualMCStack) with primary particles.
    // ---
    LOG(debug) << "FairMCApplication::GeneratePrimaries: " << fEvGen;

    if (fEvGen) {
        //    LOG(debug) << "FairMCApplication::GeneratePrimaries()";
        if (!fEvGen->GenerateEvent(fStack)) {
            StopRun();
        }
    }
}
//_____________________________________________________________________________
FairDetector* FairMCApplication::GetDetector(const char* DetName)
{
    return dynamic_cast<FairDetector*>(fModules->FindObject(DetName));
}
//_____________________________________________________________________________
void FairMCApplication::AddIons()
{
    TDatabasePDG* pdgDatabase = TDatabasePDG::Instance();
    TObjArray* NewIons = fRun->GetUserDefIons();
    TIterator* Iter = NewIons->MakeIterator();
    Iter->Reset();
    TObject* obj = 0;
    FairIon* ion = 0;
    while ((obj = Iter->Next())) {
        ion = dynamic_cast<FairIon*>(obj);
        if (ion) {
            // Check if an ion with the calculated pdg code already exists in the
            // TDatabasePDG.
            // If the ion already exist don't create a new one because this fails for
            // Geant4. Instead modify the FairIon to use the already existing ion
            // from the TDatabasePDG.
            // The problem occured for example for Alphas which exist already.
            Int_t ionPdg = GetIonPdg(ion->GetZ(), ion->GetA());
            if (!pdgDatabase->GetParticle(ionPdg)) {
                fMC->DefineIon(
                    ion->GetName(), ion->GetZ(), ion->GetA(), ion->GetQ(), ion->GetExcEnergy(), ion->GetMass());

            } else {
                ion->SetName(pdgDatabase->GetParticle(ionPdg)->GetName());
            }
            // Add Ion to gGeoManager visualization
            if (gGeoManager) {
                gGeoManager->SetPdgName(pdgDatabase->GetParticle(ion->GetName())->PdgCode(), ion->GetName());
            }
            LOG(info) << "Add Ion: " << ion->GetName() << " with PDG "
                      << pdgDatabase->GetParticle(ion->GetName())->PdgCode();
        }
    }
    delete Iter;
    /** Initialize the event generator */
    if (fEvGen) {
        fEvGen->Init();
    }
}
//_____________________________________________________________________________
void FairMCApplication::AddParticles()
{
    TObjArray* NewPart = fRun->GetUserDefParticles();
    TIterator* parIter = NewPart->MakeIterator();
    parIter->Reset();

    // check MC engine is not null (fMC is 0x00 at this line in case of Geant4)
    TVirtualMC* curMC = TVirtualMC::GetMC();
    if (!curMC)
        LOG(fatal) << "No MC engine was defined before AddParticles()";

    TObject* obj = 0;
    FairParticle* particle = 0;
    while ((obj = parIter->Next())) {
        particle = dynamic_cast<FairParticle*>(obj);
        if (particle) {   // (Int_t pdg, const char* name, TMCParticleType type, Double_t mass, Double_t charge,
                          // Double_t lifetime);
            LOG(info) << "Add Particle: " << particle->GetName() << " with PDG " << particle->GetPDG() << "\n"
                      << particle->GetName() << "           // const TString& name \n"
                      << particle->GetMCType() << "           // TMCParticleType mcType \n"
                      << particle->GetMass() << "             // Double_t mass   \n"
                      << particle->GetCharge() << "           // Double_t charge \n"
                      << particle->GetDecayTime() << "        // Double_t lifetime  \n"
                      << particle->GetPType() << "          // const TString& pType, \n"
                      << particle->GetWidth() << "         // Double_t width   \n"
                      << particle->GetSpin() << "         // Int_t iSpin  \n"
                      << particle->GetiParity() << "         // Int_t iParity \n"
                      << particle->GetConjugation() << "      // Int_t iConjugation  \n"
                      << particle->GetIsospin() << "       // Int_t iIsospin   \n"
                      << particle->GetIsospinZ() << "       // Int_t iIsospinZ    \n"
                      << particle->GetgParity() << "       // Int_t gParity    \n"
                      << particle->GetLepton() << "         // Int_t lepton  \n"
                      << particle->GetBaryon() << "         // Int_t baryon   \n"
                      << particle->IsStable() << "         // Bool_t stable   \n";

            curMC->DefineParticle(particle->GetPDG(),           // Int_t pdg
                                  particle->GetName(),          // const TString& name
                                  particle->GetMCType(),        // TMCParticleType mcType
                                  particle->GetMass(),          // Double_t mass
                                  particle->GetCharge(),        // Double_t charge
                                  particle->GetDecayTime(),     // Double_t lifetime
                                  particle->GetPType(),         // const TString& pType,
                                  particle->GetWidth(),         // Double_t width
                                  particle->GetSpin(),          // Int_t iSpin
                                  particle->GetiParity(),       // Int_t iParity
                                  particle->GetConjugation(),   // Int_t iConjugation
                                  particle->GetIsospin(),       // Int_t iIsospin
                                  particle->GetIsospinZ(),      // Int_t iIsospinZ
                                  particle->GetgParity(),       // Int_t gParity
                                  particle->GetLepton(),        // Int_t lepton
                                  particle->GetBaryon(),        // Int_t baryon
                                  particle->IsStable()          // Bool_t stable
            );

            // Add Particle to gGeoManager visualization
            if (gGeoManager)
                gGeoManager->SetPdgName(particle->GetPDG(), particle->GetName());
        }
    }

    delete parIter;
    AddDecayModes();
}

//_____________________________________________________________________________
void FairMCApplication::AddDecayModes()
{
    TString work = getenv("VMCWORKDIR");
    TString work_config = work + "/gconfig/";
    work_config.ReplaceAll("//", "/");

    TString config_dir = getenv("CONFIG_DIR");
    config_dir.ReplaceAll("//", "/");

    Bool_t AbsPath = kFALSE;

    if (!config_dir.EndsWith("/")) {
        config_dir += "/";
    }
    // set Pythia as external decayer

    if (fPythiaDecayer) {
        TString decayConfig;
        if (fPythiaDecayerConfig.IsNull()) {
            decayConfig = "DecayConfig.C";
            fPythiaDecayerConfig = decayConfig;
        } else {
            if (fPythiaDecayerConfig.Contains("/")) {
                AbsPath = kTRUE;
            }
            decayConfig = fPythiaDecayerConfig;
        }

        if (!AbsPath && TString(gSystem->FindFile(config_dir.Data(), decayConfig)) != TString("")) {
            LOG(info) << "---User path for Configuration (DecayConfig.C) is used : " << config_dir.Data();
        } else {
            if (AbsPath) {
                decayConfig = fPythiaDecayerConfig;
            } else {
                decayConfig = work_config + fPythiaDecayerConfig;
            }
        }
        // Add decay modes using an external configuration script
        LOG(info) << "External Decay Modes with script \n " << decayConfig.Data();
        // Load configuration script and execute it
        Int_t pyt = gROOT->LoadMacro(decayConfig.Data());
        if (pyt == 0) {
            gInterpreter->ProcessLine("DecayConfig()");
        }
    }
    // set user defined phase space decay for particles (ions)
    AbsPath = kFALSE;
    if (fUserDecay) {
        TString Userdecay;
        if (fUserDecayConfig.IsNull()) {
            Userdecay = "UserDecay.C";
            fUserDecayConfig = Userdecay;
        } else {
            if (fUserDecayConfig.Contains("/")) {
                AbsPath = kTRUE;
            }
            Userdecay = fUserDecayConfig;
        }

        if (!AbsPath && TString(gSystem->FindFile(config_dir.Data(), Userdecay)) != TString("")) {
            LOG(info) << "---User path for Configuration (UserDecay.C) is used : " << config_dir.Data();
        } else {
            if (AbsPath) {
                Userdecay = fUserDecayConfig;
            } else {
                Userdecay = work_config + fUserDecayConfig;
            }
        }
        LOG(info) << "User Decay Modes with script \n " << Userdecay.Data();
        Int_t dec = gROOT->LoadMacro(Userdecay.Data());
        if (dec == 0) {
            gInterpreter->ProcessLine("UserDecayConfig()");
        }
    }
}

//_____________________________________________________________________________
FairPrimaryGenerator* FairMCApplication::GetGenerator() { return fEvGen; }

//_____________________________________________________________________________
void FairMCApplication::SetGenerator(FairPrimaryGenerator* pGen) { fEvGen = pGen; }

//_____________________________________________________________________________
void FairMCApplication::AddTask(TTask* fTask)
{
    if (!fFairTaskList) {
        fFairTaskList = new FairTask("Task List", 1);
        gROOT->GetListOfBrowsables()->Add(fFairTaskList);
    }
    fFairTaskList->Add(fTask);
    SetParTask();
}

//_____________________________________________________________________________
FairGenericStack* FairMCApplication::GetStack() { return fStack; }

//_____________________________________________________________________________
TTask* FairMCApplication::GetListOfTasks() { return fFairTaskList; }

//_____________________________________________________________________________
void FairMCApplication::SetParTask()
{
    // Only RTDB init when more than Main Task list
    if (FairRun::Instance()->GetNTasks() >= 1) {
        fFairTaskList->SetParTask();
    }
    // The FairRuntimeDb parameter database has been removed: every module's
    // SetParContainers() was empty and no container was ever requested.
}
//_____________________________________________________________________________
void FairMCApplication::InitTasks()
{

    // Only RTDB init when more than Main Task list
    if (FairRun::Instance()->GetNTasks() >= 1) {
        LOG(info) << "Initialize Tasks--------------------------";
        fFairTaskList->InitTask();
    }
}

//_____________________________________________________________________________
TChain* FairMCApplication::GetChain()
{
    if (fRootManager) {
        return fRootManager->GetInChain();
    } else {
        LOG(warn) << "The function is not available in MT mode";
        return 0;
    }
}

//_____________________________________________________________________________
void FairMCApplication::SetRadiationLengthReg(Bool_t RadLen)
{
    fRadLength = RadLen;
    if (fRadLength) {
        fRadLenMan = new FairRadLenManager();
    }
}

//_____________________________________________________________________________
void FairMCApplication::SetRadiationMapReg(Bool_t RadMap)
{
    fRadMap = RadMap;
    if (fRadMap) {
        fRadMapMan = new FairRadMapManager();
    }
}

//_____________________________________________________________________________
void FairMCApplication::AddMeshList(TObjArray* meshList)
{
    if (!fRadGridMan) {
        fRadGridMan = new FairRadGridManager();
    }
    fRadGridMan->AddMeshList(meshList);
}

//_____________________________________________________________________________
Int_t FairMCApplication::GetIonPdg(Int_t z, Int_t a) const
{
    // Acording to
    // http://pdg.lbl.gov/2012/reviews/rpp2012-rev-monte-carlo-numbering.pdf

    return 1000000000 + 10 * 1000 * z + 10 * a;
}

//_____________________________________________________________________________
void FairMCApplication::UndoGeometryModifications()
{
    // Undo all misalignment done in the MisalignGeometry methods of the
    // several FairModuls.
    // In the output (parameter container and separate geometry file)
    // only the ideal geometry is stored.
    // I don't know any better way than to loop over all physical nodes
    // and to set the matrix back to the original one.
    // TODO: Check if it is more easy to write the ideal geometry before
    //       the geometry is misaligned. In this case one does not have
    //       to revert the misalignment.

    TObjArray* physNodes = gGeoManager->GetListOfPhysicalNodes();
    Int_t numPhysNodes = physNodes->GetEntriesFast();

    if (0 == numPhysNodes)
        return;

    // fRootManager->CreateGeometryFile("misaligned_geometry.root");
    LOG(info) << "Undo all misalignment";

    TGeoPhysicalNode* node = nullptr;
    TGeoHMatrix* ng3 = nullptr;
    for (Int_t k = 0; k < numPhysNodes; k++) {
        node = static_cast<TGeoPhysicalNode*>(physNodes->At(k));
        ng3 = node->GetOriginalMatrix();   //"real" global matrix, what survey sees
        node->Align(ng3);
    }

    gGeoManager->ClearPhysicalNodes(kFALSE);
}

ClassImp(FairMCApplication);
