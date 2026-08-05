/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -----                   FairRunSim source file                      -----
// -----            Created 06/01/04  by M. Al-Turany                  -----

#include "FairRunSim.h"

#include "FairField.h"              // for FairField
#include "FairFileHeader.h"         // for FairFileHeader
#include "FairGenericVMCConfig.h"   // for FairGenericVMCConfig
#include "FairGeoInterface.h"       // for FairGeoInterface
#include "FairGeoLoader.h"          // for FairGeoLoader
#include "FairLogger.h"             // for FairLogger, MESSAGE_ORIGIN
#include "FairMCEventHeader.h"      // for FairMCEventHeader
#include "FairMesh.h"               // for FairMesh
#include "FairModule.h"             // for FairModule
#include "FairPrimaryGenerator.h"   // for FairPrimaryGenerator
#include "FairRootManager.h"        // for FairRootManager
#include "FairRunIdGenerator.h"     // for FairRunIdGenerator
#include "FairTask.h"               // for FairTask
#include "FairTrajFilter.h"         // for FairTrajFilter

#include <TCollection.h>   // for TIter
#include <TGeoManager.h>   // for gGeoManager
#include <TList.h>         // for TList
#include <TObjString.h>    // for TObjString
#include <TObject.h>       // for TObject
#include <TRandom.h>       // for gRandom
#include <TSystem.h>       // for TSystem, gSystem
#include <iostream>        // for cout, endl, ostream
#include <stdlib.h>        // for getenv, nullptr
#include <string.h>        // for strcmp, strncmp

using std::cout;
using std::endl;

ClassImp(FairRunSim);

FairRunSim::FairRunSim(Bool_t isMaster)
    : FairRun(isMaster)
    , count(0)
    , fApp(nullptr)
    , fBeamMom(0)
    , fUseBeamMom(kFALSE)
    , fGen(nullptr)
    , fMCEvHead(nullptr)
    , fField(nullptr)
    , fMapName("")
    , fIons(new TObjArray())
    , fParticles(new TObjArray())
    , ListOfModules(new TObjArray())
    , MatFname("")
    , fStoreTraj(kFALSE)
    , fLoaderName(new TString("TGeo"))
    , fPythiaDecayer(kFALSE)
    , fPythiaDecayerConfig("")
    , fUserDecay(kFALSE)
    , fUserDecayConfig("")
    , fRadLength(kFALSE)
    , fRadMap(kFALSE)
    , fRadGrid(kFALSE)
    , fMeshList(new TObjArray())
    , fUserConfig("")
    , fUserCuts("SetCuts.C")
    , fIsMT(kFALSE)
    , fImportTGeoToVMC(kTRUE)
    , fSimulationConfig(nullptr)

{
    if (fginstance) {
        Fatal("FairRun", "Singleton instance already exists.");
        return;
    }
    fginstance = this;
    fRunId = 0;
    fAna = kFALSE;
}

FairRunSim::~FairRunSim()
{
    LOG(debug) << "Enter Destructor of FairRunSim ";

    // delete fApp;

    /** List of Modules is filled via AddModule from the macro, but it
   is the responsibility of FairRunSim to call the destructors of
   the modules-
  */
    LOG(debug) << "Start deleting all registered modules";
    ListOfModules->Delete();
    delete ListOfModules;
    LOG(debug) << "Finish deleting all registered modules";

    fIons->Delete();
    delete fIons;
    fParticles->Delete();
    delete fParticles;

    delete fApp;
    // delete fField;
    // Not owner of the field
    delete fGen;
    delete fMCEvHead;
}

FairRunSim* FairRunSim::Instance() { return fginstance; }

void FairRunSim::AddModule(FairModule* Mod)
{
    ListOfModules->Add(Mod);
    Mod->SetModId(count++);
}

void FairRunSim::AddMesh(FairMesh* Mesh)
{
    Mesh->print();
    Mesh->calculate();
    fMeshList->Add(Mesh);
}

TObjArray* FairRunSim::GetUserDefIons()
{
    /** return the array of user defined ions*/
    return fIons;
}

TObjArray* FairRunSim::GetUserDefParticles()
{
    /** return the array of user defined particles*/
    return fParticles;
}

void FairRunSim::Init()
{
    /**Initialize the simulation session*/
    fRootManager->InitSink();

    CheckFlukaExec();

    //  fOutFile=fRootManager->OpenOutFile(fOutname);
    LOG(info) << "==============  FairRunSim: Initialising simulation run ==============";

    FairGeoLoader* loader = new FairGeoLoader(fLoaderName->Data(), "Geo Loader");
    FairGeoInterface* GeoInterFace = loader->getGeoInterface();
    GeoInterFace->SetNoOfSets(ListOfModules->GetEntries());
    GeoInterFace->setMediaFile(MatFname.Data());
    GeoInterFace->readMedia();

    //  gSystem->cd(flout.Data());

    fApp = new FairMCApplication("Fair", "The Fair VMC App", ListOfModules, MatFname);
    fApp->SetGenerator(fGen);

    // Add a Generated run ID to the FairRunTimeDb
    if (fRunId == 0) {
        FairRunIdGenerator genid;
        fRunId = genid.generateId();
    }
    fFileHeader->SetRunId(fRunId);
    /** Add Tasks to simulation if any*/
    fApp->AddTask(fTask);
    // The FairBaseParSet / FairGeoParSet runtime-database containers were filled
    // here but never written out nor read back; the database has been removed.

    // Set global Parameter Info

    if (fPythiaDecayer) {
        fApp->SetPythiaDecayer(fPythiaDecayer);
        if (fPythiaDecayerConfig) {
            fApp->SetPythiaDecayerConfig(fPythiaDecayerConfig);
        }
    }
    if (fUserDecay) {
        fApp->SetUserDecay(fUserDecay);
        if (fUserDecayConfig) {
            fApp->SetUserDecayConfig(fUserDecayConfig);
        }
    }
    // on/off visualisation
    if (fStoreTraj) {
        LOG(info) << "Create visualisation manager ";
        new FairTrajFilter();
    }
    if (fRadLength) {
        fApp->SetRadiationLengthReg(fRadLength);
    }
    if (fRadMap) {
        fApp->SetRadiationMapReg(fRadMap);
    }
    if (fRadGrid) {
        fApp->AddMeshList(fMeshList);
    }

    if (fField) {
        fField->Init();
    }
    fApp->SetField(fField);

    /**Set the configuration for MC engine*/
    SetMCConfig();
    fRootManager->WriteFileHeader(fFileHeader);
}


void FairRunSim::CheckFlukaExec()
{
    /** Private method for setting FLUKA simulation*/
    TString work = getenv("VMCWORKDIR");
    TString work_config = work + "/gconfig/";
    work_config.ReplaceAll("//", "/");

    TString config_dir = getenv("CONFIG_DIR");
    config_dir.ReplaceAll("//", "/");
    if (!config_dir.EndsWith("/")) {
        config_dir += "/";
    }

    TString flout;
    if (strcmp(GetName(), "TFluka") == 0) {
        TString flexec = "run_fluka.sh";
        if (TString(gSystem->FindFile(config_dir.Data(), flexec)) != TString("")) {
            LOG(info) << "---User path for Configuration is used: " << config_dir.Data();
        } else {
            flexec = work_config + "run_fluka.sh";
        }
        gSystem->Exec(flexec.Data());
        // More generic for different macro
        // dir layout <D.B>
        TString macrodir = getenv("PWD");
        // cout << macrodir << endl;
        flout = macrodir + "/fluka_out";
        flout.ReplaceAll("//", "/");
        gSystem->cd(flout.Data());
    }
}

void FairRunSim::SetMCConfig()
{
    // Either setup the simulation with the provided user hook
    if (fUseSimSetupFunction) {
        fSimSetup();
    } else {
        if (fSimulationConfig == nullptr)   // RKRKRK COMMENT
            fSimulationConfig = new FairGenericVMCConfig();
        fSimulationConfig->Setup(GetName());
    }

    fApp->InitMC("foo", "bar");
}

void FairRunSim::Run(Int_t NEvents, Int_t) { fApp->RunMC(NEvents); }

void FairRunSim::SetField(FairField* field) { fField = field; }

void FairRunSim::SetGenerator(FairPrimaryGenerator* Gen) { fGen = Gen; }

void FairRunSim::SetMaterials(const char* MatFileName)
{
    TString Mat = "";
    TString work = getenv("GEOMPATH");
    work.ReplaceAll("//", "/");
    if (work.IsNull()) {
        work = getenv("VMCWORKDIR");
        Mat = work + "/geometry/";
        Mat.ReplaceAll("//", "/");
    } else {
        Mat = work;
        if (!Mat.EndsWith("/")) {
            Mat += "/";
        }
    }
    MatFname = Mat + MatFileName;
    LOG(info) << "Media file used: " << MatFname.Data();
}

void FairRunSim::SetGeoModel(char* name)
{
    if (strncmp(fName, "TGeant3", 7) == 0) {
        delete fLoaderName;
        fLoaderName = new TString(name);
        LOG(info) << "FairRun::SetGeoModel(): G3 native geometry model used ";
    } else {
        LOG(info) << "FairRun::SetGeoModel(): Geant3 MC engine only !";
        LOG(info) << "FairRun::SetGeoModel(): Method skipped ... ";
    }
}

void FairRunSim::SetPythiaDecayer(const TString& Config)
{
    /**switch On external decayer (Pythia). Config macro will be used */
    fPythiaDecayerConfig = Config;
    fPythiaDecayer = kTRUE;
}

void FairRunSim::SetUserDecay(const TString& Config)
{
    /**switch On user defined decay, Config  macro will be called  */
    fUserDecayConfig = Config;
    fUserDecay = kTRUE;
}

FairMCEventHeader* FairRunSim::GetMCEventHeader()
{
    if (nullptr == fMCEvHead) {
        fMCEvHead = new FairMCEventHeader();
    }
    return fMCEvHead;
}

TMCThreadLocal FairRunSim* FairRunSim::fginstance = 0;
