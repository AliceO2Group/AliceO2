/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRRUNSIM_H
#define FAIRRUNSIM_H

#include "FairIon.h"             // for FairIon
#include "FairMCApplication.h"   // for FairMCApplication
#include "FairParticle.h"        // for FairParticle
#include "FairRun.h"             // for FairRun

#include <Rtypes.h>      // for Bool_t, Double_t, Int_t, etc
#include <TMCtls.h>      // for multi-threading
#include <TObjArray.h>   // for TObjArray
#include <TString.h>     // for TString
#include <functional>

class FairField;
class FairMCEventHeader;
class FairMesh;
class FairModule;
class FairPrimaryGenerator;
class FairGenericVMCConfig;

/**
 * Configure the Simulation session
 * @author M. Al-Turany  D. Bertini
 * @version 0.1
 * @since 12.01.04
 */
class FairRunSim : public FairRun
{
  public:
    /** default ctor*/
    FairRunSim(Bool_t isMaster = kTRUE);
    /** default dtor*/
    virtual ~FairRunSim();
    /** Singelton instance*/
    static FairRunSim* Instance();
    /**
     *       Add a module to the simulation (e.g. PIPE, Magnet, ..etc)
     */
    void AddModule(FairModule* Mod);
    /**
     *       Add a user defined ion to the simulation
     */
    void AddNewIon(FairIon* ion) { fIons->Add(ion); }
    /**
     *       Add a user defined ion to the simulation
     */
    void AddNewParticle(FairParticle* Particle) { fParticles->Add(Particle); }
    /**
     *       this method is used by the FAIRMCApplication
     */
    TObjArray* GetUserDefIons();
    /**
     *       this method is used by the FAIRMCApplication
     */
    TObjArray* GetUserDefParticles();

    /**
     *       Initialize the Simulation
     */
    virtual void Init();
    /**
     *       run the  simulation
     */
    virtual void Run(Int_t NEvents = 0, Int_t NotUsed = 0);
    /**
     *       Set the magnetic that has to be used for simulation field
     */
    void SetField(FairField* field);
    /**
     *       Set the event generator that has to be used for simulation field
     */
    void SetGenerator(FairPrimaryGenerator* Gen);

    /**
     *       Set the experiment dependent event header
     *       for each Monte Carlo Event
     */
    void SetMCEventHeader(FairMCEventHeader* McHeader) { fMCEvHead = McHeader; }

    /** Set the material file name to be used */
    void SetMaterials(const char* MatFileName);

    /**switch On/Off the track visualisation */
    void SetStoreTraj(Bool_t storeTraj = kTRUE) { fStoreTraj = storeTraj; }

    /**Return the switch for the track visualisation */
    Bool_t GetStoreTraj() const { return fStoreTraj; }

    /**switch On/Off the debug mode */
    void SetTrackingDebugMode(Bool_t set)
    {
        if (fApp) {
            fApp->SetTrackingDebugMode(set);
        }
    }

    /**Set geometry builder*/
    void SetGeoModel(char* name);

    /**return the geometry loader used in this session*/
    TString* GetGeoModel() { return fLoaderName; }

    /**Get the field used in simulation*/
    FairField* GetField() { return fField; }

    /**Get the detector specific event header*/
    FairMCEventHeader* GetMCEventHeader();

    /**return the full list of modules used in simulation*/
    TObjArray* GetListOfModules() { return ListOfModules; }

    /**Get the used primary generator*/
    FairPrimaryGenerator* GetPrimaryGenerator() { return fGen; }

    /**switch On/Off external decayer (Pythia) */
    void SetPythiaDecayer(Bool_t decayer) { fPythiaDecayer = decayer; }

    /**switch On external decayer (Pythia). Config macro will be used */
    void SetPythiaDecayer(const TString& Config);

    /**switch On user defined decay, Config  macro will be called  */
    void SetUserDecay(const TString& Config);

    /**switch On/Off user defined decay if true gconfig/UserDecay.C macro will be called  */
    void SetUserDecay(Bool_t decay) { fUserDecay = decay; }

    /**Flag for external decayer*/
    Bool_t IsExtDecayer() { return fPythiaDecayer; }

    /**Flag for User decay*/
    Bool_t IsUserDecay() { return fUserDecay; }

    /**Switch on/off Radiation length register */
    void SetRadLenRegister(Bool_t value) { fRadLength = value; }

    void SetRadMapRegister(Bool_t value) { fRadMap = value; }

    void SetRadGridRegister(Bool_t value) { fRadGrid = value; }

    void AddMesh(FairMesh* Mesh);

    void SetUserConfig(const TString& Config) { fUserConfig = Config; }
    TString GetUserConfig() { return fUserConfig; }

    void SetUserCuts(const TString& Cuts) { fUserCuts = Cuts; }
    TString GetUserCuts() { return fUserCuts; }

    /** Set Beam energy in GeV/c */
    void SetBeamMom(Double_t BeamMom)
    {
        fBeamMom = BeamMom;
        fUseBeamMom = kTRUE;
    }

    /** Get the Beam energy */
    Double_t GetBeamMom() { return fBeamMom; }

    /**Get beam energy flag */
    Bool_t UseBeamMom() { return fUseBeamMom; }

    void SetSimSetup(std::function<void()> f)
    {
        fSimSetup = f;
        fUseSimSetupFunction = true;
    }

    void SetSimulationConfig(FairGenericVMCConfig* tconf) { fSimulationConfig = tconf; }
    FairGenericVMCConfig* GetSimulationConfig() { return fSimulationConfig; }

    void SetIsMT(Bool_t isMT) { fIsMT = isMT; }
    Bool_t IsMT() const { return fIsMT; }

    void SetImportTGeoToVMC(Bool_t v) { fImportTGeoToVMC = v; }
    Bool_t IsImportTGeoToVMC() const { return fImportTGeoToVMC; }

    void StopMCRun() { fApp->StopMCRun(); }

  private:
    FairRunSim(const FairRunSim& M);
    FairRunSim& operator=(const FairRunSim&) { return *this; }
    void SetMCConfig();
    void CheckFlukaExec();

  protected:
    Int_t count;                                    //!                               /** Internal counter*/
    FairMCApplication* fApp;                        //!                              /** Main VMC application */
    Double_t fBeamMom;                              //!                           /** Beam Energy in GeV/c  */
    Bool_t fUseBeamMom;                             //!                        /** flag for use Beam Energy  */
    FairPrimaryGenerator* fGen;                     //!                               /** Primary Event Generator */
    FairMCEventHeader* fMCEvHead;                   //!                          /** MC Event Header */
    static TMCThreadLocal FairRunSim* fginstance;   //!              /** Singleton Instance */
    FairField* fField;                              /** Magnetic Field */
    const char* fMapName;                           //!                           /** Input file name map*/
    TObjArray* fIons;                               //!                              /** Array of user defined ions */
    TObjArray* fParticles;                          //!                         /** Array of user defined particles*/
    TObjArray* ListOfModules;                       //!                       /** Array of used modules */
    TString MatFname;                               //!                           /** Material file name */
    Bool_t fStoreTraj;                              //!                       /** Trajectory store flags */
    TString* fLoaderName;                           //!                       /** Geometry Model (TGeo or G3)*/
    Bool_t fPythiaDecayer;                          //!                    /** flag for using Pythia decayer*/
    TString fPythiaDecayerConfig;                   //!               /** Macro for Pythia decay configuration*/
    Bool_t fUserDecay;                              /** flag for setting user decay */
    TString fUserDecayConfig;                       //!                   /** Macro for decay configuration*/
    Bool_t fRadLength;                              //!                       /** flag for registring radiation length*/
    Bool_t fRadMap;                                 //!                            /** flag for RadiationMapManager
    Bool_t fRadGrid;                                //!
    TObjArray* fMeshList;                           //!                          /** radiation grid scoring
    TString fUserConfig;                            //!                        /** Macro for geant configuration*/
    TString fUserCuts;                              //!                          /** Macro for geant cuts*/
    Bool_t fIsMT;                                   //!                              /** MT mode option (Geant4 only)*/
    Bool_t fImportTGeoToVMC;                        //!                   /** Allow importing TGeometry to VMC */
    std::function<void()> fSimSetup;   //!                          /** A user provided function to do sim setup /
                                       //!                          instead of using macros **/
    bool fUseSimSetupFunction = false;
    FairGenericVMCConfig* fSimulationConfig;   //!                 /** Simulation configuration */

    ClassDef(FairRunSim, 2);
};

#endif   // FAIRRUNSIM_H
