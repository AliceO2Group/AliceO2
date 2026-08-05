/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----               FairMCApplication header file                   -----
// -----            Created 06/01/04  by M. Al-Turany                  -----
// -------------------------------------------------------------------------

#ifndef FAIR_MC_APPLICATION_H
#define FAIR_MC_APPLICATION_H

#include "FairRunInfo.h"   // for FairRunInfo

#include <Rtypes.h>                  // for Int_t, Bool_t, Double_t, etc
#include <TLorentzVector.h>          // for TLorentzVector
#include <TString.h>                 // for TString
#include <TVirtualMCApplication.h>   // for TVirtualMCApplication
#include <list>                      // for list
#include <map>                       // for map, multimap, etc

class FairDetector;
class FairEventHeader;
class FairField;
class FairGenericStack;
class FairMCEventHeader;
class FairPrimaryGenerator;
class FairRadGridManager;
class FairRadLenManager;
class FairRadMapManager;
class FairRootManager;
class FairTask;
class FairTrajFilter;
class FairVolume;
class FairRunSim;
class TChain;
class TIterator;
class TObjArray;
class TRefArray;
class TTask;
class TVirtualMC;

enum class FairMCApplicationState
{
    kUnknownState,
    kConstructGeometry,
    kInitGeometry
};

/**
 * The Main Application ( Interface to MonteCarlo application )
 * @author M. Al-Turany, D. Bertini
 * @version 0.1
 * @since 12.01.04
 */

class FairMCApplication : public TVirtualMCApplication
{
  public:
    /** Standard constructor
     *@param name      name
     *@param title     title
     *@param ModList  a TObjArray containing all detectors and modules used in this simulation
     *@param MatName  material file name
     */
    FairMCApplication(const char* name, const char* title, TObjArray* ModList, const char* MatName);
    /** default constructor
     */
    FairMCApplication();
    /** default destructor
     */
    virtual ~FairMCApplication();
    /** Singelton instance
     */
    static FairMCApplication* Instance();
    virtual void AddDecayModes();
    /**  Add user defined particles (optional) */
    virtual void AddParticles();   // MC Application
    /** Add user defined ions (optional) */
    virtual void AddIons();   // MC Application
    /**
     *Add user defined Tasks to be executed after each event (optional)
     * @param fTask: Task that has to be excuted during simulation
     */
    void AddTask(TTask* fTask);
    /** Define actions at the beginning of the event */
    virtual void BeginEvent();   // MC Application
    /** Define actions at the beginning of primary track */
    virtual void BeginPrimary();   // MC Application
    /** Construct user geometry */
    virtual void ConstructGeometry();   // MC Application
    /** Align or misalign geometry before actual run       */
    virtual Bool_t MisalignGeometry();
    /** Define parameters for optical processes (optional) */
    virtual void ConstructOpGeometry();   // MC Application
    /** Define actions at the end of event */
    virtual void FinishEvent();   // MC Application
    /** Define actions at the end of primary track */
    virtual void FinishPrimary();   // MC Application
    /** Define actions at the end of run */
    void FinishRun();
    /** Generate primary particles */
    virtual void GeneratePrimaries();   // MC Application
    /** Return detector by name  */
    FairDetector* GetDetector(const char* DetName);
    /** Return Field used in simulation*/
    FairField* GetField() { return fxField; }
    /**Return primary generator*/
    FairPrimaryGenerator* GetGenerator();
    /**Return list of tasks*/
    TTask* GetListOfTasks();
    FairGenericStack* GetStack();
    TChain* GetChain();
    /** Initialize geometry */
    virtual void InitGeometry();   // MC Application
    /** Initialize MC engine */
    void InitMC(const char* setup, const char* cuts);
    /** Initialize Tasks if any*/
    void InitTasks();
    /**Define actions at the end of each track */
    virtual void PostTrack();   // MC Application
    /** Define actions at the beginning of each track*/
    virtual void PreTrack();   // MC Application

    /** Clone for worker (used in MT mode only) */
    virtual TVirtualMCApplication* CloneForWorker() const;

    /** Init application on worker (used in MT mode only) */
    virtual void InitOnWorker();

    /** Finish run on worker (used in MT mode only) */
    virtual void FinishRunOnWorker();

    /** Run the MC engine
     * @param nofEvents : number of events to simulate
     */
    void RunMC(Int_t nofEvents);
    /**
     * Set the magnetic field for simulation
     * @param field: magnetic field
     */
    void SetField(FairField* field);
    /**
     * Set the event generator  for simulation
     * @param fxGenerator: Event generator(s)
     */
    void SetGenerator(FairPrimaryGenerator* fxGenerator);
    /**
     * Set the parameter containers needed by Tasks(if any)
     */
    void SetParTask();
    /**
     * Switch for using Pythia as external decayer
     * @param decayer: if TRUE pythia will decay particles specifid in the Decay Config macro (see
     * SetPythiaDecayerConfig)
     */
    void SetPythiaDecayer(Bool_t decayer) { fPythiaDecayer = decayer; }
    /**
     * set the decay configuration macro to be used by Pythia
     */
    void SetPythiaDecayerConfig(const TString decayerConf) { fPythiaDecayerConfig = decayerConf; }
    /**
     * Switch for using the radiation length manager
     */
    void SetRadiationLengthReg(Bool_t RadLen);
    /**
     * Switch for using the radiation map manager
     */
    void SetRadiationMapReg(Bool_t RadMap);
    /**
     * Switch for debuging the tracking
     */
    void SetTrackingDebugMode(Bool_t set) { fDebug = set; }
    /**
     * Switch for using 2 or 3 body phase-space decay
     * @param  decay: if TRUE 2/3 body phase space decay will be used for particle specified in the User Decay Config
     * macro (see SetUserDecayConfig)
     */
    void SetUserDecay(Bool_t decay) { fUserDecay = decay; }
    /**
     * set the decay configuration macro to be used by user decay
     */
    void SetUserDecayConfig(const TString decayerConf) { fUserDecayConfig = decayerConf; }
    /** Define action at each step, dispatch the action to the corresponding detectors */
    virtual void Stepping();   // MC Application
    /** Stop the run*/
    virtual void StopRun();
    /** Stop the run*/
    virtual void StopMCRun();
    /**Define maximum radius for tracking (optional) */
    virtual Double_t TrackingRmax() const;   // MC Application
    /** Define maximum z for tracking (optional) */
    virtual Double_t TrackingZmax() const;   // MC Application

    void AddMeshList(TObjArray* meshList);

    /**
     * Set if the current event should be written to the output file.
     * The default value which is set back after each event is to store
     * the event.
     */
    void SetSaveCurrentEvent(Bool_t set) { fSaveCurrentEvent = set; }

    /**
     * Get the current application state.
     */
    FairMCApplicationState GetState() const { return fState; }

  private:
    // methods
    Int_t GetIonPdg(Int_t z, Int_t a) const;

    void UndoGeometryModifications();

  protected:
    // data members
    /**List of active detector */
    TRefArray* fActiveDetectors;
    /**List of FairTask*/
    FairTask* fFairTaskList;   //!
    /**detector list (Passive and Active)*/
    TRefArray* fDetectors;
    /**Map used for dispatcher*/
    TRefArray* fDetMap;
    /**Iterator for Module list*/
    TIterator* fModIter;   //!
    /**Module list in simulation*/
    TObjArray* fModules;
    /**Number of sensetive volumes in simulation session*/
    Int_t fNoSenVolumes;   //!
    /**flag for using Pythia as external decayer */
    Bool_t fPythiaDecayer;
    /** Pythia decay config macro*/
    TString fPythiaDecayerConfig;   //!
    /** Simulation Stack  */
    FairGenericStack* fStack;   //!
    /**Pointer to thr I/O Manager */
    FairRootManager* fRootManager;   //!
    /**List of sensetive volumes in all detectors*/
    TRefArray* fSenVolumes;   //!
    /**Magnetic Field Pointer*/
    FairField* fxField;   //
    /**Primary generator*/
    FairPrimaryGenerator* fEvGen;   //
    /**MC Engine 1= Geant3, 2 = Geant4*/
    Int_t fMcVersion;   // mc Version
    /** Track visualization manager */
    FairTrajFilter* fTrajFilter;   //!
    /**Flag for accepted tracks for visualization*/
    Bool_t fTrajAccepted;   //!
    /**Flag for using user decay*/
    Bool_t fUserDecay;
    /**User decay config macro*/
    TString fUserDecayConfig;   //!
    /** Debug flag*/
    Bool_t fDebug;   //!
    /**dispatcher internal use */
    FairVolume* fDisVol;
    /**dispatcher internal use */
    FairDetector* fDisDet;
    /**dispatcher internal use */
    std::multimap<Int_t, FairVolume*> fVolMap;   //!
    /**dispatcher internal use */
    std::multimap<Int_t, FairVolume*>::iterator fVolIter;   //!
    /** Track position*/
    /**dispatcher internal use RadLeng*/
    std::map<Int_t, Int_t> fModVolMap;   //!
    /**dispatcher internal use RadLen*/
    std::map<Int_t, Int_t>::iterator fModVolIter;   //!
    TLorentzVector fTrkPos;                         //!
    /** Flag for Radiation length register mode  */
    Bool_t fRadLength;   //!

    /**Radiation length Manager*/
    FairRadLenManager* fRadLenMan;   //!
    /** Flag for Radiation map register mode  */
    Bool_t fRadMap;   //!
    /**Radiation Map Manager*/
    FairRadMapManager* fRadMapMan;   //!
    /**Radiation map Grid Manager*/
    FairRadGridManager* fRadGridMan;   //!

    FairEventHeader* fEventHeader;   //!

    FairMCEventHeader* fMCEventHeader;   //!
    /** list of senstive detectors used in the simuation session*/
    std::list<FairDetector*> listActiveDetectors;   //!
    /** list of all detectors used in the simuation session*/
    std::list<FairDetector*> listDetectors;   //!
    /** Pointer to the current MC engine //!
     */
    TVirtualMC* fMC;
    /** Pointer to FairRunSim //! */
    FairRunSim* fRun;

    /** Flag if the current event should be saved */
    Bool_t fSaveCurrentEvent;

    /** Current state */
    FairMCApplicationState fState;   //!

    ClassDef(FairMCApplication, 4);

  private:
    /** Protected copy constructor */
    FairMCApplication(const FairMCApplication&);
    /** Protected assignment operator */
    FairMCApplication& operator=(const FairMCApplication&);

    FairRunInfo fRunInfo;   //!
    Bool_t fGeometryIsInitialized;

    static FairMCApplication* fgMasterInstance;
};

// inline functions

inline FairMCApplication* FairMCApplication::Instance()
{
    return static_cast<FairMCApplication*>(TVirtualMCApplication::Instance());
}

#endif
