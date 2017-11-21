// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// This file is an adaption of FairRoot::FairRunAna commit a6c5cfbe143d3391e (dev branch)
// created 28.9.20017

#ifndef O2_O2RUNANA_H
#define O2_O2RUNANA_H

#include "FairRun.h"                    // for FairRun
#include "FairRootManager.h"            // for FairRootManager
#include "FairRunInfo.h"                // for FairRunInfo
#include "Rtypes.h"                     // for Bool_t, Double_t, UInt_t, etc
#include "TString.h"                    // for TString

#include "SimulationDataFormat/ProcessingEventInfo.h" 

class FairField;
class TF1;
class TFile;

class FairFileSource;
class FairMixedSource;

namespace o2 {
namespace steer {

/// O2 specific run class; steering analysis runs
/// such as digitization, clusterization, reco
/// that typically follow a simulation
/// This class analyses the consumer tasks for their
/// input requests and reads this input from the input
/// file (for asked events)
class O2RunAna : public FairRun
{
  public:
    /// get access to singleton instance
    static O2RunAna* Instance();
    ~O2RunAna() override;

    // -------- interface functions from FairRun -----------
    
    /**initialize the run manager*/
    void        Init() override;
    /**Run from event number NStart to event number NStop */
    void        Run(Int_t NStart=0 ,Int_t NStop=0) override;
    /** Get the magnetic field **/
    FairField*  GetField() override {
      return mField;
    }

    // ------ other functions ------------------------------

    
    /**Run over the whole input file with timpe window delta_t as unit (entry)*/
    void        Run(Double_t delta_t);
    /**Run for the given single entry*/
    void        Run(Long64_t entry);

    /**Run event reconstruction from event number NStart to event number NStop */
    void        RunEventReco(Int_t NStart ,Int_t NStop);
    /**Run over all TSBuffers until the data is processed*/
    void        RunTSBuffers();
    /** the dummy run does not check the evt header or the parameters!! */
    void        DummyRun(Int_t NStart ,Int_t NStop);

    /** This methode is only needed and used with ZeroMQ
      * it read a certain event and call the task exec, but no output is written
      * @param entry : entry number in the tree
      */
    void RunMQ(Long64_t entry);

    /** finish tasks, write output*/
    void        TerminateRun();


    /**Set the input signal file
     *@param name :        signal file name
     *@param identifier :  Unsigned integer which identify the signal file
     */
    void SetSource(FairSource* tempSource) { fRootManager->SetSource(tempSource); }

    // ********************************************************* //
    // THE BELOW FUNCTIONS SHOULD BE MOVED TO FairFileSource
    /**Set the input file by name*/
    void        SetInputFile(TString fname);
    /**Add a file to input chain */
    void        AddFile(TString name);
    /** Add a friend file (input) by name)*/
    void        AddFriend(TString fName);

    // ********************************************************* //
    // THE BELOW FUNCTIONS SHOULD BE MOVED TO FairMixedSource
    void        SetSignalFile(TString name, UInt_t identifier );

    /**Add signal file to input
     *@param name :        signal file name
     *@param identifier :  Unsigned integer which identify the signal file to which this signal should be added
     */
    void        AddSignalFile(TString name, UInt_t identifier );

    /**Set the input background file by name*/
    void        SetBackgroundFile(TString name);

    /**Add input background file by name*/
    void        AddBackgroundFile(TString name);

    /**Set the signal to background ratio in event units
     *@param background :  Number of background Events for one signal
     *@param Signalid :    Signal file Id, used when adding (setting) the signal file
     * here we just forward the call to the FairRootManager
     */
    void BGWindowWidthNo(UInt_t background, UInt_t Signalid);

    /**Set the signal to background rate in time units
     *@param background :  Time of background Events before one signal
     *@param Signalid :    Signal file Id, used when adding (setting) the signal file
     * here we just forward the call to the FairRootManager
     */
     void BGWindowWidthTime(Double_t background, UInt_t Signalid);

     /**
     * This method will simply forward the call to the FairRootManager,
     * if  true all inputs are mixed, i.e: each read event will take one entry from each input and put
     * them in one big event and send it to the next step
    */
    //    void SetMixAllInputs(Bool_t Status);
    // ********************************************************* //
    // THE BELOW FUNCTIONS SHOULD BE MOVED TO FairFileSource and FairMixedSource
    /** Set the min and max limit for event time in ns */
    void SetEventTimeInterval(Double_t min, Double_t max);
    /** Set the mean time for the event in ns */
    void SetEventMeanTime(Double_t mean);
    /** Set the time intervall the beam is interacting and the gap in ns */
    void SetBeamTime(Double_t beamTime, Double_t gapTime);
    // ********************************************************* //

    /** Switch On/Off the storing of FairEventHeader in output file*/
    void SetEventHeaderPersistence(Bool_t flag){
        mStoreEventHeader=flag;
    }

    void        Reinit(UInt_t runId);
    UInt_t      getRunId() {
      return fRunId;
    }

    /** Set the magnetic Field */
    void        SetField (FairField* ffield ) {
      mField=ffield ;
    }
    /** Set external geometry file */
    void        SetGeomFile(const char* GeoFileName);
    /** Return a pointer to the geometry file */
    TFile*      GetGeoFile() {
      return mInputGeoFile;
    }
    /** Initialization of parameter container is set to static, i.e: the run id is
     *  is not checked anymore after initialization
     */

    void        SetContainerStatic(Bool_t tempBool=kTRUE);
    Bool_t      GetContainerStatic() { return mStatic; };

  private:
    O2RunAna();
    O2RunAna(const O2RunAna& M) = delete;
    O2RunAna& operator= (const  O2RunAna&) = delete;

    void Fill();
    
    FairRunInfo mRunInfo;                   //!
    /** This variable became true after Init is called*/
    Bool_t                                  mIsInitialized;
    TFile*                                  mInputGeoFile;
    Bool_t                                  mLoadGeo;
    /** true for static initialisation of parameters */
    Bool_t                                  mStatic;//!
    FairField*                              mField;
    Bool_t                                  mTimeStamps;
    Bool_t                                  mInFileIsOpen;//!
    /** min time for one event (ns) */
    Double_t                                mEventTimeMin;  //!
    /** max time for one Event (ns) */
    Double_t                                mEventTimeMax;  //!
    /** Time of event since th start (ns) */
    Double_t                                mEventTime;     //!
    /** EventMean time used (P(t)=1/fEventMeanTime*Exp(-t/fEventMeanTime) */
    Double_t                                mEventMeanTime; //!
    /** used to generate random numbers for event time; */
    TF1*                                    mTimeProb;      //!

    /** Temporary member to preserve old functionality without setting source in macro */
    FairFileSource*                         mFileSource;  //!
    /** Temporary member to preserve old functionality without setting source in macro */
    FairMixedSource*                        mMixedSource; //!
    /** Flag for Event Header Persistency */
    Bool_t  mStoreEventHeader; //!

    // branches to read from File
    std::vector<std::string> mRequestedBranches;

    // storage for event info passed along to tasks
    o2::ProcessingEventInfo* mEventInfo;
    
    ClassDefOverride(O2RunAna, 0)
};

}
}
 
#endif //O2_O2RUNANA_H
