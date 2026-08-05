/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                      FairTask header file                      -----
// -----          Created 12/01/04  by M. Al-Turany / D. Bertini       -----
// -------------------------------------------------------------------------

/** FairTask
 * @author M. Al-Turany, Denis Bertini
 * @since 12.01.04
 **
 ** Base class for tasks in the cbmroot framework.
 ** Derived classes should implement the Exec method.
 **/

#ifndef FAIRTASK_H
#define FAIRTASK_H

#include "FairRootManager.h"   // for FairRootManager

#include <Rtypes.h>    // for Int_t, FairTask::Class, etc
#include <TString.h>   // for TString
#include <TTask.h>     // for TTask
#include <map>

class FairLogger;

enum InitStatus
{
    kSUCCESS,
    kERROR,
    kFATAL
};

class FairTask : public TTask
{
  public:
    /** Default constructor **/
    FairTask();

    /** Standard constructor
     *@param name        Name of task
     *@param iVerbose    Verbosity level
     **/
    FairTask(const char* name, Int_t iVerbose = 1);

    /** Destructor **/
    virtual ~FairTask();

    /** Initialisation at begin of run. For this task and all of the subtasks.
        Method used internally in FairRoot. **/
    void InitTask();

    /** Reinitialisation. For this task and all of the subtasks.
        Method used internally in FairRoot. **/
    void ReInitTask();

    /** Set parameters. For this task and all of the subtasks.
        Method used internally in FairRoot. **/
    void SetParTask();

    /** Action at end of run. For this task and all of the subtasks.
        Method used internally in FairRoot. **/
    virtual void FinishTask();

    /** Action at end of event. For this task and all of the subtasks. **/
    virtual void FinishEvent();

    /** Set verbosity level. For this task and all of the subtasks. **/
    void SetVerbose(Int_t iVerbose);

    void SetInputPersistance(Bool_t val) { fInputPersistance = val; }

    void CheckInputPersistance(TString branchName)
    {
        FairRootManager* ioman = FairRootManager::Instance();
        fInputPersistance = ioman->CheckBranch(branchName);
    }

    virtual void ExecuteTask(Option_t* option = "0");   // *MENU*

    /** Set persistency of branch with given name true or false
     *  In case is is set to false the branch will not be written to the output.
     **/
    void SetOutputBranchPersistent(TString, Bool_t);

    /** Check if the branch with the given name is persistent.
     *  If the branch is not in the map, the default return value is true.
     **/
    Bool_t IsOutputBranchPersistent(TString);

    void SetStreamProcessing(Bool_t val = kTRUE) { fStreamProcessing = val; }

  protected:
    Int_t fVerbose;            //  Verbosity level
    Int_t fInputPersistance;   ///< Indicates if input branch is persistant
    FairLogger* fLogger;       //!
    Bool_t fStreamProcessing;

    /** Intialisation at begin of run. To be implemented in the derived class.
     *@value  Success   If not kSUCCESS, task will be set inactive.
     **/
    virtual InitStatus Init() { return kSUCCESS; };

    /** Reinitialisation. To be implemented in the derived class.
     *@value  Success   If not kSUCCESS, task will be set inactive.
     **/
    virtual InitStatus ReInit() { return kSUCCESS; };

    /** Intialise parameter containers.
        To be implemented in the derived class.
    **/
    virtual void SetParContainers(){};

    /** Action at end of run. For this task and all of the subtasks.
        To be implemented in the derived class.
    **/
    virtual void Finish(){};

    /** Recursive intialisation of subtasks at begin of run **/
    void InitTasks();

    /** Recursive reinitialisation of subtasks **/
    void ReInitTasks();

    virtual void ExecuteTasks(Option_t* option);

    /** Recursive parameter initialisation for subtasks **/
    void SetParTasks();

    /** Recursive finish of subtasks **/
    void FinishTasks();

    /** Recursive FinishEvent of subtasks **/
    void FinishEvents();

  private:
    std::map<TString, Bool_t> fOutputPersistance;

    FairTask(const FairTask&);
    FairTask& operator=(const FairTask&);

    ClassDef(FairTask, 4);
};

#endif
