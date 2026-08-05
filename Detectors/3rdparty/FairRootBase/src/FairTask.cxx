/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -----                        FairTask source file                   -----
// -----          Created 12/01/04  by M. Al-Turany / D. Bertini       -----

#include "FairTask.h"

#include "FairLogger.h"    // for FairLogger, MESSAGE_ORIGIN
#include "FairMonitor.h"   // for FairMonitor

#include <TCollection.h>   // for TIter
#include <TList.h>         // for TList
#include <TObject.h>       // for TObject
#include <cstdio>          // printf
#include <utility>         // pair

FairTask::FairTask()
    : TTask()
    , fVerbose(0)
    , fInputPersistance(-1)
    , fLogger(FairLogger::GetLogger())
    , fStreamProcessing(kFALSE)
    , fOutputPersistance()
{}

FairTask::FairTask(const char* name, Int_t iVerbose)
    : TTask(name, "FairTask")
    , fVerbose(iVerbose)
    , fInputPersistance(-1)
    , fLogger(FairLogger::GetLogger())
    , fOutputPersistance()
{}

FairTask::~FairTask() {}

void FairTask::InitTask()
{
    FairMonitor::GetMonitor()->SetCurrentTask(this);
    if (!fActive) {
        return;
    }
    InitStatus tStat = Init();
    if (tStat == kFATAL) {
        LOG(fatal) << "Initialization of Task " << fName.Data() << " failed fatally";
    }
    if (tStat == kERROR) {
        fActive = kFALSE;
    }
    FairMonitor::GetMonitor()->SetCurrentTask(0);
    InitTasks();
}

void FairTask::ReInitTask()
{
    if (!fActive) {
        return;
    }
    InitStatus tStat = ReInit();
    if (tStat == kFATAL) {
        LOG(fatal) << "Reinitialization of Task " << fName.Data() << " failed fatally";
    }
    if (tStat == kERROR) {
        fActive = kFALSE;
    }
    ReInitTasks();
}

void FairTask::SetParTask()
{
    if (!fActive) {
        return;
    }
    SetParContainers();
    SetParTasks();
}

void FairTask::FinishEvent()
{
    if (!fActive) {
        return;
    }
    FinishEvents();
}

void FairTask::FinishTask()
{
    if (!fActive) {
        return;
    }
    Finish();
    FinishTasks();
}

void FairTask::SetVerbose(Int_t iVerbose)
{
    fVerbose = iVerbose;
    TIter next(GetListOfTasks());
    FairTask* task;
    while ((task = dynamic_cast<FairTask*>(next()))) {
        task->SetVerbose(iVerbose);
    }
}

void FairTask::InitTasks()
{
    TIter next(GetListOfTasks());
    FairTask* task;
    while ((task = dynamic_cast<FairTask*>(next()))) {
        task->InitTask();
    }
}

void FairTask::ExecuteTask(Option_t* option)
{
    // Execute main task and its subtasks.
    // When calling this function, the Exec function of the corresponding class
    // is invoked, then the list of its subtasks is executed calling recursively
    // all the subtasks, etc.
    //
    // The option parameter may be used to select different execution steps
    // within a task. This parameter is passed also to all the subtasks.

    if (fgBeginTask) {
        Error("ExecuteTask", "Cannot execute task:%s, already running task: %s", GetName(), fgBeginTask->GetName());
        return;
    }
    if (!IsActive())
        return;

    fOption = option;
    fgBeginTask = this;
    fgBreakPoint = 0;

    if (fBreakin)
        return;
    if (gDebug > 1) {
        LOG(info) << "Execute task:" << GetName() << " : " << GetTitle();
    }
    FairMonitor::GetMonitor()->StartMonitoring(this, "EXEC");
    Exec(option);
    FairMonitor::GetMonitor()->StopMonitoring(this, "EXEC");

    fHasExecuted = kTRUE;
    ExecuteTasks(option);

    if (fBreakout)
        return;

    if (!fgBreakPoint) {
        fgBeginTask->CleanTasks();
        fgBeginTask = 0;
    }
}

void FairTask::ExecuteTasks(Option_t* option)
{
    // Execute all the subtasks of a task.

    TIter next(fTasks);
    FairTask* task;
    while ((task = static_cast<FairTask*>(next()))) {
        if (fgBreakPoint)
            return;
        if (!task->IsActive())
            continue;
        if (task->fHasExecuted) {
            task->ExecuteTasks(option);
            continue;
        }
        if (task->fBreakin == 1) {
            printf("Break at entry of task: %s\n", task->GetName());
            fgBreakPoint = this;
            task->fBreakin++;
            return;
        }

        if (gDebug > 1) {
            LOG(info) << "Execute task:" << task->GetName() << " : " << task->GetTitle();
        }
        FairMonitor::GetMonitor()->StartMonitoring(task, "EXEC");
        task->Exec(option);
        FairMonitor::GetMonitor()->StopMonitoring(task, "EXEC");

        task->fHasExecuted = kTRUE;
        task->ExecuteTasks(option);
        if (task->fBreakout == 1) {
            printf("Break at exit of task: %s\n", task->GetName());
            fgBreakPoint = this;
            task->fBreakout++;
            return;
        }
    }
}

void FairTask::ReInitTasks()
{
    TIter next(GetListOfTasks());
    FairTask* task;
    while ((task = dynamic_cast<FairTask*>(next()))) {
        task->ReInitTask();
    }
}

void FairTask::SetParTasks()
{
    TIter next(GetListOfTasks());
    FairTask* task;
    while ((task = dynamic_cast<FairTask*>(next()))) {
        task->SetParTask();
    }
}

void FairTask::FinishTasks()
{
    TIter next(GetListOfTasks());
    FairTask* task;
    while ((task = dynamic_cast<FairTask*>(next()))) {
        task->FinishTask();
    }
}

void FairTask::FinishEvents()
{
    TIter next(GetListOfTasks());
    FairTask* task;
    while ((task = dynamic_cast<FairTask*>(next()))) {
        task->FinishEvent();
    }
}

void FairTask::SetOutputBranchPersistent(TString branchName, Bool_t persistence)
{
    fOutputPersistance.insert(std::pair<TString, Bool_t>(branchName, persistence));
}

Bool_t FairTask::IsOutputBranchPersistent(TString branchName)
{
    std::map<TString, Bool_t>::iterator it = fOutputPersistance.find(branchName);
    if (it != fOutputPersistance.end()) {
        return it->second;
    } else {
        return kTRUE;
    }
}

ClassImp(FairTask);
