/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                   FairRun source file                         -----
// -----            Created 06/01/04  by M. Al-Turany                  -----
// -------------------------------------------------------------------------

#include "FairRun.h"

#include "FairEventHeader.h"    // for FairEventHeader
#include "FairFileHeader.h"     // for FairFileHeader
#include "FairLinkManager.h"    // for FairLinkManager
#include "FairLogger.h"         // for FairLogger, MESSAGE_ORIGIN
#include "FairRootFileSink.h"   // only temporary, should be removed after the move to FairSink is finished
#include "FairRootManager.h"    // for FairRootManager
#include "FairTask.h"           // for FairTask

#include <TFile.h>     // for TFile
#include <TList.h>     // for TList
#include <TObject.h>   // for TObject
#include <cassert>     // for... well, assert

TMCThreadLocal FairRun* FairRun::fRunInstance = 0;

FairRun* FairRun::Instance() { return fRunInstance; }

FairRun::FairRun(Bool_t isMaster)
    : TNamed()
    , fNTasks(0)
    , fTask(new FairTask("FairTaskList"))
    , fRootManager(0)
    , fSink(0)
    , fUserOutputFileName()
    , fRunId(0)
    , fAna(kFALSE)
    , fEvtHeader(nullptr)
    , fFileHeader(new FairFileHeader())
    , fGenerateRunInfo(kFALSE)
    , fIsMaster(isMaster)
    , fMarkFill(kTRUE)
{
    if (fRunInstance) {
        Fatal("FairRun", "Singleton instance already exists.");
        return;
    }
    fRunInstance = this;

    fRootManager = FairRootManager::Instance();

    new FairLinkManager();
}

FairRun::~FairRun()
{
    LOG(debug) << "Enter Destructor of FairRun";
    delete fTask;   // There is another tasklist in MCApplication,
    // but this should be independent
    delete fEvtHeader;
    LOG(debug) << "Leave Destructor of FairRun";
}

void FairRun::AddTask(FairTask* task)
{
    fTask->Add(task);
    StoreTaskNames(task);
}

void FairRun::StoreTaskNames(const FairTask* task)
{
    if (task) {
        fFileHeader->AddTaskClassName(task->ClassName());
        fNTasks++;

        const TList* subtasks = task->GetListOfTasks();
        if (subtasks) {
            for (const auto subtask : *subtasks) {
                StoreTaskNames(static_cast<FairTask*>(subtask));
            }
        }
    }
}

void FairRun::SetTask(FairTask* task)
{
    delete fTask;
    fTask = task;
    fNTasks = 0;
    TList* tasklist = fFileHeader->GetListOfTasks();
    if (tasklist) {
        tasklist->Clear();
        StoreTaskNames(task);
    }
}

void FairRun::CreateGeometryFile(const char* geofile) { fRootManager->CreateGeometryFile(geofile); }

FairTask* FairRun::GetTask(const char* taskName)
{
    TList* taskList = fTask->GetListOfTasks();
    TObject* task = taskList->FindObject(taskName);
    return dynamic_cast<FairTask*>(task);
}

FairEventHeader* FairRun::GetEventHeader()
{
    if (nullptr == fEvtHeader) {
        fEvtHeader = new FairEventHeader();
    }
    return fEvtHeader;
}

void FairRun::SetUseFairLinks(Bool_t val) { fRootManager->SetUseFairLinks(val); }

void FairRun::SetWriteRunInfoFile(Bool_t write)
{
    LOG(warn) << "Function FairRun::SetWriteRunInfoFile(Bool_t) is depcrecated and will vanish in future versions of "
                 "FairRoot.\n";
    LOG(warn) << "Please use FairRun::SetGenerateRunInfo(Bool_t) instead.";

    fGenerateRunInfo = write;
}

Bool_t FairRun::GetWriteRunInfoFile()
{
    LOG(warn)
        << "Function FairRun::GetWriteRunInfoFile() is depcrecated and will vanish in future versions of FairRoot.\n";
    LOG(warn) << "Please use FairRun::IsRunInfoGenerated() instead.";

    return fGenerateRunInfo;
}

void FairRun::SetOutputFile(const char* fname)
{
    LOG(warning) << "FairRun::SetOutputFile() deprecated. Use FairRootFileSink.";
    fSink = new FairRootFileSink(fname);
    if (fRootManager)
        fRootManager->SetSink(fSink);
    fUserOutputFileName = fname;
}

void FairRun::SetOutputFile(TFile* f)
{
    LOG(warning) << "FairRun::SetOutputFile() deprecated. Use FairRootFileSink.";
    fSink = new FairRootFileSink(f);
    if (fRootManager)
        fRootManager->SetSink(fSink);
    if (f)
        fUserOutputFileName = f->GetName();
}

void FairRun::SetOutputFileName(const TString& name)
{
    LOG(warning) << "FairRun::SetOutputFileName() deprecated. Use FairRootFileSink.";
    fSink = new FairRootFileSink(name);
    if (fRootManager)
        fRootManager->SetSink(fSink);
    fUserOutputFileName = name;
}

TFile* FairRun::GetOutputFile()
{
    LOG(warning) << "FairRun::GetOutputFile() deprecated. Use separate file to store additional data.";
    auto sink = GetSink();
    assert(sink->GetSinkType() == kFILESINK);
    auto rootFileSink = static_cast<FairRootFileSink*>(sink);
    return rootFileSink->GetRootFile();
}

void FairRun::SetUserOutputFileName(const TString& name) { fUserOutputFileName = name; }

TString FairRun::GetUserOutputFileName() const { return fUserOutputFileName; }

void FairRun::AlignGeometry() const { fAlignmentHandler.AlignGeometry(); }

void FairRun::AddAlignmentMatrices(const std::map<std::string, TGeoHMatrix>& alignmentMatrices, bool invertMatrices)
{
    fAlignmentHandler.AddAlignmentMatrices(alignmentMatrices, invertMatrices);
}

ClassImp(FairRun);
