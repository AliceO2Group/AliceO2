/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#include "FairRunInfo.h"

#include "FairLogger.h"   // for FairLogger

#include <TFile.h>       // for TFile, gFile
#include <TH1.h>         // for TH1F
#include <TIterator.h>   // for TIterator
#include <TList.h>       // for TList
#include <TString.h>     // for TString
#include <TSystem.h>     // for ProcInfo_t, TSystem, etc
#include <algorithm>     // for sort

ClassImp(FairRunInfo);

FairRunInfo::FairRunInfo()
    : TObject()
    , fTimeStamp()
    , fCpuInfo()
    , fMemInfo()
    , fProcInfo()
    , fTimeDiff()
    , fTime()
    , fResidentMemory()
    , fVirtualMemory()
{}

FairRunInfo::~FairRunInfo() {}

void FairRunInfo::StoreInfo()
{
    // Extract the Information about the used memory from the system.
    // Stores this info together with the timedifference between now
    // and the last call in container arrays for later usage.
    //  gSystem->GetCpuInfo(&cpuInfo, 10);
    //  gSystem->GetMemInfo(&memInfo);

    GetInfo();

    CalculateTimeDifference();

    PrintInfo();
}

void FairRunInfo::GetInfo()
{
    // Set the TimeStamp to the actual time and store it
    fTimeStamp.Set();
    //  fTime.push_back(fTimeStamp.GetSec());
    fTime.push_back(fTimeStamp.AsDouble());

    gSystem->GetProcInfo(&fProcInfo);
    fResidentMemory.push_back(fProcInfo.fMemResident / 1024);
    fVirtualMemory.push_back(fProcInfo.fMemVirtual / 1024);
}

void FairRunInfo::CalculateTimeDifference()
{
    // Calculates the time difference between now and the last call

    Int_t lastElement = fTime.size() - 1;
    fTimeDiff.push_back(fTime.at(lastElement) - fTime.at(lastElement - 1));
}

void FairRunInfo::PrintInfo()
{

    LOG(debug) << "Time to execute 1 event: " << fTimeDiff.back() << "s";
    LOG(debug) << "Used resident memory: " << fResidentMemory.back() << " MB";
    LOG(debug) << "Used virtual memory: " << fVirtualMemory.back() << " MB";
}

void FairRunInfo::WriteInfo()
{
    TList* histoList = new TList();
    CreateAndFillHistograms(histoList);
    //  CreateAndFillHistograms(histoList);
    WriteHistosToFile(histoList);
}

void FairRunInfo::CreateAndFillHistograms(TList* histoList)
{
    Int_t entries = fTime.size();
    Double_t timePeriod = fTime.back() - fTime.front();
    Int_t timeBins = static_cast<Int_t>(timePeriod * 10);
    if (0 == timeBins)
        timeBins = 1;

    TH1F* ResidentMemoryVsEvent = new TH1F(
        "ResidentMemoryVsEvent", "Resident Memory as function of Eventnumber;Event;Memory [MB]", entries, 0, entries);
    TH1F* VirtualMemoryVsEvent = new TH1F(
        "VirtualMemoryVsEvent", "Virtual Memory as function of Eventnumber;Event;Memory [MB]", entries, 0, entries);
    TH1F* ResidentMemoryVsTime = new TH1F(
        "ResidentMemoryVsTime", "Resident memory as function of Runtime;Time [s];Memory [MB]", timeBins, 0, timeBins);
    TH1F* VirtualMemoryVsTime = new TH1F(
        "VirtualMemoryVsTime", "Virtual memory as function of Runtime;Time [s];Memory [MB]", timeBins, 0, timeBins);
    TH1F* EventtimeVsEvent = new TH1F(
        "EventtimeVsEvent", "Runtime per Event as function of Event number;Event;Time [s]", entries - 1, 1, entries);

    std::vector<Double_t> timeDiffSorted(fTimeDiff);
    std::vector<Double_t>::iterator it;
    std::sort(timeDiffSorted.begin(), timeDiffSorted.end());

    Double_t minTime = timeDiffSorted.front();
    Double_t maxTime = timeDiffSorted.back();
    timePeriod = maxTime - minTime;

    TH1F* TimePerEvent = new TH1F("TimePerEvent",
                                  "Runtime;Time [s];Events",
                                  static_cast<Int_t>((timePeriod + 20) * 10),
                                  static_cast<Int_t>(minTime - 10),
                                  static_cast<Int_t>(maxTime + 10));

    Int_t counter = 0;
    std::vector<Long_t>::iterator lit;
    Double_t timeOffset = fTime.front() + 1.;
    for (lit = fResidentMemory.begin(); lit < fResidentMemory.end(); ++lit) {
        ResidentMemoryVsEvent->Fill(counter, *lit);
        ResidentMemoryVsTime->Fill(fTime.at(counter) - timeOffset, *lit);
        counter++;
    }
    histoList->AddLast(ResidentMemoryVsEvent);
    histoList->AddLast(ResidentMemoryVsTime);

    counter = 0;
    for (lit = fVirtualMemory.begin(); lit < fVirtualMemory.end(); ++lit) {
        VirtualMemoryVsEvent->Fill(counter, *lit);
        VirtualMemoryVsTime->Fill(fTime.at(counter) - timeOffset, *lit);
        counter++;
    }
    histoList->AddLast(VirtualMemoryVsEvent);
    histoList->AddLast(VirtualMemoryVsTime);

    counter = 1;
    for (it = fTimeDiff.begin(); it < fTimeDiff.end(); ++it) {
        TimePerEvent->Fill(*it);
        EventtimeVsEvent->Fill(counter, *it);
        counter++;
    }
    histoList->AddLast(TimePerEvent);
    histoList->AddLast(EventtimeVsEvent);
}

void FairRunInfo::WriteHistosToFile(TList* histoList)
{
    // If the file size is larger then approx. 2GB then the histos
    // can't be read any longer. Because of this problem the histos
    // are written to a separate file instead

    TFile* oldfile = gFile;

    TString directory = gFile->GetName();
    LOG(debug) << "Name: " << gFile->GetName();
    Ssiz_t posLastSlash = directory.Last('/');
    directory.Remove(posLastSlash + 1, directory.Length() - posLastSlash - 1);
    TString filename = "";
    if (directory.EndsWith("/")) {
        filename += directory;
    }

    directory = gFile->GetName();
    LOG(debug) << "Name: " << directory.Data();
    posLastSlash = directory.Last('/');
    directory.Remove(0, posLastSlash + 1);
    directory.ReplaceAll(".root", "");
    LOG(debug) << "Name: " << directory.Data();

    //  Int_t pid = gSystem->GetPid();
    filename += "FairRunInfo_";
    filename += directory;
    filename += ".root";
    LOG(debug) << "Filename: " << filename.Data();

    TFile* f1 = TFile::Open(filename, "recreate");
    f1->cd();

    TIterator* listIter = histoList->MakeIterator();
    listIter->Reset();
    TObject* obj = nullptr;
    while ((obj = listIter->Next())) {
        obj->Write();
    }

    delete listIter;
    histoList->Delete();
    delete histoList;

    f1->Close();
    f1->Delete();
    gFile = oldfile;
}

void FairRunInfo::Reset() { GetInfo(); }
