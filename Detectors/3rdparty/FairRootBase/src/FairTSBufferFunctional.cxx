/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#include "FairTSBufferFunctional.h"

#include "FairLink.h"          // for FairLink
#include "FairRootManager.h"   // for FairRootManager
#include "FairTimeStamp.h"     // for FairTimeStamp

#include <TBranch.h>        // for TBranch
#include <TClass.h>         // for TClass
#include <TClonesArray.h>   // for TClonesArray
#include <TTree.h>          // for TTree

ClassImp(FairTSBufferFunctional);

FairTSBufferFunctional::FairTSBufferFunctional(TString branchName,
                                               TTree* sourceTree,
                                               BinaryFunctor* stopFunction,
                                               BinaryFunctor* startFunction)
    : TObject()
    , fOutputArray(nullptr)
    , fBufferArray(nullptr)
    , fInputArray(nullptr)
    , fStartFunction(startFunction)
    , fStopFunction(stopFunction)
    , fBranch(nullptr)
    , fBranchIndex(-1)
    , fTerminate(kFALSE)
    , fVerbose(0)
{
    fBranch = sourceTree->GetBranch(branchName.Data());
    if (fBranch == 0) {
        std::cout << "-E- FairTSBufferFunctional::FairTSBufferFunctional Branch " << branchName << " does not exist!"
                  << std::endl;
    }
    FairRootManager* ioman = FairRootManager::Instance();
    fInputArray = static_cast<TClonesArray*>(ioman->GetObject(branchName.Data()));
    fBufferArray = new TClonesArray(fInputArray->GetClass()->GetName());
    fOutputArray = new TClonesArray(fInputArray->GetClass()->GetName());
}

TClonesArray* FairTSBufferFunctional::GetData(Double_t stopParameter)
{
    Double_t actualTime = 0.;
    int posBuffer = 0;

    if (fStopFunction == 0) {   // no function is given ==> read in data in traditional way event by event
        ReadInNextEntry();
        fOutputArray->AbsorbObjects(static_cast<TClonesArray*>(fInputArray), 0, fInputArray->GetEntriesFast() - 1);
        return fOutputArray;
    }
    if (fVerbose > 1) {
        std::cout << "-I- FairTSBufferFunctional::GetData for stopParameter: " << stopParameter << std::endl;
    }

    // if the BufferArray is empty fill it
    if (fBufferArray->GetEntriesFast() == 0) {
        if (fVerbose > 1) {
            std::cout << "-I- FairTSBufferFunctional::GetData fBufferArray is empty: Read in Data" << std::endl;
        }
        ReadInNextFilledEntry();
        AbsorbDataBufferArray();
    }

    // if the BufferArray is still empty you have reached the end of your data set
    if (fVerbose > 1) {
        std::cout << "-I- FairTSBufferFunctional::GetData fBufferArray->GetEntriesFast(): "
                  << fBufferArray->GetEntriesFast() << std::endl;
    }
    FairTimeStamp* dataPoint = static_cast<FairTimeStamp*>(fBufferArray->Last());
    if (dataPoint == 0) {
        if (fVerbose > 0) {
            std::cout << "-I- FairTSBufferFunctional::GetData dataPoint is empty ==> All Data read in" << std::endl;
        }
        return fOutputArray;
    }

    dataPoint = static_cast<FairTimeStamp*>(fBufferArray->First());

    while (!(*fStopFunction)(dataPoint, stopParameter)) {   // check if you have reached end of requested data
        posBuffer++;
        // if you have reached the end of the BufferArray fill it with new data from tree
        if (posBuffer == fBufferArray->GetEntriesFast()) {
            ReadInNextFilledEntry();
            AbsorbDataBufferArray();
        }
        // if you are still at the end of the BufferArray than break (no new data in tree)
        if (posBuffer == fBufferArray->GetEntriesFast()) {
            break;
        }
        dataPoint = static_cast<FairTimeStamp*>(fBufferArray->At(posBuffer));
        if (fVerbose > 1) {
            std::cout << posBuffer << " TimeStampData: " << dataPoint->GetTimeStamp() << std::endl;
        }
    }

    if (fVerbose > 1) {
        std::cout << "-I- FairTSBufferFunctional::GetData Index for Absorb: " << posBuffer
                  << " BufferArray size: " << fBufferArray->GetEntriesFast() << std::endl;
    }
    if (posBuffer < fBufferArray->GetEntriesFast() && posBuffer > 0) {
        if (fVerbose > 1) {
            std::cout << "-I- FairTSBufferFunctional::GetData absorb BufferArray up to posBuffer " << posBuffer
                      << " into fOutputArray" << std::endl;
        }
        fOutputArray->AbsorbObjects(fBufferArray, 0, posBuffer - 1);
        posBuffer = 0;
        return fOutputArray;
    }
    if (fVerbose > 1) {
        std::cout << "Index: " << posBuffer << " BranchIndex: " << fBranchIndex << " NBranch " << fBranch->GetEntries()
                  << std::endl;
    }

    if (posBuffer >= fBufferArray->GetEntriesFast() && posBuffer != 0 && fBranchIndex + 1 >= fBranch->GetEntries()) {
        if (fVerbose > 1) {
            std::cout << "-I- FairTSBufferFunctional::GetData end of data reached. Send the rest to the OutputArray!"
                      << std::endl;
        }
        fOutputArray->AbsorbObjects(fBufferArray, 0, fBufferArray->GetEntries() - 1);
    }

    if (fVerbose > 1) {
        std::cout << "-I- FairTSBufferFunctional::GetData: Read in up to entry: " << fBranchIndex << " with actualTime "
                  << actualTime << " and requested stopParameter " << stopParameter << std::endl;
    }

    return fOutputArray;
}

TClonesArray* FairTSBufferFunctional::GetData(Double_t startParameter, Double_t stopParameter)
{
    if (fStartFunction != 0) {
        fBufferArray->Delete();
        fOutputArray->Delete();
        Int_t startIndex = FindStartIndex(startParameter);
        //    std::cout << "StartParameter : " << startParameter << " StartIndex: " << startIndex << "/" <<
        //    GetBranchIndex() << " size BufferArray " << fBufferArray->GetEntries() <<std::endl;
        if (startIndex > -1) {
            ReadInEntry(fBranchIndex);
            fBufferArray->AbsorbObjects(fInputArray, startIndex, fInputArray->GetEntries() - 1);
        }
    }
    return GetData(stopParameter);
}

Int_t FairTSBufferFunctional::FindStartIndex(Double_t startParameter)
{
    FairTimeStamp* dataPoint;
    Int_t tempIndex = fBranchIndex;
    //  Bool_t runBackwards = kTRUE;
    Int_t previousIndex = -1;
    Int_t previousBranchIndex = -1;

    ReadInEntry(tempIndex);   //< Get Data out of Tree

    while (fInputArray->GetEntries() == 0 && tempIndex > 0) {   //< If the entry of the tree was empty read in previous
                                                                // entries until you find one which is filled
        tempIndex--;
        ReadInEntry(tempIndex);
    }

    if (fInputArray->GetEntries()
        == 0) {   // If the previous entries in the tree are also empty run in the forward direction
        ReadInNextFilledEntry();
        //    runBackwards = kFALSE;
    }

    if (fInputArray->GetEntries() == 0) {   // If there is still no data the branch is empty!
        std::cout << "-I- FairTSBufferFunctional::FindStartIndex: All entries are empty!" << std::endl;
        return -1;
    }
    fBranchIndex = tempIndex;
    // Now we have data or FindStartIndex already returned -1

    dataPoint = static_cast<FairTimeStamp*>(fInputArray->Last());
    // std::cout << "DataPoint: " << *dataPoint << std::endl;
    while (!(*fStartFunction)(dataPoint, startParameter)) {
        // std::cout << "DataPoint Search Entry: " << fBranchIndex << ": " << *dataPoint << std::endl;
        ReadInNextFilledEntry();
        if (fInputArray->GetEntries() != 0) {
            dataPoint = static_cast<FairTimeStamp*>(fInputArray->Last());
        } else {
            return -1;
        }
    }

    // Now you have data where the last element in the array does not fit to your request
    Int_t startPos = fInputArray->GetEntries() - 1;
    while ((*fStartFunction)(dataPoint, startParameter)) {
        // std::cout << "DataPoint Search in Entry: " << fBranchIndex << ": " << *dataPoint << std::endl;
        previousIndex = startPos;
        previousBranchIndex = fBranchIndex;
        startPos--;
        if (startPos == -1) {
            fBranchIndex = ReadInPreviousFilledEntry(fBranchIndex);
            startPos = fInputArray->GetEntries() - 1;
            if (startPos < 0) {
                if (fBranchIndex == 0) {
                    return 0;
                }
                return -1;
            }
        }
        dataPoint = static_cast<FairTimeStamp*>(fInputArray->At(startPos));
    }
    fBranchIndex = previousBranchIndex;
    return previousIndex;
}

void FairTSBufferFunctional::ReadInNextFilledEntry()
{
    fInputArray->Delete();

    if (fVerbose > 1) {
        std::cout << "-I- FairTSBufferFunctional::ReadInNextFilledEntry: Entries in InputArray "
                  << fInputArray->GetEntriesFast() << " Branch Entries: " << fBranch->GetEntries() << std::endl;
    }
    while (fInputArray->GetEntriesFast() == 0 && fBranchIndex + 1 < fBranch->GetEntries()) {
        fBranchIndex++;
        ReadInEntry(fBranchIndex);
    }
}

Int_t FairTSBufferFunctional::ReadInPreviousFilledEntry(Int_t startEntry)
{
    Int_t tempIndex = startEntry;

    fInputArray->Delete();
    while (fInputArray->GetEntriesFast() == 0 && tempIndex > 0) {
        tempIndex--;
        ReadInEntry(tempIndex);
    }
    return tempIndex;
}

void FairTSBufferFunctional::AbsorbDataBufferArray()
{
    if (fInputArray->GetEntriesFast() > 0) {
        if (fVerbose > 1) {
            std::cout << "-I- FairTSBufferFunctional::ReadInNextFilledEntry: Absorb InputArray into Buffer"
                      << std::endl;
        }
        fBufferArray->AbsorbObjects(fInputArray, 0, fInputArray->GetEntries() - 1);
    }
}

void FairTSBufferFunctional::ReadInNextEntry()
{
    if (fBranchIndex + 1 < fBranch->GetEntries()) {
        fBranchIndex++;
        ReadInEntry(fBranchIndex);
    }
}

void FairTSBufferFunctional::ReadInEntry(Int_t number)
{
    fInputArray->Delete();
    if (number < fBranch->GetEntries()) {
        fBranch->GetEntry(number);
        for (int i = 0; i < fInputArray->GetEntriesFast(); i++) {
            (static_cast<FairTimeStamp*>(fInputArray->At(i)))
                ->SetEntryNr(FairLink(0, number, FairRootManager::Instance()->GetBranchId(fBranch->GetName()), i, 1));
        }
        if (fVerbose > 1)
            std::cout << "-I- FairTSBufferFunctional::ReadInEntry BranchIndex: " << number
                      << " Entries: " << fInputArray->GetEntriesFast() << std::endl;
    }
}

Bool_t FairTSBufferFunctional::AllDataProcessed()
{
    if (fTerminate == kTRUE) {
        return kTRUE;
    }
    if (fBranchIndex + 1 >= fBranch->GetEntries()) {
        if (fBufferArray->GetEntriesFast() == 0) {
            if (fOutputArray->GetEntriesFast() == 0) {
                return kTRUE;
            } else {
                return kFALSE;
            }
        } else {
            return kFALSE;
        }
    } else {
        return kFALSE;
    }
}
