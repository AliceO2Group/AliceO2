/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FairTSBufferFunctionalFunctional_H_
#define FairTSBufferFunctionalFunctional_H_

#include "FairTimeStamp.h"   // for FairTimeStamp

#include <Rtypes.h>     // for Int_t, Bool_t, Double_t, etc
#include <TObject.h>    // for TObject
#include <TString.h>    // for TString
#include <functional>   // for binary_function
#include <iostream>     // for operator<<, basic_ostream, etc

class TBranch;
class TClonesArray;
class TTree;

/**
 * \class BinaryFunctor
 * \brief Base class for all functors which are used in the FairTSBufferFunctional
 * \see FairTSBufferFunctional
 *
 * The class is a base class to control which data is extracted by the FairTSBufferFunctional class for processing.
 * The important method to overwrite is Call. It gets the actual data which is read in from the tree and a parameter.
 * If the actual data is not anymore part of the data you want to have Call returns true to stop the reading of data.
 * Otherwise it should return false.
 * The method TimeOut is used to break the processing if for example always the same data is requested.
 */

class BinaryFunctor
{
  public:
    virtual bool operator()(FairTimeStamp* a, double b) { return Call(a, b); };
    virtual bool Call(FairTimeStamp* a, double b) = 0;
    virtual bool TimeOut() { return false; }
    virtual void ResetTimeOut(){};

    virtual ~BinaryFunctor(){};
};

/**
 * \class StopTime
 * Gives you all the data which is older than the given parameter StopTime.
 * It does not return the data requested before.
 */

class StopTime : public BinaryFunctor
{
  public:
    StopTime()
        : fRequestTime(-1)
        , fOldTime(-1)
        , fSameTimeRequestCounter(0){};

    /**
     * \parameter b: StopTime: All data older than StopTime is returned
     */
    bool Call(FairTimeStamp* a, double b)
    {
        fRequestTime = b;
        // std::cout << "StopTime: " << a->GetTimeStamp() << " > " <<  b << std::endl;
        return a->GetTimeStamp() > b;
    };

    bool TimeOut()
    {
        if (fRequestTime != fOldTime) {
            fOldTime = fRequestTime;
            fSameTimeRequestCounter = 0;
            // std::cout << "RequestedTime: " << fRequestTime << std::endl;
            return false;
        } else if (fRequestTime == fOldTime) {
            std::cout << "-I- FairTSBufferFunctional StopTime has requested the same data as before: " << fRequestTime
                      << std::endl;
            fSameTimeRequestCounter++;
        } else {
            std::cout << "-E- FairTSBufferFunctional StopTime Functor has requested time " << fRequestTime
                      << " smaller than old time " << fOldTime << std::endl;
            return true;
        }
        if (fSameTimeRequestCounter > 9) {
            return true;
        } else {
            return false;
        }
    }

    void ResetTimeOut() { fSameTimeRequestCounter = 0; }

  private:
    double fRequestTime;
    double fOldTime;
    int fSameTimeRequestCounter;
};

/**
 * \class TimeGap
 * Returns you all the data between two time gaps of a given length.
 */

class TimeGap : public BinaryFunctor
{
  public:
    TimeGap()
        : fOldTime(-1.){};

    /**
     * \parameter b : TimeGap: All data between two time gaps which are larger than TimeGap are returned
     */
    bool Call(FairTimeStamp* a, double b)
    {
        double aTime = a->GetTimeStamp();

        if (fOldTime < 0) {
            fOldTime = aTime;
            return false;
        }
        if (aTime - fOldTime > b) {
            fOldTime = aTime;
            return true;
        } else {
            fOldTime = aTime;
            return false;
        }
    };

  private:
    double fOldTime;
};

/**
 * \class FairTSBufferFunctional
 * \brief A class to access time ordered data in a root branch
 *
 * In the constructor of the class one has to give the branch name of the data, the tree the data is stored in
 * and a BinaryFunctor which contains the method how the data should be extracted. Several example functors already
 * exists. To extract the data one has to call GetData with a parameter which fits to the selected functor. GetData
 * returns a TClonesArray which contains the data.
 *
 *
 * Be careful! The buffer runs through the time ordered data in one time direction only. This means that you cannot
 * request data which is older than the data you have requested before.
 *
 * Addition: This is not true anymore. GetData(Double_t, Double_t) is able to get also data which is older but this only
 * works if you request a fixed time via StopTime functor. For other functors the behavior is unpredictable.
 *
 *  Created on: Feb 18, 201
 *      Author: stockman
 */

class FairTSBufferFunctional : public TObject
{
  public:
    FairTSBufferFunctional(TString branchName,
                           TTree* sourceTree,
                           BinaryFunctor* stopFunction,
                           BinaryFunctor* startFunction = 0);

    virtual ~FairTSBufferFunctional(){};
    TClonesArray* GetData(Double_t stopParameter);
    TClonesArray* GetData(Double_t startParameter, Double_t stopParameter);
    Int_t GetBranchIndex() { return fBranchIndex; }

    void SetBranchIndex(const Int_t val) { fBranchIndex = val; }
    void SetStartFunction(BinaryFunctor* function) { fStartFunction = function; }
    void SetStopFunction(BinaryFunctor* function) { fStopFunction = function; }
    Bool_t AllDataProcessed();
    void Terminate() { fTerminate = kTRUE; }

    Bool_t TimeOut()
    {
        Bool_t stopTimeOut = fStopFunction->TimeOut();
        Bool_t startTimeOut = kTRUE;
        if (fStartFunction != 0) {
            startTimeOut = fStartFunction->TimeOut();
            //        if (startTimeOut == kTRUE && stopTimeOut == kFALSE){
            //          fStartFunction->ResetTimeOut();
            //        }
            //        else if (startTimeOut == kFALSE && stopTimeOut == kTRUE){
            //          fStopFunction->ResetTimeOut();
            //        }
        }
        return (stopTimeOut && startTimeOut);
    }

    Int_t FindStartIndex(Double_t startParameter);

  private:
    void ReadInNextFilledEntry();
    Int_t ReadInPreviousFilledEntry(Int_t startEntry);
    void ReadInNextEntry();   //** used only if no function is given and input data is directly passed through to the
                              // OutputArray
    void ReadInEntry(Int_t number);
    void AbsorbDataBufferArray();   //< Absorbs the complete data from fInputArray to fBufferArray

    TClonesArray* fOutputArray;
    TClonesArray* fBufferArray;
    TClonesArray* fInputArray;

    BinaryFunctor* fStartFunction;
    BinaryFunctor* fStopFunction;

    TBranch* fBranch;
    Int_t fBranchIndex;

    Bool_t fTerminate;

    Int_t fVerbose;

    FairTSBufferFunctional(const FairTSBufferFunctional&);
    FairTSBufferFunctional& operator=(const FairTSBufferFunctional&);

    ClassDef(FairTSBufferFunctional, 0);
};

#endif
