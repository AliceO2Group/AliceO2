/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * @class FairWriteoutBuffer
 *
 * @author Tobias Stockmanns
 * @brief A container class to store digi data during events
 *
 * The FairWriteoutBuffer class provides a container which handles the storage of data between
 * events. The data has to be given with an active time, the absolute time the data is active in the
 * detector and can influence the output of the data.
 * FillNewData is used to put new data into the container.
 * WriteOutData with a given actual time stores the data which has an active time older than the given time
 * in the corresponding TClonesArray of the FairRootManager.
 * At the end of the task WriteOutAllData has to be called to store the data which is still in the buffer.
 * If one adds data via FillNewData into the buffer which is already present in the buffer, Modify is called.
 * The standard behavior of Modify is that the new data is ignored to simulate pile-up. If a different behavior
 * is wanted one has to overwrite Modify in a derived class.
 *
 * The data which should be stored in the buffer has to be derived from FairTimeStamp.
 * It needs an operator< and a method equal if the same detector element is hit.
 *
 * To use this buffer one has to derive his own buffer class from FairWriteoutBuffer and overwrite the pure virtual
 * functions.
 */

#ifndef FairWriteoutBuffer_H_
#define FairWriteoutBuffer_H_

#include "FairLogger.h"        // for FairLogger
#include "FairRootManager.h"   // for FairRootManager
#include "FairTimeStamp.h"     // for FairTimeStamp

#include <Rtypes.h>         // for Bool_t, Int_t, etc
#include <TClonesArray.h>   // for TClonesArray
#include <TObject.h>        // for TObject
#include <TString.h>        // for TString
#include <iostream>         // for cout, ostream
#include <map>              // for multimap
#include <utility>          // for pair
#include <vector>           // for vector

class FairWriteoutBuffer : public TObject
{
  public:
    FairWriteoutBuffer()
        : TObject()
        , fStartTime_map()
        , fDeadTime_map()
        , fBranchName()
        , fClassName()
        , fTreeSave(false)
        , fActivateBuffering(kFALSE)
        , fVerbose(0)
        , fLogger(FairLogger::GetLogger()){};
    FairWriteoutBuffer(TString branchName, TString className, TString folderName, Bool_t persistance);
    virtual ~FairWriteoutBuffer(){};

    virtual void SaveDataToTree(Bool_t val = kTRUE)
    {
        fTreeSave = val;   ///< If SaveDataToTree is set the data is stored at the end of the buffering into the given
                           ///< TClonesArray.
    }
    virtual void ActivateBuffering(Bool_t val = kTRUE)
    {
        fActivateBuffering = val;   ///< fActivateBuffering has to be set to kTRUE to use the buffering. Otherwise the
                                    ///< data is directly stored in the given TClonesArray.
    }

    Bool_t IsBufferingActivated() { return fActivateBuffering; }

    /// Fills a pointer to a data object into the buffer. StartTime gives the time when the data can influence later
    /// data, activeTime gives the time how long the data can influence later data. Both time data has to be given as an
    /// absolute time!
    virtual void FillNewData(FairTimeStamp* data, double startTime, double activeTime);

    virtual Int_t GetNData() { return fDeadTime_map.size(); }
    virtual std::vector<FairTimeStamp*> GetRemoveOldData(double time);
    virtual std::vector<FairTimeStamp*> GetAllData();

    virtual void SetVerbose(Int_t val) { fVerbose = val; }

    virtual void DeleteOldData()
    {
        if (fBranchName.Length() > 0) {
            TClonesArray* myArray = FairRootManager::Instance()->GetTClonesArray(fBranchName);
            myArray->Delete();
        }
    }

    virtual void WriteOutData(double time);
    virtual void WriteOutAllData();

  protected:
    virtual void AddNewDataToTClonesArray(
        FairTimeStamp* data) = 0;   ///< store the data from the FairTimeStamp pointer in a TClonesArray (you have to
                                    ///< cast it to your type of data)
    virtual double FindTimeForData(
        FairTimeStamp* data) = 0;   ///< if the same data object (like a pad or a pixel) is already present in the
                                    ///< buffer, the time of this object has to be returned otherwise -1
    virtual void FillDataMap(FairTimeStamp* data, double activeTime) = 0;   ///< add a new element in the search buffer
    virtual void EraseDataFromDataMap(
        FairTimeStamp* data) = 0;   ///< delete the element from the search buffer (see PndSdsDigiPixelWriteoutBuffer)

    /// Modify defines the behavior of the buffer if data should be stored which is already in the buffer. Parameters
    /// are the old data with the active time, the new data with an active time. Modify returns than a vector with the
    /// new data which should be stored.
    virtual std::vector<std::pair<double, FairTimeStamp*>> Modify(std::pair<double, FairTimeStamp*> oldData,
                                                                  std::pair<double, FairTimeStamp*>)
    {
        std::vector<std::pair<double, FairTimeStamp*>> result;
        result.push_back(oldData);
        return result;
    }

    virtual void WriteOutDataDeadTimeMap(double time);
    virtual void MoveDataFromStartTimeMapToDeadTimeMap(double time);
    virtual void FillDataToDeadTimeMap(FairTimeStamp* data, double activeTime, double startTime);

    virtual void PrintData(FairTimeStamp* data)
    {
        std::cout << data->GetTimeStamp();
    };   ///< Method should be overwritten in derived classes to print the data of an object stored in the buffer
    virtual void PrintDeadTimeMap();
    virtual void PrintStartTimeMap();

    std::multimap<double, std::pair<double, FairTimeStamp*>> fStartTime_map;
    std::multimap<double, FairTimeStamp*> fDeadTime_map;

    TString fBranchName;
    TString fClassName;
    Bool_t fTreeSave;
    Bool_t fActivateBuffering;
    Int_t fVerbose;
    FairLogger* fLogger;   //! /// FairLogger

  private:
    FairWriteoutBuffer(const FairWriteoutBuffer&);
    FairWriteoutBuffer& operator=(const FairWriteoutBuffer&);

    ClassDef(FairWriteoutBuffer, 1);
};

#endif /* FairWriteoutBuffer_H_ */
