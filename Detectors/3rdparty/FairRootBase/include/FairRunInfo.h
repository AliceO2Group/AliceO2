/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRRUNINFO_H
#define FAIRRUNINFO_H

#include <Rtypes.h>       // for Double_t, Long_t, etc
#include <TObject.h>      // for TObject
#include <TSystem.h>      // for CpuInfo_t, MemInfo_t, etc
#include <TTimeStamp.h>   // for TTimeStamp
#include <vector>         // for vector

class TList;

class FairRunInfo : public TObject
{
  public:
    FairRunInfo();
    ~FairRunInfo();

    void StoreInfo();
    void WriteInfo();
    void Reset();

  private:
    TTimeStamp fTimeStamp;                 //!
    CpuInfo_t fCpuInfo;                    //!
    MemInfo_t fMemInfo;                    //!
    ProcInfo_t fProcInfo;                  //!
    std::vector<Double_t> fTimeDiff;       //!
    std::vector<Double_t> fTime;           //!
    std::vector<Long_t> fResidentMemory;   //!
    std::vector<Long_t> fVirtualMemory;    //!

    void CalculateTimeDifference();
    void PrintInfo();
    void GetInfo();
    void CreateAndFillHistograms(TList* histoList);
    void WriteHistosToFile(TList* histoList);

    FairRunInfo(const FairRunInfo&);
    FairRunInfo& operator=(const FairRunInfo&);

    ClassDef(FairRunInfo, 2);
};

#endif
