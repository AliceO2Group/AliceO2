// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   HmpidEquipments.h
/// \author Antonio Franco - INFN Bari
/// \brief Base Class to describe HMPID Equipment for the decoding of RawData stream
/// \version 1.0
/// \date 24 set 2020

#ifndef COMMON_HMPIDEQUIPMENT_H_
#define COMMON_HMPIDEQUIPMENT_H_

#include <cstdio>
#include <cstdint>
#include <iostream>

#include "HMPIDBase/Geo.h"

namespace o2
{
namespace hmpid
{

const int MAXERRORS = 13;
const int MAXHMPIDERRORS = 5;

const int ERR_NOTKNOWN = 0;
const int ERR_ROWMARKEMPTY = 1;
const int ERR_DUPLICATEPAD = 2;
const int ERR_ROWMARKWRONG = 3;
const int ERR_ROWMARKLOST = 4;
const int ERR_ROWMARKERROR = 5;
const int ERR_LOSTEOEMARK = 6;
const int ERR_DOUBLEEOEMARK = 7;
const int ERR_WRONGSIZEINEOE = 8;
const int ERR_DOUBLEMARKWORD = 9;
const int ERR_WRONGSIZESEGMENTMARK = 10;
const int ERR_LOSTEOSMARK = 11;
const int ERR_HMPID = 12;

// ---- HMPID TRY errors def -------
const int TH_FILENOTEXISTS = 9;
const int TH_OPENFILE = 8;
const int TH_CREATEFILE = 7;
const int TH_READFILE = 6;
const int TH_WRITEFILE = 5;
const int TH_WRONGEQUIPINDEX = 19;
const int TH_WRONGHEADER = 15;
const int TH_WRONGFILELEN = 14;
const int TH_NULLBUFFERPOINTER = 13;
const int TH_BUFFEREMPTY = 12;
const int TH_WRONGBUFFERDIM = 11;
const int TH_BUFFERPOINTERTOEND = 16;

const uint64_t OUTRANGEEVENTNUMBER = 0x1FFFFFFFFFFF;

class HmpidEquipment
{

 private:
  uint32_t mEquipmentId;
  uint32_t mCruId;
  uint32_t mLinkId;

 public:
  uint32_t mPadSamples[Geo::N_COLUMNS][Geo::N_DILOGICS][Geo::N_CHANNELS];
  double mPadSum[Geo::N_COLUMNS][Geo::N_DILOGICS][Geo::N_CHANNELS];
  double mPadSquares[Geo::N_COLUMNS][Geo::N_DILOGICS][Geo::N_CHANNELS];

  int mErrors[MAXERRORS];

  int mWillBeRowMarker;
  int mWillBeSegmentMarker;
  int mWillBeEoE;
  int mWillBePad;
  int mRowSize;
  int mSegment;
  int mColumnCounter;
  int mWordsPerRowCounter;
  int mWordsPerSegCounter;
  int mWordsPerDilogicCounter;

  int mErrorsCounter;
  int mErrorPadsPerEvent;

  uint64_t mEventNumber;
  int mNumberOfEvents;
  float mEventSizeAverage;
  int mEventSize;

  int mSampleNumber;
  float mPadsPerEventAverage;

  float mBusyTimeValue;
  float mBusyTimeAverage;
  int mBusyTimeSamples;
  int mNumberOfEmptyEvents;
  int mNumberOfWrongEvents;
  int mTotalPads;
  int mTotalErrors;

 public:
  HmpidEquipment(int Equipment, int Cru, int Link);
  ~HmpidEquipment();

  int getEquipmentId()
  {
    return (mEquipmentId);
  };
  int getEquipmentId(int cru, int link);
  int getCruId()
  {
    return (mCruId);
  };
  int getLinkId()
  {
    return (mLinkId);
  };

  void init();
  void resetPadMap();
  void resetErrors();
  void setError(int ErrType);
  void setPad(int col, int dil, int cha, uint16_t charge);
};

} // namespace hmpid
} // namespace o2

#endif /* COMMON_HMPIDEQUIPMENT_H_ */
