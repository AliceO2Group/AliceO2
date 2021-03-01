// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   HmpidCoder.h
/// \author Antonio Franco - INFN Bari
/// \brief Base Class to code HMPID Raw Data file
///

#ifndef COMMON_HMPIDCODER2_H_
#define COMMON_HMPIDCODER2_H_

#include <cstdio>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"

#include "FairLogger.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDBase/Digit.h"

// ---- RDH 6  standard dimension -------
#define RAWBLOCKDIMENSION_W 2048
#define HEADERDIMENSION_W 16
#define PAYLOADDIMENSION_W 2032
#define PAYLOADMAXSPACE_W 2028

// ---- CHARGE CONSTANTS -----
#define CHARGE_CONST 150
#define CHARGE_RAND_MAX 400

using namespace o2::raw;

namespace o2
{

namespace hmpid
{

class HmpidCoder2
{
 public:
  int mVerbose;
  int mNumberOfEquipments;

  RawFileWriter mWriter{"HMP", false};

 private:
  // The standard definition of HMPID equipments at P2
  const int mEqIds[Geo::MAXEQUIPMENTS] = {0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11, 12, 13};
  const int mCruIds[Geo::MAXEQUIPMENTS] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3};
  const int mLinkIds[Geo::MAXEQUIPMENTS] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
  const int mFlpIds[Geo::MAXEQUIPMENTS] = {160, 160, 160, 160, 160, 160, 160, 160, 161, 161, 161, 161, 161, 161};

  char mFileName160[512];
  char mFileName161[512];
  uint32_t* mPayloadBufferPtr;
  int mEventSizePerEquipment[Geo::MAXEQUIPMENTS];
  int mEventPadsPerEquipment[Geo::MAXEQUIPMENTS];
  int mPailoadBufferDimPerEquipment;
  uint32_t* mPadMap;
  long mPadsCoded;
  bool mSkipEmptyEvents;

 public:
  HmpidCoder2(int numOfEquipments);
  virtual ~HmpidCoder2();

  void setVerbosity(int Level)
  {
    mVerbose = Level;
  };
  int getVerbosity()
  {
    return (mVerbose);
  };
  int getNumberOfEquipments()
  {
    return (mNumberOfEquipments);
  };
  void setSkipEmptyEvents(bool Skip)
  {
    mSkipEmptyEvents = Skip;
  }
  bool getSkipEmptyEvents()
  {
    return (mSkipEmptyEvents);
  }

  void openOutputStream(const char* OutputFileName, bool perFlpFile = false);
  void closeOutputStream();

  void codeEventChunkDigits(std::vector<Digit>& digits);
  void dumpResults();

 private:
  void getEquipCoord(int Equi, uint32_t* CruId, uint32_t* LinkId);
  int getEquipmentPadIndex(int eq, int col, int dil, int cha);
  void fillTheOutputBuffer(uint32_t* padMap);
  void writePaginatedEvent(uint32_t orbit, uint16_t bc);
};

} // namespace hmpid
} // namespace o2

#endif /* COMMON_HMPIDCODER_H_ */
