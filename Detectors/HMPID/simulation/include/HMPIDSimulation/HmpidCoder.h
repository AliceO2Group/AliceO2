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

#ifndef COMMON_HMPIDCODER_H_
#define COMMON_HMPIDCODER_H_

#include <cstdio>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>

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

namespace o2
{

namespace hmpid
{

class HmpidCoder
{
 public:
  int mVerbose;
  int mNumberOfEquipments;

 private:
  // The standard definition of HMPID equipments at P2
  const int mEqIds[Geo::MAXEQUIPMENTS] = {0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11, 12, 13};
  const int mCruIds[Geo::MAXEQUIPMENTS] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3};
  const int mLinkIds[Geo::MAXEQUIPMENTS] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
  const int mFlpIds[Geo::MAXEQUIPMENTS] = {160, 160, 160, 160, 160, 160, 160, 160, 161, 161, 161, 161, 161, 161};

  bool mRandomCharge;
  bool mRandomOccupancy;

  int mOccupancyPercentage;

  int mPailoadBufferDimPerEquipment;
  uint32_t* mPayloadBufferPtr;
  uint32_t* mEventBufferPtr;
  uint32_t* mEventBufferBasePtr;
  int mEventSizePerEquipment[Geo::MAXEQUIPMENTS];
  uint32_t mPacketCounterPerEquipment[Geo::MAXEQUIPMENTS];
  bool mSkipEmptyEvents;
  bool mFixedPacketLenght;

  std::string mFileName160;
  std::string mFileName161;
  FILE* mOutStream160;
  FILE* mOutStream161;

  long mPadsCoded;
  long mPacketsCoded;

  std::vector<Digit> mDigits;
  uint32_t mPreviousOrbit = 0;
  uint16_t mPreviousBc = 0;
  long mLastProcessedDigit = 0;
  uint32_t* mPadMap;

 public:
  HmpidCoder(int numOfEquipments, bool skipEmptyEvents = false, bool fixedPacketLenght = true);
  virtual ~HmpidCoder();

  void reset();
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

  void setChargeRandomize(bool isRandom)
  {
    mRandomCharge = isRandom;
  };
  bool getChargeRandomize()
  {
    return (mRandomCharge);
  };

  void setOccupancyRandomize(bool isRandom)
  {
    mRandomOccupancy = isRandom;
  };
  bool getOccupancyRandomize()
  {
    return (mRandomOccupancy);
  };

  void setOccupancy(int Occupancy)
  {
    mOccupancyPercentage = Occupancy;
  };
  int getOccupancy()
  {
    return (mOccupancyPercentage);
  };

  int addDigitsChunk(std::vector<Digit>& digits);
  void codeDigitsChunk(bool flushBuffer = false);
  void codeDigitsVector(std::vector<Digit>& digits);

  void openOutputStream(const char* OutputFileName);
  void closeOutputStream();

  void codeRandomEvent(uint32_t orbit, uint16_t bc);
  void codeTest(int Events, uint16_t charge);

  void dumpResults();

 protected:
  void createRandomPayloadPerEvent();
  void savePacket(int Flp, int packetSize);
  int calculateNumberOfPads();
  void codeEventChunkDigits(std::vector<Digit>& digits, bool flushVector = false);

 private:
  void getEquipCoord(int Equi, uint32_t* CruId, uint32_t* LinkId);
  int getEquipmentPadIndex(int eq, int col, int dil, int cha);
  int writeHeader(uint32_t* Buffer, uint32_t WordsToRead, uint32_t PayloadWords, int Equip, uint32_t PackNum, uint32_t BCDI, uint32_t ORBIT, uint32_t PageNum);
  void fillPadsMap(uint32_t* padMap);
  void fillTheOutputBuffer(uint32_t* padMap);
  void writePaginatedEvent(uint32_t orbit, uint16_t bc);
};

} // namespace hmpid
} // namespace o2

#endif /* COMMON_HMPIDCODER_H_ */
