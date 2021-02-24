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
/// \file   HmpidCoder.cxx
/// \author Antonio Franco - INFN Bari
/// \brief Base Class for coding HMPID Raw Data File
/// \version 1.0
/// \date 24 set 2020

#include "HMPIDBase/Digit.h"
#include "HMPIDSimulation/HmpidCoder.h"

using namespace o2::hmpid;

HmpidCoder::HmpidCoder(int numOfEquipments, bool skipEmptyEvents, bool fixedPacketsLenght)
{

  mNumberOfEquipments = numOfEquipments;
  mSkipEmptyEvents = skipEmptyEvents;
  mFixedPacketLenght = fixedPacketsLenght;

  mVerbose = 0;
  mOccupancyPercentage = 0;
  mRandomOccupancy = false;
  mRandomCharge = false;
  mOutStream160 = nullptr;
  mOutStream161 = nullptr;
  mFileName160 = "";
  mFileName161 = "";

  mPailoadBufferDimPerEquipment = ((Geo::N_SEGMENTS * (Geo::N_COLXSEGMENT * (Geo::N_DILOGICS * (Geo::N_CHANNELS + 1) + 1) + 1)) + 10);

  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++) {
    mPacketCounterPerEquipment[i] = 0;
  }
  mPayloadBufferPtr = (uint32_t*)std::malloc(mNumberOfEquipments * sizeof(uint32_t) * mPailoadBufferDimPerEquipment);
  mEventBufferBasePtr = (uint32_t*)std::malloc(sizeof(uint32_t) * RAWBLOCKDIMENSION_W);
  mEventBufferPtr = mEventBufferBasePtr;

  mPadMap = (uint32_t*)std::malloc(sizeof(uint32_t) * Geo::N_HMPIDTOTALPADS);
  // TODO: Add memory allocation error check

  reset();
}

HmpidCoder::~HmpidCoder()
{
  // TODO Auto-generated destructor stub
  std::free(mPayloadBufferPtr);
  std::free(mEventBufferBasePtr);
  std::free(mPadMap);
}

void HmpidCoder::reset()
{
  srand((unsigned)time(nullptr));
  mPadsCoded = 0;
  mPacketsCoded = 0;
  mDigits.clear();
  mPreviousOrbit = 0;
  mPreviousBc = 0;
  mLastProcessedDigit = 0;

  return;
}

// =====================  Random production of Raw Files ===================
int HmpidCoder::calculateNumberOfPads()
{
  int occupancyPercentage = 0;
  if (mRandomOccupancy) {
    occupancyPercentage = rand() % 1000;
  } else {
    occupancyPercentage = mOccupancyPercentage;
  }
  return (occupancyPercentage * Geo::MAXEQUIPMENTS * Geo::N_EQUIPMENTTOTALPADS / 1000);
}

void HmpidCoder::fillPadsMap(uint32_t* padMap)
{
  int numberOfpads = calculateNumberOfPads();
  int mo, yc, xr, eq, col, dil, cha;
  for (int i = 0; i < numberOfpads; i++) {
    mo = rand() % Geo::N_MODULES;
    xr = rand() % Geo::N_XROWS;
    yc = rand() % Geo::N_YCOLS;
    Digit::Absolute2Equipment(mo, xr, yc, &eq, &col, &dil, &cha);
    int index = getEquipmentPadIndex(eq, col, dil, cha);
    if (padMap[index] == 0) {
      if (mRandomCharge) {
        padMap[index] = (rand() % CHARGE_RAND_MAX) + 50;
      } else {
        padMap[index] = CHARGE_CONST;
      }
    } else {
      i--;
    }
  }
  return;
}

void HmpidCoder::createRandomPayloadPerEvent()
{
  memset(mPadMap, 0, sizeof(uint32_t) * Geo::N_HMPIDTOTALPADS);

  fillPadsMap(mPadMap);
  fillTheOutputBuffer(mPadMap);

  return;
}

void HmpidCoder::codeRandomEvent(uint32_t orbit, uint16_t bc)
{
  createRandomPayloadPerEvent();
  writePaginatedEvent(orbit, bc);
  return;
}
// =====================  END of Random production of Raw Files ===================

void HmpidCoder::getEquipCoord(int Equi, uint32_t* CruId, uint32_t* LinkId)
{
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++) {
    if (mEqIds[i] == Equi) {
      *CruId = mCruIds[i];
      *LinkId = mLinkIds[i];
      return;
    }
  }
  *CruId = mCruIds[0];
  *LinkId = mLinkIds[0];
  return;
}

constexpr int p1() { return (Geo::N_SEGMENTS * Geo::N_COLXSEGMENT * Geo::N_DILOGICS * Geo::N_CHANNELS); }
constexpr int p2() { return (Geo::N_DILOGICS * Geo::N_CHANNELS); }

int HmpidCoder::getEquipmentPadIndex(int eq, int col, int dil, int cha)
{
  return (eq * p1() + col * p2() + dil * Geo::N_CHANNELS + cha);
}

void HmpidCoder::fillTheOutputBuffer(uint32_t* padMap)
{
  uint32_t rowMarker, segMarker, eoeMarker, padWord;
  uint32_t rowSize;
  uint32_t ptr = 0;
  int pads[Geo::MAXEQUIPMENTS];
  int padsCount;
  int segSize;

  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++) {
    mEventSizePerEquipment[i] = 0;
  }

  for (int eq = 0; eq < mNumberOfEquipments; eq++) {
    int startPtr = ptr;
    padsCount = 0;
    for (int s = 1; s <= Geo::N_SEGMENTS; s++) {
      segSize = 0;
      for (int c = 1; c <= Geo::N_COLXSEGMENT; c++) {

        // ---- Pre calculate the size of each column
        for (int j = 0; j < Geo::N_DILOGICS; j++) {
          pads[j] = 0;
        }
        rowSize = 0;
        for (int j = 0; j < Geo::N_DILOGICS; j++) {
          for (int k = 0; k < Geo::N_CHANNELS; k++) {
            int idx = getEquipmentPadIndex(eq, ((s - 1) * Geo::N_COLXSEGMENT + (c - 1)), j, k);
            if (padMap[idx] > 0) {
              pads[j]++;
              rowSize++;
              padsCount++;
            }
          }
        }
        rowSize += Geo::N_DILOGICS;
        segSize += (rowSize + 1);
        rowMarker = 0x000036A8 | ((rowSize << 16) & 0x03ff0000);

        // ---- fills the Payload Buffer
        mPayloadBufferPtr[ptr++] = rowMarker;
        int col = (s - 1) * Geo::N_COLXSEGMENT + c;
        for (int d = 1; d <= Geo::N_DILOGICS; d++) {
          for (int p = 0; p < Geo::N_CHANNELS; p++) {
            int idx = getEquipmentPadIndex(eq, ((s - 1) * Geo::N_COLXSEGMENT + (c - 1)), (d - 1), p);
            if (padMap[idx] > 0) {
              padWord = ((col << 22) & 0x07c00000) | ((d << 18) & 0x003C0000) | ((p << 12) & 0x0003F000) | (padMap[idx] & 0x00000FFF);
              mPayloadBufferPtr[ptr++] = padWord;
            }
          }
          eoeMarker = 0x08000080 | ((col << 22) & 0x07c00000) | (d << 18 & 0x003C0000) | (pads[d - 1] & 0x0000007F);
          mPayloadBufferPtr[ptr++] = eoeMarker;
        }
      }
      segSize += 1;
      segMarker = 0xAB000000 | ((segSize << 8) & 0x000fff00) | (s & 0x0000000F);
      mPayloadBufferPtr[ptr++] = segMarker;
    }

    if (mSkipEmptyEvents && padsCount == 0) { // Sets the lenght of Events
      mEventSizePerEquipment[eq] = 0;
    } else {
      mEventSizePerEquipment[eq] = ptr - startPtr;
    }
  }
  return;
}

void HmpidCoder::writePaginatedEvent(uint32_t orbit, uint16_t bc)
{
  int nWordToRead;
  int count;
  int payloatPtr = 0;
  if (orbit == 0 || bc == 0) {
    std::cerr << "HmpidCoder [ERROR] : Bad Orbit/BCid number (ORBIT=" << orbit << " BCID=" << bc << ")" << std::endl;
    return;
  }

  for (int eq = 0; eq < mNumberOfEquipments; eq++) {
    int EventSize = mEventSizePerEquipment[eq];
    if (EventSize == 0) {
      continue; // Skips the Events sized with 0
    }
    int EquipmentCounter = 0;
    int numOfPages = EventSize / PAYLOADMAXSPACE_W + 1;

    for (uint32_t PageNum = 1; PageNum <= numOfPages; PageNum++) {
      count = 0;
      while (count < PAYLOADMAXSPACE_W && EquipmentCounter < EventSize) {
        mEventBufferPtr[HEADERDIMENSION_W + count++] = mPayloadBufferPtr[payloatPtr++];
        EquipmentCounter++;
      }
      nWordToRead = count;

      int nWordsTail = (mFixedPacketLenght) ? PAYLOADDIMENSION_W : ((count / 8 + 1) * 8);
      while (count < nWordsTail) {
        mEventBufferPtr[HEADERDIMENSION_W + count++] = 0;
      }
      uint32_t PackNum = mPacketCounterPerEquipment[eq]++;
      int PacketSize = writeHeader(mEventBufferPtr, nWordToRead, nWordsTail, mEqIds[eq], PackNum, bc, orbit, PageNum);
      savePacket(mFlpIds[eq], PacketSize);
    }
  }
  return;
}

int HmpidCoder::writeHeader(uint32_t* Buffer, uint32_t WordsToRead, uint32_t PayloadWords,
                            int Equip, uint32_t PackNum, uint32_t BCDI, uint32_t ORBIT, uint32_t PageNum)
{
  uint32_t CruId, LinkId;
  uint32_t TType = 0;
  uint32_t HeStop = 0;
  uint32_t FirmVers = 9;
  uint32_t HeError = 0;
  uint32_t Busy = 3000;
  uint32_t PAR = 0;

  uint32_t MemSize = WordsToRead * sizeof(uint32_t) + HEADERDIMENSION_W * sizeof(uint32_t);
  uint32_t OffsetNext = PayloadWords * sizeof(uint32_t) + HEADERDIMENSION_W * sizeof(uint32_t);
  getEquipCoord(Equip, &CruId, &LinkId);

  //      FEEID  Header Size     Header version
  Buffer[0] = 0xFFF00000 | ((Equip & 0x0F) << 16) | 0x00004000 | 0x00000006;
  //      Priority      System ID
  Buffer[1] = 0x00000100 | 0x00000006;
  //  .....Memory Size       HeOffsetNewPack;
  Buffer[2] = (MemSize << 16) | (OffsetNext & 0x0000FFFF);
  //   DW     CruId   PacNum    Link Num
  Buffer[3] = 0x10000000 | ((CruId & 0x00FF) << 16) | ((PackNum & 0x0FF) << 8) | (LinkId & 0x0FF);
  Buffer[4] = 0x00000FFF & BCDI;
  Buffer[5] = ORBIT;
  Buffer[6] = 0;
  Buffer[7] = 0;
  Buffer[8] = TType;
  Buffer[9] = ((HeStop & 0x00ff0000) << 16) | (PageNum & 0x0000FFFF);
  Buffer[10] = 0;
  Buffer[11] = 0;
  Buffer[12] = (Busy << 9) | ((HeError & 0x000001F0) << 4) | (FirmVers & 0x0000000f);
  Buffer[13] = PAR;
  Buffer[14] = 0xAAAA0001;
  Buffer[15] = 0xAAAA0001;
  return (OffsetNext);
}

void HmpidCoder::codeDigitsChunk(bool flushBuffer)
{
  codeEventChunkDigits(mDigits, flushBuffer);
  return;
}

int HmpidCoder::addDigitsChunk(std::vector<Digit>& digits)
{
  mDigits.insert(mDigits.end(), digits.begin(), digits.end());
  return (mDigits.size());
}

void HmpidCoder::codeDigitsVector(std::vector<Digit>& digits)
{
  codeEventChunkDigits(digits, true);
  return;
}

void HmpidCoder::codeEventChunkDigits(std::vector<Digit>& digits, bool flushVector)
{
  int eq, col, dil, cha, mo, x, y, idx;
  uint32_t orbit = 0;
  uint16_t bc = 0;

  int padsCount = 0;
  int lastEventDigit = -1;
  mLastProcessedDigit = -1;

  for (o2::hmpid::Digit d : digits) {
    orbit = d.getOrbit();
    bc = d.getBC();
    lastEventDigit++;

    if (orbit != mPreviousOrbit || bc != mPreviousBc) { //the event is changed
      if (mPreviousOrbit != 0 || mPreviousBc != 0) {    // isn't the first !
        fillTheOutputBuffer(mPadMap);
        writePaginatedEvent(mPreviousOrbit, mPreviousBc);
        mLastProcessedDigit = lastEventDigit;
      }
      memset(mPadMap, 0, sizeof(uint32_t) * Geo::N_HMPIDTOTALPADS); // Update for the new event
      mPreviousOrbit = orbit;
      mPreviousBc = bc;
    }
    Digit::Pad2Equipment(d.getPadID(), &eq, &col, &dil, &cha); // From Digit to Hardware coords
    eq = mEqIds[eq];                                           // converts the Equipment Id in Cru/Link position ref
    idx = getEquipmentPadIndex(eq, col, dil, cha);             // finally to the unique padmap index

    if (mPadMap[idx] != 0) { // We already have the pad set
      std::cerr << "HmpidCoder [ERROR] : Duplicated DIGIT =" << d << " (" << eq << "," << col << "," << dil << "," << cha << ")" << std::endl;
    } else {
      mPadMap[idx] = d.getCharge();
      padsCount++;
    }
  }
  mPreviousOrbit = 0;
  mPreviousBc = 0;
  mPadsCoded += padsCount;

  if (flushVector) {
    fillTheOutputBuffer(mPadMap); // Finalize the last event
    writePaginatedEvent(orbit, bc);
    digits.clear();
  } else {
    if (mLastProcessedDigit >= 0) {
      digits.erase(digits.begin(), digits.begin() + mLastProcessedDigit);
    }
  }
  memset(mPadMap, 0, sizeof(uint32_t) * Geo::N_HMPIDTOTALPADS); // Update for the new event
  return;
}

void HmpidCoder::codeTest(int Events, uint16_t charge)
{
  uint32_t orbit = 0;
  uint16_t bc = 0;

  //uint32_t *mPadMap = (uint32_t *) std::malloc(sizeof(uint32_t) * Geo::N_HMPIDTOTALPADS);
  int eq, col, dil, cha, mo, x, y, idx;

  for (int e = 0; e < Events; e++) {
    orbit++;
    bc++;
    charge++;
    for (eq = 0; eq < Geo::MAXEQUIPMENTS; eq++) {
      for (col = 0; col < Geo::N_COLUMNS; col++) {
        for (dil = 0; dil < Geo::N_DILOGICS; dil++) {
          for (cha = 0; cha < Geo::N_CHANNELS; cha++) {
            idx = getEquipmentPadIndex(eq, col, dil, cha);
            mPadMap[idx] = charge;
          }
        }
      }
    }
    fillTheOutputBuffer(mPadMap);
    writePaginatedEvent(orbit, bc);
    memset(mPadMap, 0, sizeof(uint32_t) * Geo::N_HMPIDTOTALPADS);
  }
  return;
}
// ====================== FILES management functions ======================

void HmpidCoder::savePacket(int Flp, int packetSize)
{
  mPacketsCoded++;
  if (Flp == 160) {
    fwrite(mEventBufferPtr, packetSize, sizeof(uint8_t), mOutStream160);
  } else {
    fwrite(mEventBufferPtr, packetSize, sizeof(uint8_t), mOutStream161);
  }
  return;
}

/// @throws TH_CREATEFILE Thrown if Fails to create the file
void HmpidCoder::openOutputStream(const char* OutputFileName)
{
  char FileName[512];
  sprintf(FileName, "%s%d%s", OutputFileName, 160, ".dat");
  mOutStream160 = fopen(FileName, "wb");
  if (mOutStream160 == nullptr) {
    LOG(ERROR) << "ERROR to open Output file " << FileName;
    throw nullptr;
  }
  mFileName160 = FileName;

  sprintf(FileName, "%s%d%s", OutputFileName, 161, ".dat");
  mOutStream161 = fopen(FileName, "wb");
  if (mOutStream161 == nullptr) {
    LOG(ERROR) << "ERROR to open Output file " << FileName;
    throw nullptr;
  }
  mFileName161 = FileName;
  return;
}

void HmpidCoder::closeOutputStream()
{
  fclose(mOutStream160);
  fclose(mOutStream161);
  return;
}

void HmpidCoder::dumpResults()
{
  std::cout << " ****  HMPID RawFile Coder : results ****" << std::endl;
  std::cout << " Created files : " << mFileName160 << " ," << mFileName161 << std::endl;
  std::cout << " Number of Pads coded : " << mPadsCoded << std::endl;
  std::cout << " Number of Packets written : " << mPacketsCoded << std::endl;
  std::cout << " ----------------------------------------" << std::endl;
}
