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
/// \file   HmpidDecoder.h
/// \author Antonio Franco - INFN Bari
/// \brief Base Class to decode HMPID Raw Data stream
///

#ifndef COMMON_HMPIDDECODER2_H_
#define COMMON_HMPIDDECODER2_H_

#include <cstdio>
#include <cstdint>
#include <iostream>
#include <cstring>

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileReader.h"

#include "DataFormatsHMP/Digit.h"

#include <fairlogger/Logger.h>

#include "HMPIDReconstruction/HmpidEquipment.h"

#define MAXDESCRIPTIONLENGHT 70

// ---- RDH 6  standard dimension -------
#define RAWBLOCKDIMENSION_W 2048
#define HEADERDIMENSION_W 16
#define PAYLOADDIMENSION_W 2032

// ---- Defines for the decoding
#define WTYPE_ROW 1
#define WTYPE_EOS 2
#define WTYPE_PAD 3
#define WTYPE_EOE 4
#define WTYPE_NONE 0

using namespace o2::raw;

// Hmpid Equipment class
namespace o2
{

namespace hmpid
{

class HmpidDecoder2
{

  // Members
 public:
  int mVerbose;
  HmpidEquipment* mTheEquipments[Geo::MAXEQUIPMENTS];
  int mNumberOfEquipments;

  static char sErrorDescription[MAXERRORS][MAXDESCRIPTIONLENGHT];
  static char sHmpidErrorDescription[MAXHMPIDERRORS][MAXDESCRIPTIONLENGHT];

 public:
  uint64_t mHeEvent;
  int mHeBusy;
  int mNumberWordToRead;
  int mPayloadTail;

  int mHeFEEID;
  int mHeSize;
  int mHeVer;
  int mHePrior;
  int mHeStop;
  int mHePages;
  int mEquipment;

  int mHeOffsetNewPack;
  int mHeMemorySize;

  int mHeDetectorID;
  int mHeDW;
  int mHeCruID;
  int mHePackNum;
  int mHePAR;

  int mHePageNum;
  int mHeLinkNum;
  int mHeFirmwareVersion;
  int mHeHmpidError;
  int mHeBCDI;
  int mHeORBIT;
  int mHeTType;

  uint32_t* mActualStreamPtr;
  uint32_t* mEndStreamPtr;
  uint32_t* mStartStreamPtr;
  int mRDHSize;
  int mRDHAcceptedVersion;

  o2::InteractionRecord mIntReco;
  std::vector<o2::hmpid::Digit> mDigits;

  // Methods
 public:
  HmpidDecoder2(int* EqIds, int* CruIds, int* LinkIds, int numOfEquipments);
  HmpidDecoder2(int numOfEquipments);
  ~HmpidDecoder2();

  void init();
  bool setUpStream(void* Buffer, long BufferLen);
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
  int getEquipmentIndex(int EquipmentId);
  int getEquipmentIndex(int CruID, int LinkId);
  int getEquipmentID(int CruId, int LinkId);

  void decodePage(uint32_t** streamBuffer);
  void decodePageFast(uint32_t** streamBuf);
  bool decodeBuffer();
  bool decodeBufferFast();

  uint16_t getChannelSamples(int Equipment, int Column, int Dilogic, int Channel);
  double getChannelSum(int Equipment, int Column, int Dilogic, int Channel);
  double getChannelSquare(int Equipment, int Column, int Dilogic, int Channel);
  uint16_t getPadSamples(int Module, int Row, int Column);
  double getPadSum(int Module, int Row, int Column);
  double getPadSquares(int Module, int Row, int Column);

  void dumpErrors(int Equipment);
  void dumpPads(int Equipment, int type = 0);
  void writeSummaryFile(char* summaryFileName);

  float getAverageEventSize(int Equipment);
  float getAverageBusyTime(int Equipment);

 protected:
  int checkType(uint32_t wp, int* p1, int* p2, int* p3, int* p4);
  bool isRowMarker(uint32_t wp, int* Err, int* rowSize, int* mark);
  bool isSegmentMarker(uint32_t wp, int* Err, int* segSize, int* Seg, int* mark);
  bool isEoEmarker(uint32_t wp, int* Err, int* Col, int* Dilogic, int* Eoesize);
  void setPad(HmpidEquipment* eq, int col, int dil, int ch, uint16_t charge);

 public:
  bool decodeHmpidError(int ErrorField, char* outbuf);
  void dumpHmpidError(int ErrorField);
  bool isPadWord(uint32_t wp, int* Err, int* Col, int* Dilogic, int* Channel, int* Charge);
  int decodeHeader(uint32_t* streamPtrAdr, int* EquipIndex);
  HmpidEquipment* evaluateHeaderContents(int EquipmentIndex);
  void updateStatistics(HmpidEquipment* eq);

 protected:
  bool getBlockFromStream(uint32_t** streamPtr, uint32_t Size);
  bool getHeaderFromStream(uint32_t** streamPtr);
  bool getWordFromStream(uint32_t* word);
  uint32_t readWordFromStream();
  uint32_t* getActualStreamPtr()
  {
    return (mActualStreamPtr);
  };
};
} // namespace hmpid
} // namespace o2
#endif /* COMMON_HMPIDDECODER2ß_H_ */
