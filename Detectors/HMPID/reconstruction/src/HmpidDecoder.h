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
/// \file   HmpidDecoder.h
/// \author Antonio Franco - INFN Bari
/// \brief Base Class to decode HMPID Raw Data stream
///

#ifndef COMMON_HMPIDDECODER_H_
#define COMMON_HMPIDDECODER_H_

#include <cstdio>
#include <stdint.h>
#include <iostream>
#include <cstring>

#include "FairLogger.h"

#include "HmpidEquipment.h"

#define MAXDESCRIPTIONLENGHT 50

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


// Hmpid Equipment class
namespace o2 {

namespace hmpid {

class HmpidDecoder
{

  // Members
  public:
    int mVerbose;
    HmpidEquipment *mTheEquipments[Geo::MAXEQUIPMENTS];
    int mNumberOfEquipments;

    static char sErrorDescription[MAXERRORS][MAXDESCRIPTIONLENGHT];
    static char sHmpidErrorDescription[MAXHMPIDERRORS][MAXDESCRIPTIONLENGHT];

  public:
    int mHeEvent;
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

    int32_t *mActualStreamPtr;
    int32_t *mEndStreamPtr;
    int32_t *mStartStreamPtr;

  // Methods
  public:
    HmpidDecoder(int *EqIds, int *CruIds, int *LinkIds, int numOfEquipments);
    HmpidDecoder(int numOfEquipments);
    ~HmpidDecoder();

    void init();
    virtual bool setUpStream(void *Buffer, long BufferLen) = 0;
    void setVerbosity(int Level)
    {
      mVerbose = Level;
    }
    ;
    int getVerbosity()
    {
      return (mVerbose);
    }
    ;

    int getNumberOfEquipments()
    {
      return (mNumberOfEquipments);
    }
    ;
    int getEquipmentIndex(int EquipmentId);
    int getEquipmentIndex(int CruID, int LinkId);
    int getEquipmentID(int CruId, int LinkId);

    bool decodeBuffer();
    bool decodeBufferFast();

    uint16_t getChannelSamples(int Equipment, int Column, int Dilogic, int Channel);
    double getChannelSum(int Equipment, int Column, int Dilogic, int Channel);
    double getChannelSquare(int Equipment, int Column, int Dilogic, int Channel);
    uint16_t getPadSamples(int Module, int Column, int Row);
    double getPadSum(int Module, int Column, int Row);
    double getPadSquares(int Module, int Column, int Row);

    void dumpErrors(int Equipment);
    void dumpPads(int Equipment, int type = 0);
    void writeSummaryFile(char *summaryFileName);

    float getAverageEventSize(int Equipment);
    float getAverageBusyTime(int Equipment);

  protected:
    int checkType(int32_t wp, int *p1, int *p2, int *p3, int *p4);
    bool isRowMarker(int32_t wp, int *Err, int *rowSize, int *mark);
    bool isSegmentMarker(int32_t wp, int *Err, int *segSize, int *Seg, int *mark);
    bool isPadWord(int32_t wp, int *Err, int *Col, int *Dilogic, int *Channel, int *Charge);
    bool isEoEmarker(int32_t wp, int *Err, int *Col, int *Dilogic, int *Eoesize);
    int decodeHeader(int32_t *streamPtrAdr, int *EquipIndex);
    bool decodeHmpidError(int ErrorField, char *outbuf);
    void dumpHmpidError(int ErrorField);
    HmpidEquipment* evaluateHeaderContents(int EquipmentIndex);
    void updateStatistics(HmpidEquipment *eq);

    virtual void setPad(HmpidEquipment *eq, int col, int dil, int ch, int charge) = 0;

    virtual bool getBlockFromStream(int32_t **streamPtr, uint32_t Size) = 0;
    virtual bool getHeaderFromStream(int32_t **streamPtr) = 0;
    virtual bool getWordFromStream(int32_t *word) = 0;
    int32_t* getActualStreamPtr()
    {
      return (mActualStreamPtr);
    }
    ;

};
}
}
#endif /* COMMON_HMPIDDECODER_H_ */
