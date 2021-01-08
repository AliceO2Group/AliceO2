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
/// \file   HmpidDecoder.cxx
/// \author Antonio Franco - INFN Bari
/// \brief Base Class to decode HMPID Raw Data stream
/// \version 1.0
/// \date 17/11/2020

/* ------ HISTORY ---------
*/

#include "FairLogger.h"
#include "HmpidDecoder.h"

using namespace o2::hmpid;

// ============= HmpidDecoder Class implementation =======


/// Decoding Error Messages Definitions
char HmpidDecoder::sErrorDescription[MAXERRORS][MAXDESCRIPTIONLENGHT] = { "Word that I don't known !",
    "Row Marker Word with 0 words", "Duplicated Pad Word !", "Row Marker Wrong/Lost -> to EoE",
    "Row Marker Wrong/Lost -> to EoE", "Row Marker reports an ERROR !", "Lost EoE Marker !", "Double EoE marker",
    "Wrong size definition in EoE Marker", "Double Mark Word", "Wrong Size in Segment Marker", "Lost EoS Marker !",
    "HMPID Header Errors" };

/// HMPID Firmware Error Messages Definitions
char HmpidDecoder::sHmpidErrorDescription[MAXHMPIDERRORS][MAXDESCRIPTIONLENGHT] = { "L0 Missing,"
    "L1 is received without L0", "L1A signal arrived before the L1 Latency", "L1A signal arrived after the L1 Latency",
    "L1A is missing or L1 timeout", "L1A Message is missing or L1 Message" };

/// Constructor : accepts the number of equipments to define
///               The mapping is the default at P2
///               Allocates instances for all defined equipments
///               normally it is equal to 14
/// @param[in] numOfEquipments : the number of equipments to define [1..14]
HmpidDecoder::HmpidDecoder(int numOfEquipments)
{
  // The standard definition of HMPID equipments at P2
  int EqIds[] = { 0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11, 12, 13 };
  int CruIds[] = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3 };
  int LinkIds[] = { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2 };

  mNumberOfEquipments = numOfEquipments;
  for (int i = 0; i < mNumberOfEquipments; i++) {
    mTheEquipments[i] = new HmpidEquipment(EqIds[i], CruIds[i], LinkIds[i]);
  }
}

/// Constructor : accepts the number of equipments to define
///               and their complete address map
///               Allocates instances for all defined equipments
///
///  The Address map is build from three array
/// @param[in] numOfEquipments : the number of equipments to define [1..14]
/// @param[in] *EqIds : the pointer to the Equipments ID array
/// @param[in] *CruIds : the pointer to the CRU ID array
/// @param[in] *LinkIds : the pointer to the Link ID array
HmpidDecoder::HmpidDecoder(int *EqIds, int *CruIds, int *LinkIds, int numOfEquipments)
{
  mNumberOfEquipments = numOfEquipments;
  for (int i = 0; i < mNumberOfEquipments; i++) {
    mTheEquipments[i] = new HmpidEquipment(EqIds[i], CruIds[i], LinkIds[i]);
  }
}

/// Destructor : remove the Equipments instances
HmpidDecoder::~HmpidDecoder()
{
  for (int i = 0; i < mNumberOfEquipments; i++) {
    delete mTheEquipments[i];
  }
}

/// Resets to 0 all the class members
void HmpidDecoder::init()
{
  mVerbose = 0;
  mHeEvent = 0;
  mHeBusy = 0;
  mNumberWordToRead = 0;
  mPayloadTail = 0;

  mHeFEEID = 0;
  mHeSize = 0;
  mHeVer = 0;
  mHePrior = 0;
  mHeStop = 0;
  mHePages = 0;
  mEquipment = 0;

  mHeOffsetNewPack = 0;
  mHeMemorySize = 0;

  mHeDetectorID = 0;
  mHeDW = 0;
  mHeCruID = 0;
  mHePackNum = 0;
  mHePAR = 0;
  mHePageNum = 0;
  mHeLinkNum = 0;
  mHeFirmwareVersion = 0;
  mHeHmpidError = 0;
  mHeBCDI = 0;
  mHeORBIT = 0;
  mHeTType = 0;

  mActualStreamPtr = 0;
  mEndStreamPtr = 0;
  mStartStreamPtr = 0;

}

/// Returns the Equipment Index (Pointer of the array) converting
/// the FLP hardware coords (CRU_Id and Link_Id)
/// @param[in] CruId : the CRU ID [0..3] -> FLP 160 = [0,1]  FLP 161 = [2,3]
/// @param[in] LinkId : the Link ID [0..3]
/// @returns EquipmentIndex : the index in the Equipment array [0..13] (-1 := error)
int HmpidDecoder::getEquipmentIndex(int CruId, int LinkId)
{
  for (int i = 0; i < mNumberOfEquipments; i++) {
    if (mTheEquipments[i]->getEquipmentId(CruId, LinkId) != -1) {
      return (i);
    }
  }
  return (-1);
}

/// Returns the Equipment Index (Pointer of the array) converting
/// the Equipment_ID (Firmaware defined Id AKA FFEID)
/// @param[in] EquipmentId : the Equipment ID [0..13]
/// @returns EquipmentIndex : the index in the Equipment array [0..13] (-1 := error)
int HmpidDecoder::getEquipmentIndex(int EquipmentId)
{
  for (int i = 0; i < mNumberOfEquipments; i++) {
    if (mTheEquipments[i]->getEquipmentId() == EquipmentId) {
      return (i);
    }
  }
  return (-1);
}

/// Returns the Equipment_ID converting the FLP hardware coords
/// @param[in] CruId : the CRU ID [0..3] -> FLP 160 = [0,1]  FLP 161 = [2,3]
/// @param[in] LinkId : the Link ID [0..3]
/// @returns EquipmentID : the ID of the Equipment [0..13] (-1 := error)
int HmpidDecoder::getEquipmentID(int CruId, int LinkId)
{
  for (int i = 0; i < mNumberOfEquipments; i++) {
    if (mTheEquipments[i]->getEquipmentId(CruId, LinkId) != -1) {
      return (mTheEquipments[i]->getEquipmentId());
    }
  }
  return (-1);
}

/// Scans the BitMap of Raw Data File word and detect the type
/// and the parameters
/// @param[in] wp : the word to analyze
/// @param[out] *p1 : first parameter extract (if it exists)
/// @param[out] *p2 : second parameter extract (if it exists)
/// @param[out] *p3 : third parameter extract (if it exists)
/// @param[out] *p4 : fourth parameter extract (if it exists)
/// @returns Type of Word : the type of word [0..4] (0 := undetect)
int HmpidDecoder::checkType(int32_t wp, int *p1, int *p2, int *p3, int *p4)
{
  if ((wp & 0x0000ffff) == 0x36A8 || (wp & 0x0000ffff) == 0x32A8 || (wp & 0x0000ffff) == 0x30A0
      || (wp & 0x0800ffff) == 0x080010A0) {
    *p2 = (wp & 0x03ff0000) >> 16; // Number of words of row
    *p1 = wp & 0x0000ffff;
    return (WTYPE_ROW);
  }
  if ((wp & 0xfff00000) >> 20 == 0xAB0) {
    *p2 = (wp & 0x000fff00) >> 8; // Number of words of Segment
    *p1 = (wp & 0xfff00000) >> 20;
    *p3 = wp & 0x0000000F;
    if (*p3 < 4 && *p3 > 0) {
      return (WTYPE_EOS);
    }
  }
  // #EX MASK Raul 0x3803FF80  # ex mask 0xF803FF80 - this is EoE marker 0586800B0
  if ((wp & 0x0803FF80) == 0x08000080) {
    *p1 = (wp & 0x07c00000) >> 22;
    *p2 = (wp & 0x003C0000) >> 18;
    *p3 = (wp & 0x0000007F);
    if (*p1 < 25 && *p2 < 11) {
      return (WTYPE_EOE);
    }
  }
  if ((wp & 0x08000000) == 0) { //  # this is a pad
    // PAD:0000.0ccc.ccdd.ddnn.nnnn.vvvv.vvvv.vvvv :: c=col,d=dilo,n=chan,v=value
    *p1 = (wp & 0x07c00000) >> 22;
    *p2 = (wp & 0x003C0000) >> 18;
    *p3 = (wp & 0x0003F000) >> 12;
    *p4 = (wp & 0x00000FFF);
    if (*p1 > 0 && *p1 < 25 && *p2 > 0 && *p2 < 11 && *p3 < 48) {
      return (WTYPE_PAD);
    }
  } else {
    return (WTYPE_NONE);
  }
  return (WTYPE_NONE);
}

/// Checks if is a Raw Marker and extract the Row Size
/// @param[in] wp : the word to check
/// @param[out] *Err : true if an error is detected
/// @param[out] *rowSize : the number of words of the row
/// @param[out] *mark : the row marker
/// @returns True if Row Marker is detected
bool HmpidDecoder::isRowMarker(int32_t wp, int *Err, int *rowSize, int *mark)
{
  if ((wp & 0x0000ffff) == 0x36A8 || (wp & 0x0000ffff) == 0x32A8 || (wp & 0x0000ffff) == 0x30A0
      || (wp & 0x0800ffff) == 0x080010A0) {
    *rowSize = (wp & 0x03ff0000) >> 16;      // # Number of words of row
    *mark = wp & 0x0000ffff;
    *Err = false;
    return (true);
  } else {
    *Err = true;
    return (false);
  }
}

/// Checks if is a Segment Marker and extracts the Segment number and the size
/// @param[in] wp : the word to check
/// @param[out] *Err : true if an error is detected
/// @param[out] *segSize : the number of words of the segment
/// @param[out] *Seg : the Segment number [1..3]
/// @param[out] *mark : the Segment Marker
/// @returns True if Segment Marker is detected
bool HmpidDecoder::isSegmentMarker(int32_t wp, int *Err, int *segSize, int *Seg, int *mark)
{
  *Err = false;
  if ((wp & 0xfff00000) >> 20 == 0xAB0) {
    *segSize = (wp & 0x000fff00) >> 8;      // # Number of words of Segment
    *mark = (wp & 0xfff00000) >> 20;
    *Seg = wp & 0x0000000F;

    if (*Seg > 3 || *Seg < 1) {
      ILOG(INFO) << " Wrong segment Marker Word, bad Number of segment" << *Seg << "!" << FairLogger::endl;
      *Err = true;
    }
    return (true);
  } else {
    return (false);
  }
}

/// Checks if is a PAD Word and extracts all the parameters
/// PAD map : 0000.0ccc.ccdd.ddnn.nnnn.vvvv.vvvv.vvvv :: c=col,d=dilo,n=chan,v=value
/// @param[in] wp : the word to check
/// @param[out] *Err : true if an error is detected
/// @param[out] *Col : the column number [1..24]
/// @param[out] *Dilogic : the dilogic number [1..10]
/// @param[out] *Channel : the channel number [0..47]
/// @param[out] *Charge : the value of Charge [0..4095]
/// @returns True if PAD Word is detected
bool HmpidDecoder::isPadWord(int32_t wp, int *Err, int *Col, int *Dilogic, int *Channel, int *Charge)
{
  *Err = false;
  if ((wp & 0x08000000) == 0) { //  # this is a pad
    *Col = (wp & 0x07c00000) >> 22;
    *Dilogic = (wp & 0x003C0000) >> 18;
    *Channel = (wp & 0x0003F000) >> 12;
    *Charge = (wp & 0x00000FFF);
    if (*Dilogic > 10 || *Channel > 47 || *Dilogic < 1 || *Col > 24 || *Col < 1) {
      ILOG(WARNING) << " Wrong Pad values Col=" << *Col << " Dilogic="<< *Dilogic << \
          " Channel=" << *Channel << " Charge=" << *Charge << FairLogger::endl;
      *Err = true;
    }
    return (true);
  } else {
    return (false);
  }
}

/// Checks if is a EoE Marker and extracts the Column, Dilogic and the size
/// @param[in] wp : the word to check
/// @param[out] *Err : true if an error is detected
/// @param[out] *Col : the column number [1..24]
/// @param[out] *Dilogic : the dilogic number [1..10]
/// @param[out] *Eoesize : the number of words for dilogic
/// @returns True if EoE marker is detected
bool HmpidDecoder::isEoEmarker(int32_t wp, int *Err, int *Col, int *Dilogic, int *Eoesize)
{
  *Err = false;
  // #EX MASK Raul 0x3803FF80  # ex mask 0xF803FF80 - this is EoE marker 0586800B0
  if ((wp & 0x0803FF80) == 0x08000080) {
    *Col = (wp & 0x07c00000) >> 22;
    *Dilogic = (wp & 0x003C0000) >> 18;
    *Eoesize = (wp & 0x0000007F);
    if (*Col > 24 || *Dilogic > 10) {
      ILOG(INFO) << " EoE size wrong definition. Col=" << *Col << " Dilogic=" << *Dilogic << FairLogger::endl;
      *Err = true;
    }
    return (true);
  } else {
    return (false);
  }
}

/// Decode the HMPID error BitMap field (5 bits) and returns true if there are
/// errors and in addition the concat string that contains the error messages
/// ATTENTION : the char * outbuf MUST point to a 250 bytes buffer
/// @param[in] ErrorField : the HMPID Error field
/// @param[out] *outbuf : the output buffer that contains the error description
/// @returns True if EoE marker is detected
bool HmpidDecoder::decodeHmpidError(int ErrorField, char *outbuf)
{
  int res = false;
  outbuf[0] = '\0';
  for (int i = 0; i < MAXHMPIDERRORS; i++) {
    if ((ErrorField & (0x01 << i)) != 0) {
      res = true;
      strcat(outbuf, sHmpidErrorDescription[i]);
    }
  }
  return (res);
}

/// This Decode the Raw Data Header, returns the EquipmentIndex
/// that is obtained with the FLP hardware coords
///
/// ATTENTION : the 'EquipIndex' parameter and the mEquipment member
/// are different data: the first is the pointer in the Equipments instances
/// array, the second is the FEE_ID number
///
/// The EVENT_NUMBER : actually is calculated from the ORBIT number
///
/// @param[in] *streamPtrAdr : the pointer to the Header buffer
/// @param[out] *EquipIndex : the Index to the Equipment Object Array [0..13]
/// @returns True every time
/// @throws TH_WRONGEQUIPINDEX Thrown if the Equipment Index is out of boundary (Equipment not recognized)
int HmpidDecoder::decodeHeader(int32_t *streamPtrAdr, int *EquipIndex)
{
  int32_t *buffer = streamPtrAdr; // Sets the pointer to buffer

  mHeFEEID = (buffer[0] & 0x000f0000) >> 16;
  mHeSize = (buffer[0] & 0x0000ff00) >> 8;
  mHeVer = (buffer[0] & 0x000000ff);
  mHePrior = (buffer[1] & 0x000000FF);
  mHeDetectorID = (buffer[1] & 0x0000FF00) >> 8;
  mHeOffsetNewPack = (buffer[2] & 0x0000FFFF);
  mHeMemorySize = (buffer[2] & 0xffff0000) >> 16;
  mHeDW = (buffer[3] & 0xF0000000) >> 24;
  mHeCruID = (buffer[3] & 0x0FF0000) >> 16;
  mHePackNum = (buffer[3] & 0x0000FF00) >> 8;
  mHeLinkNum = (buffer[3] & 0x000000FF);
  mHeBCDI = (buffer[4] & 0x00000FFF);
  mHeORBIT = buffer[5];
  mHeTType = buffer[8];
  mHePageNum = (buffer[9] & 0x0000FFFF);
  mHeStop = (buffer[9] & 0x00ff0000) >> 16;
  mHeBusy = (buffer[12] & 0xfffffe00) >> 9;
  mHeFirmwareVersion = buffer[12] & 0x0000000f;
  mHeHmpidError = (buffer[12] & 0x000001F0) >> 4;
  mHePAR = buffer[13] & 0x0000FFFF;

  *EquipIndex = getEquipmentIndex(mHeCruID, mHeLinkNum);
  //  mEquipment = (*EquipIndex != -1) ? mTheEquipments[*EquipIndex]->getEquipmentId() : -1;
  mEquipment = mHeFEEID;
  mNumberWordToRead = ((mHeMemorySize - mHeSize) / sizeof(uint32_t));
  mPayloadTail = ((mHeOffsetNewPack - mHeMemorySize) / sizeof(uint32_t));

  // ---- Event ID  : Actualy based on ORBIT NUMBER ...
  mHeEvent = mHeORBIT;

  ILOG(INFO) << "FEE-ID=" << mHeFEEID << " HeSize=" <<  mHeSize << " HePrior=" << mHePrior << " Det.Id=" << mHeDetectorID <<\
      " HeMemorySize=" << mHeMemorySize << " HeOffsetNewPack=" << mHeOffsetNewPack << FairLogger::endl;
  ILOG(INFO) << "      Equipment=" << mEquipment << " PakCounter=" <<  mHePackNum << " Link=" << mHeLinkNum << " CruID=" << \
      mHeCruID << " DW=" << mHeDW << " BC=" << mHeBCDI << " ORBIT=" << mHeORBIT << FairLogger::endl;
  ILOG(INFO) << "      TType=" << mHeTType << " HeStop=" << mHeStop << " PagesCounter=" << mHePageNum << " FirmVersion=" << \
      mHeFirmwareVersion << " BusyTime=" << mHeBusy << " Error=" << mHeHmpidError << " PAR=" << mHePAR << FairLogger::endl;
  ILOG(INFO) << "      Payload :  Words to read=" << mNumberWordToRead << " PailoadTail=" << mPayloadTail << FairLogger::endl;

  if (*EquipIndex == -1) {
    ILOG(ERROR) << "ERROR ! Bad equipment Number: " << mEquipment << FairLogger::endl;
    throw TH_WRONGEQUIPINDEX;
  }
  return (true);
}

/// Updates some information related to the Event
/// this function is called at the end of the event
/// @param[in] *eq : the pointer to the Equipment Object
void HmpidDecoder::updateStatistics(HmpidEquipment *eq)
{
  eq->mPadsPerEventAverage = ((eq->mPadsPerEventAverage * (eq->mNumberOfEvents - 1)) + eq->mSampleNumber)
      / (eq->mNumberOfEvents);
  eq->mEventSizeAverage = ((eq->mEventSizeAverage * (eq->mNumberOfEvents - 1)) + eq->mEventSize)
      / (eq->mNumberOfEvents);
  eq->mBusyTimeAverage = ((eq->mBusyTimeAverage * eq->mBusyTimeSamples) + eq->mBusyTimeValue)
      / (++(eq->mBusyTimeSamples));
  if (eq->mSampleNumber == 0)
    eq->mNumberOfEmptyEvents += 1;
  if (eq->mErrorsCounter > 0)
    eq->mNumberOfWrongEvents += 1;
  eq->mTotalPads += eq->mSampleNumber;
  eq->mTotalErrors += eq->mErrorsCounter;
  return;
}

/// Evaluates the content of the header and detect the change of the event
/// with the relevant updates...
/// @param[in] EquipmentIndex : the pointer to the Array of Equipments Array
/// @returns the Pointer to the modified Equipment object
HmpidEquipment* HmpidDecoder::evaluateHeaderContents(int EquipmentIndex)
{
  HmpidEquipment *eq = mTheEquipments[EquipmentIndex];
  if (mHeEvent != eq->mEventNumber) { // Is a new event
    if (eq->mEventNumber != -1) { // skip the first
      updateStatistics(eq); // update previous statistics
    }
    eq->mNumberOfEvents++;
    eq->mEventNumber = mHeEvent;
    eq->mBusyTimeValue = mHeBusy * 0.00000005;
    eq->mEventSize = 0;    // reset the event
    eq->mSampleNumber = 0;
    eq->mErrorsCounter = 0;
  }
  eq->mEventSize += mNumberWordToRead * sizeof(uint32_t); // Calculate the size in bytes
  if (mHeHmpidError != 0) {
    ILOG(ERROR) << "HMPID Header reports an error : " << mHeHmpidError << FairLogger::endl;
    dumpHmpidError(mHeHmpidError);
    eq->setError(ERR_HMPID);
  }
  return (eq);
}

/// --------------- Read Raw Data Buffer ---------------
/// Read the stream, decode the contents and store resuls.
/// ATTENTION : Assumes that the input stream was set
/// @throws TH_WRONGHEADER Thrown if the Fails to decode the Header
bool HmpidDecoder::decodeBuffer()
{
  // ---------resets the PAdMap-----------
  for (int i = 0; i < mNumberOfEquipments; i++) {
    mTheEquipments[i]->init();
    mTheEquipments[i]->resetPadMap();
    mTheEquipments[i]->resetErrors();
  }

  int type;
  int equipmentIndex = -1;
  int isIt;
  HmpidEquipment *eq;
  int32_t *streamBuf;
  ILOG(DEBUG) << "Enter decoding !" << FairLogger::endl;

  // Input Stream Main Loop
  while (true) {
    try {
      getHeaderFromStream(&streamBuf);
    }
    catch (int e) {
      // The stream end !
      ILOG(DEBUG) << "End main decoding loop !" << FairLogger::endl;
      break;
    }
    try {
      decodeHeader(streamBuf, &equipmentIndex);
    }
    catch (int e) {
      ILOG(ERROR) << "Failed to decode the Header !" << FairLogger::endl;
      throw TH_WRONGHEADER;
    }

    eq = evaluateHeaderContents(equipmentIndex);

    int wpprev = 0;
    int wp = 0;
    int newOne = true;
    int p1, p2, p3, p4;
    int error;

    int payIndex = 0;
    while (payIndex < mNumberWordToRead) {  //start the payload loop word by word
      if (newOne == true) {
        wpprev = wp;
        if (!getWordFromStream(&wp)) { // end the stream
          break;
        }

        type = checkType(wp, &p1, &p2, &p3, &p4);
        if (type == WTYPE_NONE) {
          if (eq->mWillBePad == true) { // try to recover the first pad !
            type = checkType((wp & 0xF7FFFFFF), &p1, &p2, &p3, &p4);
            if (type == WTYPE_PAD && p3 == 0 && eq->mWordsPerDilogicCounter == 0) {
              newOne = false; // # reprocess as pad
              continue;
            }
          }
          eq->setError(ERR_NOTKNOWN);
          ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_NOTKNOWN] << " [" << wp << "]" << FairLogger::endl;
          eq->mWordsPerRowCounter++;
          eq->mWordsPerSegCounter++;
          payIndex++;
          continue;
        }
      }

      if (eq->mWillBeRowMarker == true) { // #shoud be a Row Marker
        if (type == WTYPE_ROW) {
          eq->mColumnCounter++;
          eq->mWordsPerSegCounter++;
          eq->mRowSize = p2;
          switch (p2) {
            case 0: // Empty column
              eq->setError(ERR_ROWMARKEMPTY);
              ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_ROWMARKEMPTY] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
                  "[" << p1 << "]" << FairLogger::endl;
              eq->mWillBeRowMarker = true;
              break;
            case 0x3FF: // Error in column
              eq->setError(ERR_ROWMARKERROR);
              ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_ROWMARKERROR] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
                  "[" << p1 << "]" << FairLogger::endl;
              eq->mWillBeRowMarker = true;
              break;
            case 0x3FE: // Masked column
              ILOG(INFO) << "Equip=" << mEquipment << "The column=" << (eq->mSegment) * 8 + eq->mColumnCounter << " is Masked !" << FairLogger::endl;
              eq->mWillBeRowMarker = true;
              break;
            default:
              eq->mWillBeRowMarker = false;
              eq->mWillBePad = true;
              break;
          }
          newOne = true;
        } else {
          if (wp == wpprev) {
            eq->setError(ERR_DUPLICATEPAD);
            ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_DUPLICATEPAD] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
                "[" << p1 << "]" << FairLogger::endl;
            newOne = true;
          } else if (type == WTYPE_EOE) { // # Could be a EoE
            eq->mColumnCounter++;
            eq->setError(ERR_ROWMARKWRONG);
            eq->mWillBeRowMarker = false;
            eq->mWillBePad = true;
            newOne = true;
          } else if (type == WTYPE_PAD) { //# Could be a PAD
            eq->mColumnCounter++;
            eq->setError(ERR_ROWMARKLOST);
            ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_ROWMARKLOST] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
                "[" << p1 << "]" << FairLogger::endl;
            eq->mWillBeRowMarker = false;
            eq->mWillBePad = true;
            newOne = true;
          } else if (type == WTYPE_EOS) { // # Could be a EoS
            eq->mWillBeRowMarker = false;
            eq->mWillBeSegmentMarker = true;
            newOne = false;
          } else {
            eq->mColumnCounter++;
            eq->setError(ERR_ROWMARKLOST);
            ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_ROWMARKLOST] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
                "[" << p1 << "]" << FairLogger::endl;
            eq->mWillBeRowMarker = false;
            eq->mWillBePad = true;
            newOne = true;
          }
        }
      } else if (eq->mWillBePad == true) { // # We expect a pad
        //# PAD:0000.0ccc.ccdd.ddnn.nnnn.vvvv.vvvv.vvvv :: c=col,d=dilo,n=chan,v=value
        //   c = 1..24   d = 1..10  n = 0..47
        if (type == WTYPE_PAD) {
          newOne = true;
          if (wp == wpprev) {
            eq->setError(ERR_DUPLICATEPAD);
            ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_DUPLICATEPAD] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
                "[" << p1 << "]" << FairLogger::endl;
          } else if (p1 != (eq->mSegment * 8 + eq->mColumnCounter)) {        // # Manage
            // We try to recover the RowMarker misunderstanding
            isIt = isRowMarker(wp, &error, &p2, &p1);
            if (isIt == true && error == false) {
              type = WTYPE_ROW;
              newOne = false;
              eq->mWillBeEoE = true;
              eq->mWillBePad = false;
            } else {
              ILOG(DEBUG) << "Equip=" << mEquipment << " Mismatch in column" << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
                  "[" << p1 << "]" << FairLogger::endl;
              eq->mColumnCounter = p1 % 8;
            }
          } else {
            setPad(eq, p1 - 1, p2 - 1, p3, p4);
            eq->mWordsPerDilogicCounter++;
            eq->mSampleNumber++;
            if (p3 == 47) {
              eq->mWillBeEoE = true;
              eq->mWillBePad = false;
            }
          }
          eq->mWordsPerRowCounter++;
          eq->mWordsPerSegCounter++;
        } else if (type == WTYPE_EOE) { //# the pads are end ok
          eq->mWillBeEoE = true;
          eq->mWillBePad = false;
          newOne = false;
        } else if (type == WTYPE_ROW) { // # We Lost the EoE !
          // We try to recover the PAD misunderstanding
          isIt = isPadWord(wp, &error, &p1, &p2, &p3, &p4);
          if (isIt == true && error == false) {
            type = WTYPE_PAD;
            newOne = false;            // # reprocess as pad
          } else {
            eq->setError(ERR_LOSTEOEMARK);
            ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_LOSTEOEMARK] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
                "[" << p1 << "]" << FairLogger::endl;
            eq->mWillBeRowMarker = true;
            eq->mWillBePad = false;
            newOne = false;
          }
        } else if (type == WTYPE_EOS) {            // # We Lost the EoE !
          eq->setError(ERR_LOSTEOEMARK);
          ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_LOSTEOEMARK] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
              "[" << p1 << "]" << FairLogger::endl;
          eq->mWillBeSegmentMarker = true;
          eq->mWillBePad = false;
          newOne = false;
        }
      } else if (eq->mWillBeEoE == true) {            // # We expect a EoE
        if (type == WTYPE_EOE) {
          eq->mWordsPerRowCounter++;
          eq->mWordsPerSegCounter++;
          if (wpprev == wp) {
            eq->setError(ERR_DOUBLEEOEMARK);
            ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_DOUBLEEOEMARK] << " col=" << p1 << FairLogger::endl;
          } else if (p3 != eq->mWordsPerDilogicCounter) {
            eq->setError(ERR_WRONGSIZEINEOE);
            ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_WRONGSIZEINEOE] << " col=" << p1 << FairLogger::endl;
          }
          eq->mWordsPerDilogicCounter = 0;
          if (p2 == 10) {
            if (p1 % 8 != 0) {            // # we expect the Row Marker
              eq->mWillBeRowMarker = true;
            } else {
              eq->mWillBeSegmentMarker = true;
            }
          } else {
            eq->mWillBePad = true;
          }
          eq->mWillBeEoE = false;
          newOne = true;
        } else if (type == WTYPE_EOS) {            // We Lost the EoE !
          eq->setError(ERR_LOSTEOEMARK);
          ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_LOSTEOEMARK] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
              "[" << p1 << "]" << FairLogger::endl;
          eq->mWillBeSegmentMarker = true;
          eq->mWillBeEoE = false;
          newOne = false;
        } else if (type == WTYPE_ROW) { //# We Lost the EoE !
          eq->setError(ERR_LOSTEOEMARK);
          ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_LOSTEOEMARK] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
              "[" << p1 << "]" << FairLogger::endl;
          eq->mWillBeRowMarker = true;
          eq->mWillBeEoE = false;
          newOne = false;
        } else if (type == WTYPE_PAD) { // # We Lost the EoE !
          int typb, p1b, p2b, p3b, p4b;
          typb = checkType((wp | 0x08000000), &p1b, &p2b, &p3b, &p4b);
          if (typb == WTYPE_EOE && p3b == 48) {
            type = typb;
            p1 = p1b;
            p2 = p2b;
            p3 = p3b;
            p4 = p4b;
            newOne = false; // # reprocess as EoE
          } else {
            eq->setError(ERR_LOSTEOEMARK);
            ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_LOSTEOEMARK] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
                "[" << p1 << "]" << FairLogger::endl;
            eq->mWillBePad = true;
            eq->mWillBeEoE = false;
            newOne = false;
          }
        }
      } else if (eq->mWillBeSegmentMarker == true) { // # We expect a EoSegment
        if (wpprev == wp) {
          eq->setError(ERR_DOUBLEMARKWORD);
          ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_DOUBLEMARKWORD] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
              "[" << p1 << "]" << FairLogger::endl;
          newOne = true;
        } else if (type == 2) {
          if (abs(eq->mWordsPerSegCounter - p2) > 5) {
            eq->setError(ERR_WRONGSIZESEGMENTMARK);
            ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_WRONGSIZESEGMENTMARK] << " Seg=" << p2 << FairLogger::endl;
          }
          eq->mWordsPerSegCounter = 0;
          eq->mWordsPerRowCounter = 0;
          eq->mColumnCounter = 0;
          eq->mSegment = p3 % 3;
          eq->mWillBeRowMarker = true;
          eq->mWillBeSegmentMarker = false;
          newOne = true;
        } else {
          eq->setError(ERR_LOSTEOSMARK);
          ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_LOSTEOSMARK] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
              "[" << p1 << "]" << FairLogger::endl;
          eq->mWillBeSegmentMarker = false;
          eq->mWillBeRowMarker = true;
          newOne = false;
        }
      }
      if (newOne) {
        payIndex += 1;
      }
    }
    for (int i = 0; i < mPayloadTail; i++) { // move the pointer to skip the Payload Tail
      getWordFromStream(&wp);
    }
  } // this is the end of stream

  // cycle in order to update info for the last event
  for (int i = 0; i < mNumberOfEquipments; i++) {
    if (mTheEquipments[i]->mNumberOfEvents > 0) {
      updateStatistics(mTheEquipments[i]);
    }
  }
  return (true);
}

/// ---------- Read Raw Data Buffer with Fast Decoding ----------
/// Read the stream, decode the contents and store resuls.
/// Fast alghoritm : no parsing of control words !
/// ATTENTION : Assumes that the input stream was set
/// @throws TH_WRONGHEADER Thrown if the Fails to decode the Header
bool HmpidDecoder::decodeBufferFast()
{
  // ---------resets the PAdMap-----------
  for (int i = 0; i < mNumberOfEquipments; i++) {
    mTheEquipments[i]->init();
    mTheEquipments[i]->resetPadMap();
  }

  int type;
  int equipmentIndex = -1;
  int isIt;
  HmpidEquipment *eq;
  int32_t *streamBuf;
  ILOG(DEBUG) << "Enter FAST decoding !" << FairLogger::endl;

  // Input Stream Main Loop
  while (true) {
    try {
      getHeaderFromStream(&streamBuf);
    }
    catch (int e) {
      // The stream end !
      ILOG(DEBUG) << "End main decoding loop !" << FairLogger::endl;
      break;
    }
    try {
      decodeHeader(streamBuf, &equipmentIndex);
    }
    catch (int e) {
      ILOG(ERROR) << "Failed to decode the Header !" << FairLogger::endl;
      throw TH_WRONGHEADER;
    }

    eq = evaluateHeaderContents(equipmentIndex);

    int wpprev = 0;
    int wp = 0;
    int newOne = true;
    int p1, p2, p3, p4;
    int error;

    int payIndex = 0;
    while (payIndex < mNumberWordToRead) {  //start the payload loop word by word
      wpprev = wp;
      if (!getWordFromStream(&wp)) { // end the stream
        break;
      }
      if (wp == wpprev) {
        ILOG(DEBUG) << "Equip=" << mEquipment << sErrorDescription[ERR_DUPLICATEPAD] << " col=" << (eq->mSegment) * 8 + eq->mColumnCounter << \
            "[" << p1 << "]" << FairLogger::endl;
      } else {
        if( isPadWord(wp, &error, &p1, &p2, &p3, &p4) == true) {
          if( error != false) {
            setPad(eq, p1 - 1, p2 - 1, p3, p4);
            eq->mSampleNumber++;
          }
        }
      }
      payIndex += 1;
    }
    for (int i = 0; i < mPayloadTail; i++) { // move the pointer to skip the Payload Tail
      getWordFromStream(&wp);
    }
  } // this is the end of stream

  // cycle in order to update info for the last event
  for (int i = 0; i < mNumberOfEquipments; i++) {
    if (mTheEquipments[i]->mNumberOfEvents > 0) {
      updateStatistics(mTheEquipments[i]);
    }
  }
  return (true);
}



// =========================================================

/// Getter method to extract Statistic Data in Digit Coords
/// @param[in] Module : the HMPID Module number [0..6]
/// @param[in] Column : the HMPID Module Column number [0..143]
/// @param[in] Row : the HMPID Module Row number [0..159]
/// @returns The Number of entries for specified pad
uint16_t HmpidDecoder::getPadSamples(int Module, int Column, int Row)
{
  int e, c, d, h;
  Geo::Module2Equipment(Module, Column, Row, &e, &c, &d, &h);
  int EqInd = getEquipmentIndex(e);
  if (EqInd < 0)
    return (0);
  return (mTheEquipments[EqInd]->mPadSamples[c][d][h]);
}

/// Getter method to extract Statistic Data in Digit Coords
/// @param[in] Module : the HMPID Module number [0..6]
/// @param[in] Column : the HMPID Module Column number [0..143]
/// @param[in] Row : the HMPID Module Row number [0..159]
/// @returns The Sum of Charges for specified pad
double HmpidDecoder::getPadSum(int Module, int Column, int Row)
{
  int e, c, d, h;
  Geo::Module2Equipment(Module, Column, Row, &e, &c, &d, &h);
  int EqInd = getEquipmentIndex(e);
  if (EqInd < 0)
    return (0);
  return (mTheEquipments[EqInd]->mPadSum[c][d][h]);
}

/// Getter method to extract Statistic Data in Digit Coords
/// @param[in] Module : the HMPID Module number [0..6]
/// @param[in] Column : the HMPID Module Column number [0..143]
/// @param[in] Row : the HMPID Module Row number [0..159]
/// @returns The Sum of Square Charges for specified pad
double HmpidDecoder::getPadSquares(int Module, int Column, int Row)
{
  int e, c, d, h;
  Geo::Module2Equipment(Module, Column, Row, &e, &c, &d, &h);
  int EqInd = getEquipmentIndex(e);
  if (EqInd < 0)
    return (0);
  return (mTheEquipments[EqInd]->mPadSquares[c][d][h]);
}

/// Getter method to extract Statistic Data in Hardware Coords
/// @param[in] EquipmId : the HMPID EquipmentId [0..13]
/// @param[in] Column : the HMPID Module Column number [0..23]
/// @param[in] Dilogic : the HMPID Module Row number [0..9]
/// @param[in] Channel : the HMPID Module Row number [0..47]
/// @returns The Number of Entries for specified pad
uint16_t HmpidDecoder::getChannelSamples(int EquipmId, int Column, int Dilogic, int Channel)
{
  int EqInd = getEquipmentIndex(EquipmId);
  if (EqInd < 0)
    return (0);
  return (mTheEquipments[EqInd]->mPadSamples[Column][Dilogic][Channel]);
}

/// Getter method to extract Statistic Data in Hardware Coords
/// @param[in] EquipmId : the HMPID EquipmentId [0..13]
/// @param[in] Column : the HMPID Module Column number [0..23]
/// @param[in] Dilogic : the HMPID Module Row number [0..9]
/// @param[in] Channel : the HMPID Module Row number [0..47]
/// @returns The Sum of Charges for specified pad
double HmpidDecoder::getChannelSum(int EquipmId, int Column, int Dilogic, int Channel)
{
  int EqInd = getEquipmentIndex(EquipmId);
  if (EqInd < 0)
    return (0);
  return (mTheEquipments[EqInd]->mPadSum[Column][Dilogic][Channel]);
}

/// Getter method to extract Statistic Data in Hardware Coords
/// @param[in] EquipmId : the HMPID EquipmentId [0..13]
/// @param[in] Column : the HMPID Module Column number [0..23]
/// @param[in] Dilogic : the HMPID Module Row number [0..9]
/// @param[in] Channel : the HMPID Module Row number [0..47]
/// @returns The Sum of Square Charges for specified pad
double HmpidDecoder::getChannelSquare(int EquipmId, int Column, int Dilogic, int Channel)
{
  int EqInd = getEquipmentIndex(EquipmId);
  if (EqInd < 0)
    return (0);
  return (mTheEquipments[EqInd]->mPadSquares[Column][Dilogic][Channel]);
}

/// Gets the Average Event Size value
/// @param[in] EquipmId : the HMPID EquipmentId [0..13]
/// @returns The Average Event Size value ( 0 for wrong Equipment Id)
float HmpidDecoder::getAverageEventSize(int EquipmId)
{
  int EqInd = getEquipmentIndex(EquipmId);
  if (EqInd < 0)
    return (0.0);
  return (mTheEquipments[EqInd]->mEventSizeAverage);
}

/// Gets the Average Busy Time value
/// @param[in] EquipmId : the HMPID EquipmentId [0..13]
/// @returns The Average Busy Time value ( 0 for wrong Equipment Id)
float HmpidDecoder::getAverageBusyTime(int EquipmId)
{
  int EqInd = getEquipmentIndex(EquipmId);
  if (EqInd < 0)
    return (0.0);
  return (mTheEquipments[EqInd]->mBusyTimeAverage);
}

// ===================================================
// Methods to dump info


/// Prints on the standard output the table of decoding
/// errors for one equipment
/// @param[in] EquipmId : the HMPID EquipmentId [0..13]
void HmpidDecoder::dumpErrors(int EquipmId)
{
  int EqInd = getEquipmentIndex(EquipmId);
  if (EqInd < 0)
    return;

  std::cout << "Dump Errors for the Equipment = " << EquipmId << std::endl;
  for (int i = 0; i < MAXERRORS; i++) {
    std::cout << sErrorDescription[i] << "  = " << mTheEquipments[EqInd]->mErrors[i] << std::endl;
  }
  std::cout << " -------- " << std::endl;
  return;
}

/// Prints on the standard output a Table of statistical
/// decoding information for one equipment
/// @param[in] EquipmId : the HMPID EquipmentId [0..13]
/// @type[in] The type of info.  0 = Entries, 1 = Sum, 2 = Sum of squares
void HmpidDecoder::dumpPads(int EquipmId, int type)
{
  int EqInd = getEquipmentIndex(EquipmId);
  if (EqInd < 0)
    return;

  int Module = EquipmId / 2;
  int StartRow = (EquipmId % 2 == 1) ? 80 : 0;
  int EndRow = (EquipmId % 2 == 1) ? 160 : 80;
  std::cout << "Dump Pads for the Equipment = " << EquipmId << std::endl;
  for (int c = 0; c < 144; c++) {
    for (int r = StartRow; r < EndRow; r++) {
      switch (type) {
        case 0:
          std::cout << getPadSamples(Module, c, r) << ",";
          break;
        case 1:
          std::cout << getPadSum(Module, c, r) << ",";
          break;
        case 2:
          std::cout << getPadSquares(Module, c, r) << ",";
          break;
      }
    }
    std::cout << std::endl;
  }
  std::cout << " -------- " << std::endl;
  return;
}

/// Prints on the standard output the decoded HMPID error field
/// @param[in] ErrorField : the HMPID readout error field
void HmpidDecoder::dumpHmpidError(int ErrorField)
{
  char printbuf[MAXHMPIDERRORS * MAXDESCRIPTIONLENGHT];
  if (decodeHmpidError(ErrorField, printbuf) == true) {
    ILOG(ERROR) << "HMPID Error field = " << ErrorField << " : " << printbuf << FairLogger::endl;
  }
  return;
}

/// Writes in a ASCCI File the complete report of the decoding
/// procedure
/// @param[in] *summaryFileName : the name of the output file
/// @throws TH_CREATEFILE Thrown if was not able to create the file
void HmpidDecoder::writeSummaryFile(char *summaryFileName)
{
  FILE *fs = fopen(summaryFileName, "w");
  if (fs == 0) {
    printf("Error opening the file %s !\n", summaryFileName);
    throw TH_CREATEFILE;
  }

  fprintf(fs, "HMPID Readout Raw Data Decoding Summary File\n");
  fprintf(fs, "Equipment Id\t");
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
    fprintf(fs, "%d\t", mTheEquipments[i]->getEquipmentId());
  fprintf(fs, "\n");

  fprintf(fs, "Number of events\t");
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
    fprintf(fs, "%d\t", mTheEquipments[i]->mNumberOfEvents);
  fprintf(fs, "\n");

  fprintf(fs, "Average Event Size\t");
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
    fprintf(fs, "%f\t", mTheEquipments[i]->mEventSizeAverage);
  fprintf(fs, "\n");

  fprintf(fs, "Total pads\t");
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
    fprintf(fs, "%d\t", mTheEquipments[i]->mTotalPads);
  fprintf(fs, "\n");

  fprintf(fs, "Average pads per event\t");
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
    fprintf(fs, "%f\t", mTheEquipments[i]->mPadsPerEventAverage);
  fprintf(fs, "\n");

  fprintf(fs, "Busy Time average\t");
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
    fprintf(fs, "%e\t", mTheEquipments[i]->mBusyTimeAverage);
  fprintf(fs, "\n");

  fprintf(fs, "Event rate\t");
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
    fprintf(fs, "%e\t", 1 / mTheEquipments[i]->mBusyTimeAverage);
  fprintf(fs, "\n");

  fprintf(fs, "Number of Empty Events\t");
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
    fprintf(fs, "%d\t", mTheEquipments[i]->mNumberOfEmptyEvents);
  fprintf(fs, "\n");

  fprintf(fs, "-------------Errors--------------------\n");
  fprintf(fs, "Wrong events\t");
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
    fprintf(fs, "%d\t", mTheEquipments[i]->mNumberOfWrongEvents);
  fprintf(fs, "\n");

  for (int j = 0; j < MAXERRORS; j++) {
    fprintf(fs, "%s\t", sErrorDescription[j]);
    for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
      fprintf(fs, "%d\t", mTheEquipments[i]->mErrors[j]);
    fprintf(fs, "\n");
  }

  fprintf(fs, "Total errors\t");
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++)
    fprintf(fs, "%d\t", mTheEquipments[i]->mTotalErrors);
  fprintf(fs, "\n");

  fclose(fs);
  return;
}
