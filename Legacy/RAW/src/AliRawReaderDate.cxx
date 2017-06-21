/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
///
/// This is a class for reading raw data from a date file or event.
///
/// The AliRawReaderDate is constructed either with a pointer to a
/// date event or with a file name and an event number.
///
///////////////////////////////////////////////////////////////////////////////

#include "AliRawReaderDate.h"
#include "event.h"

ClassImp(AliRawReaderDate)


AliRawReaderDate::AliRawReaderDate(void* event, Bool_t owner) :
  fFile(NULL),
  fEvent(NULL),
  fSubEvent(NULL),
  fEquipment(NULL),
  fPosition(NULL),
  fEnd(NULL),
  fOwner(owner)
{
// create an object to read digits from the given date event

  fEvent = (eventHeaderStruct*) event;
}

AliRawReaderDate::AliRawReaderDate(const char* fileName, Int_t eventNumber) :
  fFile(NULL),
  fEvent(NULL),
  fSubEvent(NULL),
  fEquipment(NULL),
  fPosition(NULL),
  fEnd(NULL),
  fOwner(kTRUE)
{
// create an object to read digits from the given date event

  fFile = fopen(fileName, "rb");
  if (!fFile) {
    Error("AliRawReaderDate", "could not open file %s", fileName);
    fIsValid = kFALSE;
    return;
  }
  if (eventNumber < 0) return;

  eventHeaderStruct header;
  UInt_t headerSize = sizeof(eventHeaderStruct);
  while (fread(&header, 1, headerSize, fFile) == headerSize) {
    if (eventNumber == 0) {
      UChar_t* buffer = new UChar_t[header.eventSize];
      fseek(fFile, -(long)headerSize, SEEK_CUR);
      if (fread(buffer, 1, header.eventSize, fFile) != header.eventSize) break;
      fEvent = (eventHeaderStruct*) buffer;
      break;
    }
    fseek(fFile, header.eventSize-headerSize, SEEK_CUR);
    eventNumber--;
  }

}

AliRawReaderDate::~AliRawReaderDate()
{
// destructor

  if (fEvent && fOwner) delete[] fEvent;
  if (fFile) {
    fclose(fFile);
  }
}


UInt_t AliRawReaderDate::GetType() const
{
// get the type from the event header

  if (!fEvent) return 0;
  return fEvent->eventType;
}

UInt_t AliRawReaderDate::GetRunNumber() const
{
// get the run number from the event header

  if (!fEvent) return 0;
  return fEvent->eventRunNb;
}

const UInt_t* AliRawReaderDate::GetEventId() const
{
// get the event id from the event header

  if (!fEvent) return NULL;
  return fEvent->eventId;
}

const UInt_t* AliRawReaderDate::GetTriggerPattern() const
{
// get the trigger pattern from the event header

  if (!fEvent) return NULL;
  return fEvent->eventTriggerPattern;
}

const UInt_t* AliRawReaderDate::GetDetectorPattern() const
{
// get the detector pattern from the event header

  if (!fEvent) return NULL;
  return fEvent->eventDetectorPattern;
}

const UInt_t* AliRawReaderDate::GetAttributes() const
{
// get the type attributes from the event header

  if (!fEvent) return NULL;
  return fEvent->eventTypeAttribute;
}

const UInt_t* AliRawReaderDate::GetSubEventAttributes() const
{
// get the type attributes from the sub event header

  if (!fSubEvent) return NULL;
  return fSubEvent->eventTypeAttribute;
}

UInt_t AliRawReaderDate::GetLDCId() const
{
// get the LDC Id from the event header

  if (!fSubEvent) return 0;
  return fSubEvent->eventLdcId;
}

UInt_t AliRawReaderDate::GetGDCId() const
{
// get the GDC Id from the event header

  if (!fEvent) return 0;
  return fEvent->eventGdcId;
}

UInt_t AliRawReaderDate::GetTimestamp() const
{
// get the timestamp from the event header

  if (!fEvent) return 0;
  return fEvent->eventTimestamp;
}

Int_t AliRawReaderDate::GetEquipmentSize() const
{
// get the size of the equipment (including the header)

  if (!fEquipment) return 0;
  if (fSubEvent->eventVersion <= 0x00030001) {
    return fEquipment->equipmentSize + sizeof(equipmentHeaderStruct);
  } else {
    return fEquipment->equipmentSize;
  }
}

Int_t AliRawReaderDate::GetEquipmentType() const
{
// get the type from the equipment header

  if (!fEquipment) return -1;
  return fEquipment->equipmentType;
}

Int_t AliRawReaderDate::GetEquipmentId() const
{
// get the ID from the equipment header

  if (!fEquipment) return -1;
  return fEquipment->equipmentId;
}

const UInt_t* AliRawReaderDate::GetEquipmentAttributes() const
{
// get the attributes from the equipment header

  if (!fEquipment) return NULL;
  return fEquipment->equipmentTypeAttribute;
}

Int_t AliRawReaderDate::GetEquipmentElementSize() const
{
// get the basic element size from the equipment header

  if (!fEquipment) return 0;
  return fEquipment->equipmentBasicElementSize;
}

Int_t AliRawReaderDate::GetEquipmentHeaderSize() const
{
  // Get the equipment header size
  // 28 bytes by default
  return sizeof(equipmentHeaderStruct);
}

Bool_t AliRawReaderDate::ReadHeader()
{
// read a data header at the current position
// returns kFALSE if the data header could not be read

  fErrorCode = 0;

  fHeader = NULL;
  if (!fEvent) return kFALSE;
  // check whether there are sub events
  if (fEvent->eventSize <= fEvent->eventHeadSize) return kFALSE;

  do {
    // skip payload (if event was not selected)
    if (fCount > 0) fPosition += fCount;

    // get the first or the next equipment if at the end of an equipment
    if (!fEquipment || (fPosition >= fEnd)) {
      fEquipment = NULL;

      // get the first or the next sub event if at the end of a sub event
      if (!fSubEvent || 
	  (fPosition >= ((UChar_t*)fSubEvent) + fSubEvent->eventSize)) {

	// check for end of event data
	if (fPosition >= ((UChar_t*)fEvent)+fEvent->eventSize) return kFALSE;
        if (!TEST_SYSTEM_ATTRIBUTE(fEvent->eventTypeAttribute, 
                                   ATTR_SUPER_EVENT)) {
	  fSubEvent = fEvent;   // no super event
	} else if (fSubEvent) {
	  fSubEvent = (eventHeaderStruct*) (((UChar_t*)fSubEvent) + 
					    fSubEvent->eventSize);
	} else {
	  fSubEvent = (eventHeaderStruct*) (((UChar_t*)fEvent) + 
					    fEvent->eventHeadSize);
	}

	// check the magic word of the sub event
	if (fSubEvent->eventMagic != EVENT_MAGIC_NUMBER) {
	  Error("ReadHeader", "wrong magic number in sub event!\n"
		" run: %d  event: %d %d  LDC: %d  GDC: %d\n", 
		fSubEvent->eventRunNb, 
		fSubEvent->eventId[0], fSubEvent->eventId[1],
		fSubEvent->eventLdcId, fSubEvent->eventGdcId);
	  fErrorCode = kErrMagic;
	  return kFALSE;
	}

	// continue if no data in the subevent
	if (fSubEvent->eventSize == fSubEvent->eventHeadSize) {
	  fPosition = fEnd = ((UChar_t*)fSubEvent) + fSubEvent->eventSize;
	  fCount = 0;
	  continue;
	}

	fEquipment = (equipmentHeaderStruct*)
	  (((UChar_t*)fSubEvent) + fSubEvent->eventHeadSize);

      } else {
	fEquipment = (equipmentHeaderStruct*) fEnd;
      }

      fCount = 0;
      fPosition = ((UChar_t*)fEquipment) + sizeof(equipmentHeaderStruct);
      if (fSubEvent->eventVersion <= 0x00030001) {
        fEnd = fPosition + fEquipment->equipmentSize;
      } else {
        fEnd = ((UChar_t*)fEquipment) + fEquipment->equipmentSize;
      }
    }

    // continue with the next sub event if no data left in the payload
    if (fPosition >= fEnd) continue;

    if (fRequireHeader) {
      // check that there are enough bytes left for the data header
      if (fPosition + sizeof(AliRawDataHeader) > fEnd) {
	Error("ReadHeader", "could not read data header data!");
	Warning("ReadHeader", "skipping %ld bytes\n"
		" run: %d  event: %d %d  LDC: %d  GDC: %d\n", 
		fEnd - fPosition, fSubEvent->eventRunNb, 
		fSubEvent->eventId[0], fSubEvent->eventId[1],
		fSubEvent->eventLdcId, fSubEvent->eventGdcId);
	fCount = 0;
	fPosition = fEnd;
	fErrorCode = kErrNoDataHeader;
	continue;
      }

      // "read" the data header
      fHeader = (AliRawDataHeader*) fPosition;
      // Now check the version of the header
      UChar_t version = 2;
      if (fHeader) version=fHeader->GetVersion();

      switch (version) {
      case 2:
	{
	  if ((fPosition + fHeader->fSize) != fEnd) {
	    if ((fHeader->fSize != 0xFFFFFFFF) &&
		(fEquipment->equipmentId != 4352))
	      Warning("ReadHeader",
		      "raw data size found in the header is wrong (%d != %ld)! Using the equipment size instead !",
		      fHeader->fSize, fEnd - fPosition);
	    fHeader->fSize = fEnd - fPosition;
	  }
	  fPosition += sizeof(AliRawDataHeader);
	  fHeaderV3 = 0;
	  break;
	} 
      case 3:
	{
	  fHeaderV3 = (AliRawDataHeaderV3*) fPosition;
	  if ((fPosition + fHeaderV3->fSize) != fEnd) {
	    if ((fHeaderV3->fSize != 0xFFFFFFFF) &&
		(fEquipment->equipmentId != 4352))
	      Warning("ReadHeader",
		      "raw data size found in the header is wrong (%d != %ld)! Using the equipment size instead !",
		      fHeaderV3->fSize, fEnd - fPosition);
	    fHeaderV3->fSize = fEnd - fPosition;
	  }
	  fPosition += sizeof(AliRawDataHeaderV3);
	  fHeader = 0;
	  break;
	}
      default:
	// We have got a version we don't know
  		if (fEquipment->equipmentId != 4352)
		{
	Error("ReadHeader", 
	      "version is neither 2 nor 3, we can't handle it (version found : %d). Jump to the end of the equipment",version);
	Warning("ReadHeader", 
		" run: %d  event: %d %d  LDC: %d  GDC: %d\n", 
		fSubEvent->eventRunNb, 
		fSubEvent->eventId[0], fSubEvent->eventId[1],
		fSubEvent->eventLdcId, fSubEvent->eventGdcId);
		}
	fHeader = 0x0;
	fHeaderV3 = 0x0;
	fPosition = fEnd;
	continue;
      }
    }
    if (fHeader && (fHeader->fSize != 0xFFFFFFFF)) {
      fCount = fHeader->fSize - sizeof(AliRawDataHeader);

      // check consistency of data size in the header and in the sub event
      if (fPosition + fCount > fEnd) {
	Error("ReadHeader", "size in data header exceeds event size!");
	Warning("ReadHeader", "skipping %ld bytes\n"
		" run: %d  event: %d %d  LDC: %d  GDC: %d\n", 
		fEnd - fPosition, fSubEvent->eventRunNb, 
		fSubEvent->eventId[0], fSubEvent->eventId[1],
		fSubEvent->eventLdcId, fSubEvent->eventGdcId);
	fCount = 0;
	fPosition = fEnd;
	fErrorCode = kErrSize;
	continue;
      }

    } else if (fHeaderV3 && (fHeaderV3->fSize != 0xFFFFFFFF)) {
      fCount = fHeaderV3->fSize - sizeof(AliRawDataHeaderV3);

      // check consistency of data size in the header and in the sub event
      if (fPosition + fCount > fEnd) {
	Error("ReadHeader", "size in data header exceeds event size!");
	Warning("ReadHeader", "skipping %ld bytes\n"
		" run: %d  event: %d %d  LDC: %d  GDC: %d\n", 
		fEnd - fPosition, fSubEvent->eventRunNb, 
		fSubEvent->eventId[0], fSubEvent->eventId[1],
		fSubEvent->eventLdcId, fSubEvent->eventGdcId);
	fCount = 0;
	fPosition = fEnd;
	fErrorCode = kErrSize;
	continue;
      }

    } else {
      fCount = fEnd - fPosition;
    }

  } while (!fEquipment || !IsSelected());

  return kTRUE;
}

Bool_t AliRawReaderDate::ReadNextData(UChar_t*& data)
{
// reads the next payload at the current position
// returns kFALSE if the data could not be read

  fErrorCode = 0;
  while (fCount == 0) {
    if (!ReadHeader()) return kFALSE;
  }
  data = fPosition;
  fPosition += fCount;  
  fCount = 0;
  return kTRUE;
}

Bool_t AliRawReaderDate::ReadNext(UChar_t* data, Int_t size)
{
// reads the next block of data at the current position
// returns kFALSE if the data could not be read

  fErrorCode = 0;
  if (fPosition + size > fEnd) {
    Error("ReadNext", "could not read data!");
    fErrorCode = kErrOutOfBounds;
    return kFALSE;
  }
  memcpy(data, fPosition, size);
  fPosition += size;
  fCount -= size;
  return kTRUE;
}


Bool_t AliRawReaderDate::Reset()
{
// reset the current position to the beginning of the event

  fSubEvent = NULL;
  fEquipment = NULL;
  fCount = 0;
  fPosition = fEnd = NULL;
  fHeader=NULL;
  fHeaderV3=NULL;
  return kTRUE;
}


Bool_t AliRawReaderDate::NextEvent()
{
// go to the next event in the date file

  if (!fFile) {
    if (fEventNumber < 0 && fEvent) {
      fEventNumber++;
      return kTRUE;
    }
    else
      return kFALSE;
  }

  Reset();
  eventHeaderStruct header;
  UInt_t headerSize = sizeof(eventHeaderStruct);
  if (fEvent) delete[] fEvent;
  fEvent = &header;

  while (fread(&header, 1, headerSize, fFile) == headerSize) {
    if (!IsEventSelected()) {
      fseek(fFile, header.eventSize-headerSize, SEEK_CUR);
      continue;
    }
    UChar_t* buffer = new UChar_t[header.eventSize];
    fseek(fFile, -(long)headerSize, SEEK_CUR);
    if (fread(buffer, 1, header.eventSize, fFile) != header.eventSize) {
      Error("NextEvent", "could not read event from file");
      delete[] buffer;
      break;
    }
    fEvent = (eventHeaderStruct*) buffer;
    fEventNumber++;
    return kTRUE;
  };

  fEvent = NULL;

  return kFALSE;
}

Bool_t AliRawReaderDate::RewindEvents()
{
// go back to the beginning of the date file

  if (fFile)
    fseek(fFile, 0, SEEK_SET);

  fEventNumber = -1;
  return Reset();
}


Int_t AliRawReaderDate::CheckData() const
{
// check the consistency of the data

  if (!fEvent) return 0;
  // check whether there are sub events
  if (fEvent->eventSize <= fEvent->eventHeadSize) return 0;

  eventHeaderStruct* subEvent = NULL;
  UChar_t* position = 0;
  UChar_t* end = 0;
  Int_t result = 0;

  while (kTRUE) {
    // get the first or the next sub event if at the end of a sub event
    if (!subEvent || (position >= end)) {

      // check for end of event data
      if (position >= ((UChar_t*)fEvent)+fEvent->eventSize) return result;
      if (!TEST_SYSTEM_ATTRIBUTE(fEvent->eventTypeAttribute, 
                                 ATTR_SUPER_EVENT)) {
        subEvent = fEvent;   // no super event
      } else if (subEvent) {
	subEvent = (eventHeaderStruct*) (((UChar_t*)subEvent) + 
					 subEvent->eventSize);
      } else {
	subEvent = (eventHeaderStruct*) (((UChar_t*)fEvent) + 
					 fEvent->eventHeadSize);
      }

      // check the magic word of the sub event
      if (subEvent->eventMagic != EVENT_MAGIC_NUMBER) {
	result |= kErrMagic;
	return result;
      }

      position = ((UChar_t*)subEvent) + subEvent->eventHeadSize + 
	sizeof(equipmentHeaderStruct);
      end = ((UChar_t*)subEvent) + subEvent->eventSize;
    }

    // continue with the next sub event if no data left in the payload
    if (position >= end) continue;

    if (fRequireHeader) {
    // check that there are enough bytes left for the data header
      if (position + sizeof(AliRawDataHeader) > end) {
	result |= kErrNoDataHeader;
	position = end;
	continue;
      }

      // Here we have to check if we have header v2 or v3
      // check consistency of data size in the data header and in the sub event
      AliRawDataHeader* header = (AliRawDataHeader*) position;
      UChar_t version = header->GetVersion();
      if (version==2) {
	if ((position + header->fSize) != end) {
	  if (header->fSize != 0xFFFFFFFF)
	    Warning("CheckData",
		    "raw data size found in the header V2 is wrong (%d != %ld)! Using the equipment size instead !",
		    header->fSize, end - position);
	  header->fSize = end - position;
	  result |= kErrSize;
	}
      }
      else if (version==3) {
	AliRawDataHeaderV3 * headerV3 =  (AliRawDataHeaderV3*) position;
	if ((position + headerV3->fSize) != end) {
	  if (headerV3->fSize != 0xFFFFFFFF)
	    Warning("CheckData",
		    "raw data size found in the header V3 is wrong (%d != %ld)! Using the equipment size instead !",
		    headerV3->fSize, end - position);
	  headerV3->fSize = end - position;
	  result |= kErrSize;
	}
      }

    }
    position = end;
  };

  return 0;
}

AliRawReader* AliRawReaderDate::CloneSingleEvent() const
{
  // Clones the current event and
  // creates raw-reader for the cloned event
  // Can be used in order to make asynchronious
  // access to the current raw data within
  // several threads (online event display/reco)

  if (fEvent) {
    UInt_t evSize = fEvent->eventSize;
    if (evSize) {
      UChar_t *newEvent = new UChar_t[evSize];
      memcpy(newEvent,fEvent,evSize);
      return new AliRawReaderDate((void *)newEvent,kTRUE);
    }
  }
  return NULL;
}
