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
/// This is a class for reading raw data files.
///
/// The files of one event are expected to be in one directory. The name 
/// of the directory is "raw" + the event number. Each file contains
/// the raw data (with data header) of one DDL. The convention for the
/// file names is "DET_#DDL.ddl". "DET" is the name of the detector and
/// "#DDL" is the unique equipment ID.
///
/// The constructor of AliRawReaderFile takes the event number or the
/// directory name as argument.
/// 
///////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include "AliRawReaderFile.h"
#include "AliDAQ.h"
#include <TSystem.h>
#include <TArrayC.h>
#include <fstream>


using std::ios;
ClassImp(AliRawReaderFile)


AliRawReaderFile::AliRawReaderFile(Int_t eventNumber) :
  fEventIndex(eventNumber),
  fDirName("."),
  fDirectory(NULL),
  fStream(NULL),
  fEquipmentId(-1),
  fBuffer(NULL),
  fBufferSize(0),
  fEquipmentSize(0),
  fDDLIndex(NULL),
  fDDLCurrent(-1),
  fType(7),
  fRunNb(0),
  fDetectorPattern(0),
  fTimestamp(0)
{
// create an object to read digits from the given event
// in the current directory

  fDirectory = OpenDirectory();
  if (!fDirectory) fIsValid = kFALSE;
  if (!OpenNextFile()) fIsValid = kFALSE;

  fId[0] = fId[1] = 0;
  fTriggerPattern[0] = fTriggerPattern[1] = fTriggerPattern[3] = fTriggerPattern[3] = 0;
}

AliRawReaderFile::AliRawReaderFile(const char* dirName, Int_t eventNumber) :
  fEventIndex(eventNumber),
  fDirName(dirName),
  fDirectory(NULL),
  fStream(NULL),
  fEquipmentId(-1),
  fBuffer(NULL),
  fBufferSize(0),
  fEquipmentSize(0),
  fDDLIndex(NULL),
  fDDLCurrent(-1),
  fType(7),
  fRunNb(0),
  fDetectorPattern(0),
  fTimestamp(0)
{
// create an object to read digits from the given directory

  fDirectory = OpenDirectory();
  if (!fDirectory) fIsValid = kFALSE;
  if (fEventIndex >= 0 && (!OpenNextFile())) fIsValid = kFALSE;

  fId[0] = fId[1] = 0;
  fTriggerPattern[0] = fTriggerPattern[1] = fTriggerPattern[3] = fTriggerPattern[3] = 0;
}

AliRawReaderFile::~AliRawReaderFile()
{
// close the input file

  if (fDirectory) gSystem->FreeDirectory(fDirectory);
  if (fStream) {
#if defined(__HP_aCC) || defined(__DECCXX)
    if (fStream->rdbuf()->is_open()) fStream->close();
#else
    if (fStream->is_open()) fStream->close();
#endif
    delete fStream;
  }
  if (fHeader) delete fHeader;
  if (fHeaderV3) delete fHeaderV3;

  if (fBuffer) delete[] fBuffer;
  if (fDDLIndex) delete fDDLIndex; fDDLIndex=NULL;
}

void AliRawReaderFile::RequireHeader(Bool_t required)
{
  // Reading of raw data in case of missing
  // raw data header is not implemented for
  // this class
  if (!required) {
    Warning("AliRawReaderFile","Reading of raw data without raw data header!");
    if (fHeader) delete fHeader;
    if (fHeaderV3) delete fHeaderV3;
    fHeader = NULL;
    fHeaderV3 = NULL;
  }

  AliRawReader::RequireHeader(required);
}

TString AliRawReaderFile::GetDirName() const
{
// return the current directory name

  TString dirName(fDirName);
  if (fEventIndex >= 0) {
    dirName += "/raw";
    dirName += fEventIndex;
  }
  return dirName;
}

void* AliRawReaderFile::OpenDirectory()
{
// open and return the directory

  TString dirName = GetDirName();
  void* directory = gSystem->OpenDirectory(dirName);
  if (!directory) {
    Error("OpenDirectory", "could not open directory %s", dirName.Data());
  }
  return directory;
}

Bool_t AliRawReaderFile::CreateFileIndex()
{
// scan the files of the directory and create index of all DDL files
// returns kFALSE if no DDL files available
  Bool_t result=kFALSE;
  fDDLCurrent=-1;
  if (fDDLIndex) return fDDLIndex->GetSize()>0;
  if (!fDirectory) return kFALSE;
  fDDLIndex=new TArrayC(0);
  if (!fDDLIndex) return kFALSE;
  TString entry;
  while ((entry = gSystem->GetDirEntry(fDirectory))) {
    const char* filename=entry.Data();
    if (!filename || entry.IsNull()) break;
    if (entry.BeginsWith("run")) {
      entry.ReplaceAll("run","");
      fRunNb = entry.Atoi();
      continue;
    }
    if (!entry.EndsWith(".ddl")) continue;
    result=kTRUE;
    entry.Remove(0, entry.Last('_')+1);
    entry.Remove(entry.Length()-4);
    Int_t equipmentId = atoi(entry.Data());
    Int_t ddlIndex = -1;
    fDetectorPattern |= (1 << AliDAQ::DetectorIDFromDdlID(equipmentId,ddlIndex));
    if (fDDLIndex->GetSize()<=equipmentId) {
      fDDLIndex->Set(equipmentId+1);
    }
    char* array=(char*)fDDLIndex->GetArray();
    array[equipmentId]=1;
  }

  return result;
}

Bool_t AliRawReaderFile::OpenNextFile()
{
// open the next file
// returns kFALSE if the current file is the last one

  if (!fDDLIndex && !CreateFileIndex()) return kFALSE;
  if (fSelectMinEquipmentId>=0 && fSelectMinEquipmentId>fEquipmentId)
    fDDLCurrent=fSelectMinEquipmentId-1;

  if (fStream) {
#if defined(__HP_aCC) || defined(__DECCXX)
    if (fStream->rdbuf()->is_open()) fStream->close();
#else
    if (fStream->is_open()) fStream->close();
#endif
    delete fStream;
    fStream = NULL;
    fEquipmentId = -1;
    fEquipmentSize = 0;
  }

  if (!fDirectory) return kFALSE;
  while (++fDDLCurrent<(fDDLIndex->GetSize()) && 
	 (fDDLCurrent<=fSelectMaxEquipmentId || fSelectMaxEquipmentId<0)) {
    if (fDDLIndex->At(fDDLCurrent)==0) continue;
    Int_t dummy=0;
    TString entry;
    entry.Form("%s_%d.ddl", AliDAQ::DetectorNameFromDdlID(fDDLCurrent, dummy), fDDLCurrent);
    char* fileName = gSystem->ConcatFileName(GetDirName(), entry);
    if (!fileName) continue;
    // read the timestamp
    FileStat_t buf;
    if (gSystem->GetPathInfo(fileName,buf) == 0) {
      fTimestamp = buf.fMtime;
    }
#ifndef __DECCXX 
    fStream = new fstream(fileName, ios::binary|ios::in);
#else
    fStream = new fstream(fileName, ios::in);
#endif
    delete [] fileName;
    break;
  }

  if (!fStream) return kFALSE;
  fEquipmentId = fDDLCurrent;
#if defined(__HP_aCC) || defined(__DECCXX)
  return (fStream->rdbuf()->is_open());
#else
  return (fStream->is_open());
#endif
}


Bool_t AliRawReaderFile::ReadHeader()
{
// read a data header at the current stream position
// returns kFALSE if the mini header could not be read

  if (!fStream && !OpenNextFile()) return kFALSE;
  do {
    UInt_t dataSize=0;
    UInt_t headerSize=0;
    if (fCount > 0) fStream->seekg(Int_t(fStream->tellg()) + fCount);
    UInt_t currentPos = fStream->tellg();
    if (fRequireHeader) {
      if(! fHeader) fHeader = new AliRawDataHeader;
      while (!fStream->read((char*) fHeader, sizeof(AliRawDataHeader))) {
	if (!OpenNextFile()) return kFALSE;
	currentPos = fStream->tellg();
      }
      dataSize=fHeader->fSize;
      headerSize=sizeof(AliRawDataHeader);
      if(fHeader->GetVersion() == 3){
	delete fHeader;
	fHeader = NULL;
	if(! fHeaderV3) fHeaderV3 = new AliRawDataHeaderV3;
	//rewind
	fStream->seekg(currentPos);
	if (!fStream->read((char*) fHeaderV3, sizeof(AliRawDataHeaderV3)))
	  return kFALSE;
	headerSize=sizeof(AliRawDataHeaderV3);
      } else if (fHeaderV3) {
	delete fHeaderV3;
	fHeaderV3 = NULL;
      }
      
    }
    else {
      if (fStream->eof())
	if (!OpenNextFile()) return kFALSE;
    }
    if ( headerSize && dataSize != 0xFFFFFFFF) {
      fCount = dataSize - headerSize;
    } else {
      currentPos = fStream->tellg();
      fStream->seekg(0, ios::end);
      fCount = UInt_t(fStream->tellg()) - currentPos;
      fStream->seekg(currentPos);
    }
    fEquipmentSize = fCount + headerSize;
  } while (!IsSelected());
  return kTRUE;
}

Bool_t AliRawReaderFile::ReadNextData(UChar_t*& data)
{
// reads the next payload at the current stream position
// returns kFALSE if the data could not be read

  while (fCount == 0) {
    if (!ReadHeader()) return kFALSE;
  }
  if (fBufferSize < fCount) {
    if (fBuffer) delete[] fBuffer;
    fBufferSize = Int_t(fCount*1.2);
    fBuffer = new UChar_t[fBufferSize];
  }
  if (!fStream->read((char*) fBuffer, fCount)) {
    Error("ReadNext", "could not read data!");
    return kFALSE;
  }
  fCount = 0;

  data = fBuffer;
  return kTRUE;
}

Bool_t AliRawReaderFile::ReadNext(UChar_t* data, Int_t size)
{
// reads the next block of data at the current stream position
// returns kFALSE if the data could not be read

  if (!fStream->read((char*) data, size)) {
    Error("ReadNext", "could not read data!");
    return kFALSE;
  }
  fCount -= size;
  return kTRUE;
}


Bool_t AliRawReaderFile::Reset()
{
// reset the current stream position to the first DDL file of the curevent

  void* directory = OpenDirectory();
  if (!directory) return kFALSE;

  if (fStream) {
#if defined(__HP_aCC) || defined(__DECCXX)
    if (fStream->rdbuf()->is_open()) fStream->close();
#else
    if (fStream->is_open()) fStream->close();
#endif
    delete fStream;
    fStream = NULL;
  }

  if (fDirectory) gSystem->FreeDirectory(fDirectory);
  fDirectory = directory;

  // Matthias 05.06.2008
  // do not open the next file. That might collide with a subsequent
  // SelectEquipment call as the 'file pointer' is already set.
  // This is important for the indexing of the DDL files.
  // ---------------------------------------------------------
  // All ReadNext functions first require the fCount member to be
  // non zero or call ReadHeader. That allows to postpone the call
  // to OpenNextFile to the next invocation of ReadHeader.
  // ReadHeader has been mofified according to that.
  /*
  OpenNextFile();
  */
  fEquipmentId=-1;
  fDDLCurrent=-1;
  fCount = 0;
  return kTRUE;
}

Bool_t AliRawReaderFile::NextEvent()
{
// go to the next event directory

  if (fDDLIndex) delete fDDLIndex;
  fDDLIndex=NULL;
  fDetectorPattern = 0;
  if (fEventIndex < -1) return kFALSE;

  do {
    TString dirName = fDirName + "/raw";
    dirName += (fEventIndex + 1);
    void* directory = gSystem->OpenDirectory(dirName);
    if (!directory) return kFALSE;
    gSystem->FreeDirectory(directory);

    fEventIndex++;
    Reset();
  } while (!IsEventSelected());

  // Read the header of the first payload
  // in order to fill the 'fake' event header
  if (ReadHeader() && ( fHeader || fHeaderV3) ) {
    UInt_t eid1=0, eid2=0;
    ULong64_t trg1=0, trg2=0;
    if(fHeader){
      eid1=fHeader->GetEventID1();
      eid2=fHeader->GetEventID2();
      trg1=fHeader->GetTriggerClasses();
    } else {
      eid1=fHeaderV3->GetEventID1();
      eid2=fHeaderV3->GetEventID2();
      trg1=fHeaderV3->GetTriggerClasses();
      trg2=fHeaderV3->GetTriggerClassesNext50();
    }

    fId[0] = ((eid2 >> 20) & 0xf);
    fId[1] = (eid1 & 0xfff) | ((eid2 & 0xfffff) << 12);
    fTriggerPattern[0] = (trg1 & 0xffffffff);
    fTriggerPattern[1] = ((trg1 >> 32) & 0x3ffff) | ((trg2 & 0x3fff) << 18);
    fTriggerPattern[2] = ((trg2 >> 14) & 0xffffffff);
    fTriggerPattern[3] = ((trg2 >> 46) & 0xf);
  }
  else {
    Warning("AliRawReaderFile","Can not read CDH header! The event header fields will be empty!");
  }
  Reset();

  fEventNumber++;

  return kTRUE;
}

Bool_t AliRawReaderFile::RewindEvents()
{
// reset the event counter

  if (fEventIndex >= 0)  fEventIndex = -1;
  fEventNumber = -1;
  return Reset();
}
