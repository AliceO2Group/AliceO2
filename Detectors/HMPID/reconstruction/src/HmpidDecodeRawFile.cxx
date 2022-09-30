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
/// \file   HmpidDecodeRawFile.cxx
/// \author Antonio Franco - INFN Bari
/// \brief Derived Class for decoding Raw Data File stream
/// \version 1.0
/// \date 24 set 2020

/* ------ HISTORY ---------
*/
#include <fairlogger/Logger.h> // for LOG
#include "Framework/Logger.h"

#include "HMPIDReconstruction/HmpidDecodeRawFile.h"

using namespace o2::hmpid;

/// Constructor with the default HMPID equipments map at P2
/// @param[in] numOfEquipments : number of defined equipments [0..13]
HmpidDecodeRawFile::HmpidDecodeRawFile(int numOfEquipments)
  : HmpidDecoder(numOfEquipments)
{
  fh = 0;
}

/// Constructor with the HMPID address map
/// @param[in] numOfEquipments : the number of equipments to define [1..14]
/// @param[in] *EqIds : the pointer to the Equipments ID array
/// @param[in] *CruIds : the pointer to the CRU ID array
/// @param[in] *LinkIds : the pointer to the Link ID array
HmpidDecodeRawFile::HmpidDecodeRawFile(int* EqIds, int* CruIds, int* LinkIds, int numOfEquipments)
  : HmpidDecoder(EqIds, CruIds, LinkIds, numOfEquipments)
{
  fh = 0;
}

/// Destructor
HmpidDecodeRawFile::~HmpidDecodeRawFile()
{
}

/// Setup the Input Stream with a File Handle
/// verify the existence and try to open it
/// @param[in] *FileName : the string that contains the File Name
/// @param[in] Size : not used
/// @returns True if the file is opened
/// @throws TH_FILENOTEXISTS Thrown if the file doesn't exists
/// @throws TH_OPENFILE Thrown if Fails to open the file
bool HmpidDecodeRawFile::setUpStream(void* FileName, long Size)
{
  strcpy(mInputFile, (const char*)FileName);
  // files section ----
  if (!fileExists(mInputFile)) {
    LOG(error) << "The input file " << mInputFile << " does not exist at this time.";
    throw TH_FILENOTEXISTS;
  }
  // open the file
  fh = fopen(mInputFile, "rb");
  if (fh == 0) {
    LOG(error) << "ERROR to open Input file ! [" << mInputFile << "]";
    throw TH_OPENFILE;
  }

  mActualStreamPtr = 0; // sets the pointer to the Buffer
  mEndStreamPtr = 0;    //sets the End of buffer
  mStartStreamPtr = 0;

  return (true);
}

/// Gets a sized chunk from the stream. Read from the file and update the pointers
/// ATTENTION : in order to optimize the disk accesses the block read pre-load a
/// complete Header+Payload block, the Size parameter is recalculated with the
/// dimension of the pack extract from the header field 'Offeset'
///
/// verify the existence and try to open it
/// @param[in] **streamPtr : the pointer to the memory buffer
/// @param[in] Size : not used
/// @returns True if the file is opened
/// @throws TH_WRONGFILELEN Thrown if the file doesn't contains enough words
bool HmpidDecodeRawFile::getBlockFromStream(uint32_t** streamPtr, uint32_t Size)
{
  if (Size > MAXRAWFILEBUFFER)
    return (false);
  int nr = fread(mFileBuffer, sizeof(int32_t), HEADERDIMENSION_W, fh);
  if (nr != HEADERDIMENSION_W) {
    throw TH_WRONGFILELEN;
  }
  Size = ((mFileBuffer[2] & 0x0000FFFF) / sizeof(int32_t)) - HEADERDIMENSION_W;
  nr = fread(mFileBuffer + HEADERDIMENSION_W, sizeof(int32_t), Size, fh);
  LOG(debug) << " getBlockFromStream read " << nr << " of " << Size + HEADERDIMENSION_W << " words !";
  if (nr != Size) {
    throw TH_WRONGFILELEN;
  }
  *streamPtr = mFileBuffer;
  mStartStreamPtr = mFileBuffer;
  mActualStreamPtr = mFileBuffer;
  mEndStreamPtr = mFileBuffer + Size;
  return (true);
}

/// Reads the Header from the file
/// @param[in] **streamPtr : the pointer to the memory buffer
/// @returns True if the header is read
bool HmpidDecodeRawFile::getHeaderFromStream(uint32_t** streamPtr)
{
  bool flag = getBlockFromStream(streamPtr, RAWBLOCKDIMENSION_W); // reads the 8k block
  mActualStreamPtr += HEADERDIMENSION_W;                          // Move forward for the first word
  return (flag);
}

/// Read one word from the pre-load buffer
/// @param[in] *word : the buffer for the read word
/// @returns True every time
bool HmpidDecodeRawFile::getWordFromStream(uint32_t* word)
{
  *word = *mActualStreamPtr;
  mActualStreamPtr++;
  return (true);
}

/// -----   Sets the Pad ! ------
/// this is an overloaded method. In this version the value of the charge
/// is used to update the statistical matrix of the base class
///
/// @param[in] *eq : the pointer to the Equipment object
/// @param[in] col : the column [0..23]
/// @param[in] dil : the dilogic [0..9]
/// @param[in] ch : the channel [0..47]
/// @param[in] charge : the value of the charge
void HmpidDecodeRawFile::setPad(HmpidEquipment* eq, int col, int dil, int ch, uint16_t charge)
{
  eq->setPad(col, dil, ch, charge);
  return;
}

/// Checks if the file exists !
/// @param[in] *filewithpath : the File Name to check
/// @returns True if the file exists
int HmpidDecodeRawFile::fileExists(char* filewithpath)
{
  if (access(filewithpath, F_OK) != -1) {
    return (true);
  } else {
    return (false);
  }
}
o2::hmpid::Digit
