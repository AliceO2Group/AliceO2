// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

#include "Headers/DataHeader.h"
#include <cstdio> // printf
#include <cstring> // strncpy

//the answer to life and everything
const uint32_t o2::header::BaseHeader::sMagicString = String2<uint32_t>("O2O2");

//possible serialization types
const o2::header::SerializationMethod o2::header::gSerializationMethodAny    ("*******");
const o2::header::SerializationMethod o2::header::gSerializationMethodInvalid("INVALID");
const o2::header::SerializationMethod o2::header::gSerializationMethodNone   ("NONE");
const o2::header::SerializationMethod o2::header::gSerializationMethodROOT   ("ROOT");
const o2::header::SerializationMethod o2::header::gSerializationMethodFlatBuf("FLATBUF");

//__________________________________________________________________________________________________
//possible data origins
const o2::header::DataOrigin o2::header::gDataOriginAny    ("***");
const o2::header::DataOrigin o2::header::gDataOriginInvalid("NIL");
const o2::header::DataOrigin o2::header::gDataOriginFLP    ("FLP");
const o2::header::DataOrigin o2::header::gDataOriginACO    ("ACO");
const o2::header::DataOrigin o2::header::gDataOriginCPV    ("CPV");
const o2::header::DataOrigin o2::header::gDataOriginCTP    ("CTP");
const o2::header::DataOrigin o2::header::gDataOriginEMC    ("EMC");
const o2::header::DataOrigin o2::header::gDataOriginFIT    ("FIT");
const o2::header::DataOrigin o2::header::gDataOriginHMP    ("HMP");
const o2::header::DataOrigin o2::header::gDataOriginITS    ("ITS");
const o2::header::DataOrigin o2::header::gDataOriginMCH    ("MCH");
const o2::header::DataOrigin o2::header::gDataOriginMFT    ("MFT");
const o2::header::DataOrigin o2::header::gDataOriginMID    ("MID");
const o2::header::DataOrigin o2::header::gDataOriginPHS    ("PHS");
const o2::header::DataOrigin o2::header::gDataOriginTOF    ("TOF");
const o2::header::DataOrigin o2::header::gDataOriginTPC    ("TPC");
const o2::header::DataOrigin o2::header::gDataOriginTRD    ("TRD");
const o2::header::DataOrigin o2::header::gDataOriginZDC    ("ZDC");

//possible data types
const o2::header::DataDescription o2::header::gDataDescriptionAny     ("***************");
const o2::header::DataDescription o2::header::gDataDescriptionInvalid ("INVALID_DESC");
const o2::header::DataDescription o2::header::gDataDescriptionRawData ("RAWDATA");
const o2::header::DataDescription o2::header::gDataDescriptionClusters("CLUSTERS");
const o2::header::DataDescription o2::header::gDataDescriptionTracks  ("TRACKS");
const o2::header::DataDescription o2::header::gDataDescriptionConfig  ("CONFIGURATION");
const o2::header::DataDescription o2::header::gDataDescriptionInfo    ("INFORMATION");
const o2::header::DataDescription o2::header::gDataDescriptionROOTStreamers("ROOT STREAMERS");

//definitions for Stack statics
std::default_delete<byte[]> o2::header::Stack::sDeleter;

//storage for BaseHeader static members, all invalid
const uint32_t o2::header::BaseHeader::sVersion = o2::header::gInvalidToken32;
const o2::header::HeaderType o2::header::BaseHeader::sHeaderType = o2::header::gInvalidToken64;
const o2::header::SerializationMethod o2::header::BaseHeader::sSerializationMethod = o2::header::gInvalidToken64;

//storage for DataHeader static members
const uint32_t o2::header::DataHeader::sVersion = 1;
const o2::header::HeaderType o2::header::DataHeader::sHeaderType = String2<uint64_t>("DataHead");
const o2::header::SerializationMethod o2::header::DataHeader::sSerializationMethod = o2::header::gSerializationMethodNone;

using namespace o2::header;

//__________________________________________________________________________________________________
o2::header::BaseHeader::BaseHeader(uint32_t mySize, HeaderType desc,
                                        SerializationMethod ser, uint32_t version)
  : magicStringInt(sMagicString)
  , headerSize(mySize)
  , flags(0)
  , headerVersion(version)
  , description(desc)
  , serialization(ser)
{
}

//__________________________________________________________________________________________________
o2::header::DataHeader::DataHeader()
  : BaseHeader(sizeof(DataHeader),sHeaderType,sSerializationMethod,sVersion)
  , dataDescription(gDataDescriptionInvalid)
  , dataOrigin(gDataOriginInvalid)
  , reserved(gInvalidToken32)
  , payloadSerializationMethod(gSerializationMethodInvalid)
  , subSpecification(0)
  , payloadSize(0)
{
}

//__________________________________________________________________________________________________
o2::header::DataHeader::DataHeader(DataDescription desc,
                                   DataOrigin origin,
                                   SubSpecificationType subspec,
                                   uint64_t size
                                   )
  : BaseHeader(sizeof(DataHeader),sHeaderType,sSerializationMethod,sVersion)
  , dataDescription(desc)
  , dataOrigin(origin)
  , reserved(gInvalidToken32)
  , payloadSerializationMethod(gSerializationMethodInvalid)
  , subSpecification(subspec)
  , payloadSize(size)
{
}

//__________________________________________________________________________________________________
void o2::header::DataHeader::print() const
{
  printf("Data header version %i, flags: %i\n",headerVersion, flags);
  printf("  origin       : %s\n", dataOrigin.str);
  printf("  serialization: %s\n", payloadSerializationMethod.str);
  printf("  description  : %s\n", dataDescription.str);
  printf("  sub spec.    : %llu\n", (long long unsigned int)subSpecification);
  printf("  header size  : %i\n", headerSize);
  printf("  payloadSize  : %llu\n", (long long unsigned int)payloadSize);
}

//__________________________________________________________________________________________________
DataHeader& o2::header::DataHeader::operator=(const DataHeader& that)
{
  magicStringInt = that.magicStringInt;
  dataOrigin = that.dataOrigin;
  dataDescription = that.dataDescription;
  payloadSerializationMethod= that.payloadSerializationMethod;
  subSpecification = that.subSpecification;
  flags = that.flags;
  headerVersion = that.headerVersion;
  headerSize = that.headerSize;
  payloadSize = that.payloadSize;
  return *this;
}

//__________________________________________________________________________________________________
DataHeader& o2::header::DataHeader::operator=(const DataOrigin& that)
{
  dataOrigin = that;
  return *this;
}

//__________________________________________________________________________________________________
DataHeader& o2::header::DataHeader::operator=(const DataDescription& that)
{
  dataDescription = that;
  return *this;
}

//__________________________________________________________________________________________________
DataHeader& o2::header::DataHeader::operator=(const SerializationMethod& that)
{
  payloadSerializationMethod = that;
  return *this;
}

//__________________________________________________________________________________________________
bool o2::header::DataHeader::operator==(const DataOrigin& that) const
{
  return (that == gDataOriginAny||
          that == dataOrigin );
}

//__________________________________________________________________________________________________
bool o2::header::DataHeader::operator==(const DataDescription& that) const
{
  return ((that.itg[0] == gDataDescriptionAny.itg[0] &&
           that.itg[1] == gDataDescriptionAny.itg[1]) ||
          (that.itg[0] == dataDescription.itg[0] &&
           that.itg[1] == dataDescription.itg[1] ));
}

//__________________________________________________________________________________________________
bool o2::header::DataHeader::operator==(const SerializationMethod& that) const
{
  return (that == gSerializationMethodAny||
          that == payloadSerializationMethod );
}

//__________________________________________________________________________________________________
bool o2::header::DataHeader::operator==(const DataHeader& that) const
{
  return( magicStringInt == that.magicStringInt &&
          dataOrigin == that.dataOrigin &&
          dataDescription == that.dataDescription &&
          subSpecification == that.subSpecification );
}

//__________________________________________________________________________________________________
void o2::header::printDataDescription::operator()(const char* str) const
{
  printf("Data description  : %s\n", str);
}

//__________________________________________________________________________________________________
void o2::header::printDataOrigin::operator()(const char* str) const
{
  printf("Data origin  : %s\n", str);
}

//__________________________________________________________________________________________________
o2::header::DataIdentifier::DataIdentifier()
  : dataDescription(), dataOrigin()
{
}

//__________________________________________________________________________________________________
bool o2::header::DataIdentifier::operator==(const DataIdentifier& other) const {
  if (other.dataOrigin != gDataOriginAny && dataOrigin != other.dataOrigin) return false;
  if (other.dataDescription != gDataDescriptionAny &&
      dataDescription != other.dataDescription) return false;
  return true;
}

//__________________________________________________________________________________________________
void o2::header::DataIdentifier::print() const
{
  dataOrigin.print();
  dataDescription.print();
}

//__________________________________________________________________________________________________
void o2::header::hexDump (const char* desc, const void* voidaddr, size_t len, size_t max)
{
  size_t i;
  unsigned char buff[17];       // stores the ASCII data
  memset(&buff[0],'\0',17);
  const byte* addr = reinterpret_cast<const byte*>(voidaddr);

  // Output description if given.
  if (desc != nullptr)
    printf ("%s, ", desc);
  printf("%zu bytes:", len);
  if (max>0 && len>max) {
    len = max;  //limit the output if requested
    printf(" output limited to %zu bytes\n", len);
  } else {
    printf("\n");
  }

  // In case of null pointer addr
  if (addr==nullptr) {printf("  nullptr, size: %zu\n", len); return;}

  // Process every byte in the data.
  for (i = 0; i < len; i++) {
    // Multiple of 16 means new line (with line offset).
    if ((i % 16) == 0) {
      // Just don't print ASCII for the zeroth line.
      if (i != 0)
        printf ("  %s\n", buff);

      // Output the offset.
      //printf ("  %04x ", i);
      printf ("  %p ", &addr[i]);
    }

    // Now the hex code for the specific character.
    printf (" %02x", addr[i]);

    // And store a printable ASCII character for later.
    if ((addr[i] < 0x20) || (addr[i] > 0x7e))
      buff[i % 16] = '.';
    else
      buff[i % 16] = addr[i];
    buff[(i % 16) + 1] = '\0';
    fflush(stdout);
  }

  // Pad out last line if not exactly 16 characters.
  while ((i % 16) != 0) {
    printf ("   ");
    fflush(stdout);
    i++;
  }

  // And print the final ASCII bit.
  printf ("  %s\n", buff);
  fflush(stdout);
}

