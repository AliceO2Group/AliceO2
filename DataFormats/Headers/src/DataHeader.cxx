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
const uint32_t o2::Header::BaseHeader::sMagicString = String2<uint32_t>("O2O2");

//possible serialization types
const o2::Header::SerializationMethod o2::Header::gSerializationMethodAny    ("*******");
const o2::Header::SerializationMethod o2::Header::gSerializationMethodInvalid("       ");
const o2::Header::SerializationMethod o2::Header::gSerializationMethodNone   ("NONE   ");
const o2::Header::SerializationMethod o2::Header::gSerializationMethodROOT   ("ROOT   ");
const o2::Header::SerializationMethod o2::Header::gSerializationMethodFlatBuf("FLATBUF");

//__________________________________________________________________________________________________
//possible data origins
const o2::Header::DataOrigin o2::Header::gDataOriginAny    ("***");
const o2::Header::DataOrigin o2::Header::gDataOriginInvalid("   ");
const o2::Header::DataOrigin o2::Header::gDataOriginTPC    ("TPC");
const o2::Header::DataOrigin o2::Header::gDataOriginTRD    ("TRD");
const o2::Header::DataOrigin o2::Header::gDataOriginTOF    ("TOF");

//possible data types
const o2::Header::DataDescription o2::Header::gDataDescriptionAny     ("***************");
const o2::Header::DataDescription o2::Header::gDataDescriptionInvalid ("               ");
const o2::Header::DataDescription o2::Header::gDataDescriptionRawData ("RAWDATA        ");
const o2::Header::DataDescription o2::Header::gDataDescriptionClusters("CLUSTERS       ");
const o2::Header::DataDescription o2::Header::gDataDescriptionTracks  ("TRACKS         ");
const o2::Header::DataDescription o2::Header::gDataDescriptionConfig  ("CONFIG         ");
const o2::Header::DataDescription o2::Header::gDataDescriptionInfo    ("INFO           ");

//definitions for Stack statics
std::default_delete<byte[]> o2::Header::Stack::sDeleter;

//storage for BaseHeader static members, all invalid
const uint32_t o2::Header::BaseHeader::sVersion = o2::Header::gInvalidToken32;
const o2::Header::HeaderType o2::Header::BaseHeader::sHeaderType = o2::Header::gInvalidToken64;
const o2::Header::SerializationMethod o2::Header::BaseHeader::sSerializationMethod = o2::Header::gInvalidToken64;

//storage for DataHeader static members
const uint32_t o2::Header::DataHeader::sVersion = 1;
const o2::Header::HeaderType o2::Header::DataHeader::sHeaderType = String2<uint64_t>("DataHead");
const o2::Header::SerializationMethod o2::Header::DataHeader::sSerializationMethod = o2::Header::gSerializationMethodNone;

//storage fr NameHeader static
template <>
const o2::Header::HeaderType o2::Header::NameHeader<0>::sHeaderType = "NameHead";

using namespace o2::Header;

//__________________________________________________________________________________________________
o2::Header::BaseHeader::BaseHeader()
  : magicStringInt(sMagicString)
  , headerSize(sizeof(BaseHeader))
  , flags(0)
  , headerVersion(gInvalidToken32)
  , description(gInvalidToken64)
  , serialization(gInvalidToken64)
{
}

//__________________________________________________________________________________________________
o2::Header::BaseHeader::BaseHeader(uint32_t mySize, HeaderType desc,
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
o2::Header::DataHeader::DataHeader()
  : BaseHeader(sizeof(DataHeader),sHeaderType,sSerializationMethod,sVersion)
  , dataOrigin(gDataOriginInvalid)
  , reserved(gInvalidToken32)
  , payloadSerializationMethod(gSerializationMethodInvalid)
  , dataDescription(gDataDescriptionInvalid)
  , subSpecification(0)
  , payloadSize(0)
{
}

//__________________________________________________________________________________________________
void o2::Header::DataHeader::print() const
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
DataHeader& o2::Header::DataHeader::operator=(const DataHeader& that)
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
DataHeader& o2::Header::DataHeader::operator=(const DataOrigin& that)
{
  dataOrigin = that;
  return *this;
}

//__________________________________________________________________________________________________
DataHeader& o2::Header::DataHeader::operator=(const DataDescription& that)
{
  dataDescription = that;
  return *this;
}

//__________________________________________________________________________________________________
DataHeader& o2::Header::DataHeader::operator=(const SerializationMethod& that)
{
  payloadSerializationMethod = that;
  return *this;
}

//__________________________________________________________________________________________________
bool o2::Header::DataHeader::operator==(const DataOrigin& that) const
{
  return (that == gDataOriginAny||
          that == dataOrigin );
}

//__________________________________________________________________________________________________
bool o2::Header::DataHeader::operator==(const DataDescription& that) const
{
  return ((that.itg[0] == gDataDescriptionAny.itg[0] &&
	   that.itg[1] == gDataDescriptionAny.itg[1]) ||
          (that.itg[0] == dataDescription.itg[0] &&
	   that.itg[1] == dataDescription.itg[1] ));
}

//__________________________________________________________________________________________________
bool o2::Header::DataHeader::operator==(const SerializationMethod& that) const
{
  return (that == gSerializationMethodAny||
          that == payloadSerializationMethod );
}

//__________________________________________________________________________________________________
bool o2::Header::DataHeader::operator==(const DataHeader& that) const
{
  return( magicStringInt == that.magicStringInt &&
          dataOrigin == that.dataOrigin &&
          dataDescription == that.dataDescription &&
          subSpecification == that.subSpecification );
}

//__________________________________________________________________________________________________
void o2::Header::printDataOrigin::operator()(const char* str) const
{
  printf("Data origin  : %s\n", str);
}

//__________________________________________________________________________________________________
o2::Header::DataDescription::DataDescription()
  : itg()
{
  itg[0] = gInvalidToken64;
  itg[1] = gInvalidToken64<<8 | gInvalidToken64;
}

//__________________________________________________________________________________________________
bool o2::Header::DataDescription::operator==(const DataDescription& other) const {
  return (itg[0] == other.itg[0] &&
          itg[1] == other.itg[1]);
}

//__________________________________________________________________________________________________
void o2::Header::DataDescription::print() const
{
  printf("Data descr.  : %s\n", str);
}

//__________________________________________________________________________________________________
o2::Header::DataIdentifier::DataIdentifier()
  : dataDescription(), dataOrigin()
{
}

//__________________________________________________________________________________________________
bool o2::Header::DataIdentifier::operator==(const DataIdentifier& other) const {
  if (other.dataOrigin != gDataOriginAny && dataOrigin != other.dataOrigin) return false;
  if (other.dataDescription != gDataDescriptionAny &&
      dataDescription != other.dataDescription) return false;
  return true;
}

//__________________________________________________________________________________________________
void o2::Header::DataIdentifier::print() const
{
  dataOrigin.print();
  dataDescription.print();
}

//__________________________________________________________________________________________________
void o2::Header::hexDump (const char* desc, const void* voidaddr, size_t len, size_t max)
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
  }

  // Pad out last line if not exactly 16 characters.
  while ((i % 16) != 0) {
    printf ("   ");
    i++;
  }

  // And print the final ASCII bit.
  printf ("  %s\n", buff);
}

