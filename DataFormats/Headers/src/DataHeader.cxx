#include "Headers/DataHeader.h"
#include <cstdio> // printf
#include <cstring> // strncpy

//the answer to life and everything
const uint32_t AliceO2::Header::BaseHeader::sMagicString = CharArr2uint32("O2O2");

using namespace AliceO2::Header;

//__________________________________________________________________________________________________
//possible data origins
const DataHeader::DataOrigin AliceO2::Header::gDataOriginAny    ("***");
const DataHeader::DataOrigin AliceO2::Header::gDataOriginInvalid("   ");
const DataHeader::DataOrigin AliceO2::Header::gDataOriginTPC    ("TPC");
const DataHeader::DataOrigin AliceO2::Header::gDataOriginTRD    ("TRD");
const DataHeader::DataOrigin AliceO2::Header::gDataOriginTOF    ("TOF");

//possible data types
const DataHeader::DataDescription AliceO2::Header::gDataDescriptionAny     ("***************");
const DataHeader::DataDescription AliceO2::Header::gDataDescriptionInvalid ("               ");
const DataHeader::DataDescription AliceO2::Header::gDataDescriptionRawData ("RAWDATA        ");
const DataHeader::DataDescription AliceO2::Header::gDataDescriptionClusters("CLUSTERS       ");
const DataHeader::DataDescription AliceO2::Header::gDataDescriptionTracks  ("TRACKS         ");
const DataHeader::DataDescription AliceO2::Header::gDataDescriptionConfig  ("CONFIG         ");
const DataHeader::DataDescription AliceO2::Header::gDataDescriptionInfo    ("INFO           ");

//possible serialization types
const BaseHeader::SerializationMethod AliceO2::Header::gSerializationMethodAny    ("*******");
const BaseHeader::SerializationMethod AliceO2::Header::gSerializationMethodInvalid("       ");
const BaseHeader::SerializationMethod AliceO2::Header::gSerializationMethodNone   ("NONE   ");
const BaseHeader::SerializationMethod AliceO2::Header::gSerializationMethodROOT   ("ROOT   ");
const BaseHeader::SerializationMethod AliceO2::Header::gSerializationMethodFlatBuf("FLATBUF");

//__________________________________________________________________________________________________
//static version numbers
const uint32_t BaseHeader::sVersion=gInvalidToken32;
const BaseHeader::HeaderType BaseHeader::sHeaderType = gInvalidToken64;
const BaseHeader::SerializationMethod BaseHeader::sSerializationMethod = gInvalidToken64;

const uint32_t DataHeader::sVersion=1;
const BaseHeader::HeaderType DataHeader::sHeaderType = String2uint64("DataHead");
const BaseHeader::SerializationMethod DataHeader::sSerializationMethod = AliceO2::Header::gSerializationMethodNone;

//__________________________________________________________________________________________________
AliceO2::Header::BaseHeader::BaseHeader()
  : magicStringInt(sMagicString)
  , headerSize(sizeof(BaseHeader))
  , flags(0)
  , headerVersion(gInvalidToken32)
  , description(gInvalidToken64)
  , serialization(gInvalidToken64)
{
}

//__________________________________________________________________________________________________
AliceO2::Header::BaseHeader::BaseHeader(uint32_t mySize, HeaderType desc,
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
AliceO2::Header::DataHeader::DataHeader()
  : BaseHeader(sizeof(DataHeader),sHeaderType,sSerializationMethod,sVersion)
  , dataOrigin(gDataOriginInvalid)
  , reserved(0)
  , payloadSerializationMethod(gSerializationMethodInvalid)
  , dataDescription(gDataDescriptionInvalid)
  , subSpecification(0)
  , payloadSize(0)
{
}

//__________________________________________________________________________________________________
void AliceO2::Header::DataHeader::print() const
{
  printf("Data header version %i, flags: %i\n",headerVersion, flags);
  printf("  origin       : %s\n", dataOrigin.str);
  printf("  serialization: %s\n", payloadSerializationMethod.str);
  printf("  description  : %s\n", dataDescription.str);
  printf("  sub spec.    : %llu\n", subSpecification);
  printf("  header size  : %i\n", headerSize);
  printf("  payloadSize  : %llu\n", payloadSize);
}

//__________________________________________________________________________________________________
DataHeader& AliceO2::Header::DataHeader::operator=(const DataHeader& that)
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
DataHeader& AliceO2::Header::DataHeader::operator=(const DataOrigin& that)
{
  dataOrigin = that;
  return *this;
}

//__________________________________________________________________________________________________
DataHeader& AliceO2::Header::DataHeader::operator=(const DataDescription& that)
{
  dataDescription = that;
  return *this;
}

//__________________________________________________________________________________________________
DataHeader& AliceO2::Header::DataHeader::operator=(const SerializationMethod& that)
{
  payloadSerializationMethod = that;
  return *this;
}

//__________________________________________________________________________________________________
bool AliceO2::Header::DataHeader::operator==(const DataOrigin& that) const
{
  return (that == gDataOriginAny||
          that == dataOrigin );
}

//__________________________________________________________________________________________________
bool AliceO2::Header::DataHeader::operator==(const DataDescription& that) const
{
  return ((that.itg[0] == gDataDescriptionAny.itg[0] &&
	   that.itg[1] == gDataDescriptionAny.itg[1]) ||
          (that.itg[0] == dataDescription.itg[0] &&
	   that.itg[1] == dataDescription.itg[1] ));
}

//__________________________________________________________________________________________________
bool AliceO2::Header::DataHeader::operator==(const SerializationMethod& that) const
{
  return (that == gSerializationMethodAny||
          that == payloadSerializationMethod );
}

//__________________________________________________________________________________________________
bool AliceO2::Header::DataHeader::operator==(const DataHeader& that) const
{
  return( magicStringInt == that.magicStringInt &&
          dataOrigin == that.dataOrigin &&
          dataDescription == that.dataDescription &&
          subSpecification == that.subSpecification );
}

//__________________________________________________________________________________________________
AliceO2::Header::DataHeader::DataOrigin::DataOrigin() : itg(gInvalidToken32) {}

//__________________________________________________________________________________________________
AliceO2::Header::DataHeader::DataOrigin::DataOrigin(const char* origin)
  : itg(gInvalidToken32)
{
  if (origin) {
    strncpy(str, origin, gSizeDataOriginString-1);
  }
}

//__________________________________________________________________________________________________
bool AliceO2::Header::DataHeader::DataOrigin::operator==(const DataOrigin& other) const
{
  return itg == other.itg;
}

//__________________________________________________________________________________________________
void AliceO2::Header::DataHeader::DataOrigin::print() const
{
  printf("Data origin  : %s\n", str);
}

//__________________________________________________________________________________________________
AliceO2::Header::DataHeader::DataDescription::DataDescription()
  : itg()
{
  itg[0] = gInvalidToken64;
  itg[1] = gInvalidToken64<<8 | gInvalidToken64;
}

//__________________________________________________________________________________________________
AliceO2::Header::DataHeader::DataDescription::DataDescription(const char* desc)
  : itg()
{
  *this = DataDescription(); // initialize by standard constructor
  if (desc) {
    strncpy(str, desc, gSizeDataDescriptionString-1);
  }
}

//__________________________________________________________________________________________________
bool AliceO2::Header::DataHeader::DataDescription::operator==(const DataDescription& other) const {
  return (itg[0] == other.itg[0] &&
          itg[1] == other.itg[1]);
}

//__________________________________________________________________________________________________
void AliceO2::Header::DataHeader::DataDescription::print() const
{
  printf("Data descr.  : %s\n", str);
}

//__________________________________________________________________________________________________
AliceO2::Header::DataIdentifier::DataIdentifier()
  : dataDescription(), dataOrigin()
{
}

//__________________________________________________________________________________________________
AliceO2::Header::DataIdentifier::DataIdentifier(const char* desc, const char* origin)
  : dataDescription(), dataOrigin()
{
  dataDescription = AliceO2::Header::DataHeader::DataDescription(desc);
  dataOrigin = AliceO2::Header::DataHeader::DataOrigin(origin);
}

//__________________________________________________________________________________________________
bool AliceO2::Header::DataIdentifier::operator==(const DataIdentifier& other) const {
  if (other.dataOrigin != gDataOriginAny && dataOrigin != other.dataOrigin) return false;
  if (other.dataDescription != gDataDescriptionAny &&
      dataDescription != other.dataDescription) return false;
  return true;
}

//__________________________________________________________________________________________________
void AliceO2::Header::DataIdentifier::print() const
{
  dataOrigin.print();
  dataDescription.print();
}

//__________________________________________________________________________________________________
void AliceO2::Header::hexDump (const char* desc, const void* voidaddr, size_t len, size_t max)
{
  size_t i;
  unsigned char buff[17];       // stores the ASCII data
  memset(&buff[0],'\0',17);
  const byte* addr = reinterpret_cast<const byte*>(voidaddr);

  // Output description if given.
  if (desc != NULL)
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

