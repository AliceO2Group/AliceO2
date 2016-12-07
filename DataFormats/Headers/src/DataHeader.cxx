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
const uint32_t BaseHeader::sVersion=1;

const BaseHeader::HeaderType DataHeader::sHeaderType = String2uint64("DataHead");
const BaseHeader::SerializationMethod DataHeader::sSerializationMethod = AliceO2::Header::gSerializationMethodNone;

const BaseHeader::HeaderType NameHeader::sHeaderType = "NameHead";
const BaseHeader::SerializationMethod NameHeader::sSerializationMethod = gSerializationMethodNone;

//__________________________________________________________________________________________________
AliceO2::Header::BaseHeader::BaseHeader()
  : magicStringInt(sMagicString)
  , headerSize(sizeof(BaseHeader))
  , flags(0)
  , headerVersion(sVersion)
  , description(gInvalidToken64)
  , serialization(gInvalidToken64)
{
}

//__________________________________________________________________________________________________
AliceO2::Header::BaseHeader::BaseHeader(uint32_t size, HeaderType desc, SerializationMethod ser)
  : magicStringInt(sMagicString)
  , headerSize(size)
  , flags(0)
  , headerVersion(sVersion)
  , description(desc)
  , serialization(ser)
{
}

//__________________________________________________________________________________________________
AliceO2::Header::DataHeader::DataHeader()
  : BaseHeader(sizeof(DataHeader),sHeaderType,sSerializationMethod)
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
bool AliceO2::Header::DataHeader::operator==(const DataOrigin& that)
{
  return (that == gDataOriginAny||
          that == dataOrigin );
}

//__________________________________________________________________________________________________
bool AliceO2::Header::DataHeader::operator==(const DataDescription& that)
{
  return ((that.itg[0] == gDataDescriptionAny.itg[0] &&
	   that.itg[1] == gDataDescriptionAny.itg[1]) ||
          (that.itg[0] == dataDescription.itg[0] &&
	   that.itg[1] == dataDescription.itg[1] ));
}

//__________________________________________________________________________________________________
bool AliceO2::Header::DataHeader::operator==(const SerializationMethod& that)
{
  return (that == gSerializationMethodAny||
          that == payloadSerializationMethod );
}

//__________________________________________________________________________________________________
bool AliceO2::Header::DataHeader::operator==(const DataHeader& that)
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

