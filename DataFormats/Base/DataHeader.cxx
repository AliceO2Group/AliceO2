#include "DataHeader.h"
#include <cstdio> // printf
#include <cstring> // strncpy

const char* AliceO2::Base::BaseHeader::sMagicString = "O2 ";

using namespace AliceO2::Base;

//possible data origins
const DataOrigin AliceO2::Base::gDataOriginAny    ("***");
const DataOrigin AliceO2::Base::gDataOriginInvalid("   ");
const DataOrigin AliceO2::Base::gDataOriginTPC    ("TPC");
const DataOrigin AliceO2::Base::gDataOriginTRD    ("TRD");
const DataOrigin AliceO2::Base::gDataOriginTOF    ("TOF");

//possible data types
const DataDescription AliceO2::Base::gDataDescriptionAny     ("***************");
const DataDescription AliceO2::Base::gDataDescriptionInvalid ("               ");
const DataDescription AliceO2::Base::gDataDescriptionRawData ("RAWDATA        ");
const DataDescription AliceO2::Base::gDataDescriptionClusters("CLUSTERS       ");
const DataDescription AliceO2::Base::gDataDescriptionTracks  ("TRACKS         ");

//possible serialization types
const PayloadSerialization AliceO2::Base::gSerializationAny    ("*******");
const PayloadSerialization AliceO2::Base::gSerializationInvalid("       ");
const PayloadSerialization AliceO2::Base::gSerializationNone   ("NONE   ");
const PayloadSerialization AliceO2::Base::gSerializationROOT   ("ROOT   ");
const PayloadSerialization AliceO2::Base::gSerializationFlatBuf("FLATBUF");

////_________________________________________________________________________________________________
AliceO2::Base::BaseHeader::BaseHeader()
  : magicStringInt(*reinterpret_cast<const uint32_t*>(sMagicString))
  , headerSize(sizeof(BaseHeader))
  , flags(0)
  , headerVersion(gInvalidVersion)
  , headerDescriptionInt(gInvalidToken64)
  , headerSerializationInt(gInvalidToken64)
{
}

//_________________________________________________________________________________________________
AliceO2::Base::BaseHeader::BaseHeader(const BaseHeader& that)
  : magicStringInt(that.magicStringInt)
  , headerSize(that.headerSize)
  , flags(that.flags)
  , headerVersion(that.headerVersion)
  , headerDescriptionInt(that.headerDescriptionInt)
  , headerSerializationInt(that.headerSerializationInt)
{
}

////_________________________________________________________________________________________________
AliceO2::Base::DataHeader::DataHeader()
  : BaseHeader()
  , dataOriginInt(gDataOriginInvalid.dataOriginInt)
  , reserved(0)
  , payloadSerializationInt(gSerializationInvalid.payloadSerializationInt)
  , dataDescriptionInt{gDataDescriptionInvalid.dataDescriptionInt[0],
                       gDataDescriptionInvalid.dataDescriptionInt[1]}
  , subSpecification(0)
  , payloadSize(0)
{
}

//_________________________________________________________________________________________________
AliceO2::Base::DataHeader::DataHeader(const DataHeader& that)
  : BaseHeader(that)
  , dataOriginInt(that.dataOriginInt)
  , reserved(that.reserved)
  , payloadSerializationInt(that.payloadSerializationInt)
  , dataDescriptionInt{that.dataDescriptionInt[0], that.dataDescriptionInt[1]}
  , subSpecification(that.subSpecification)
  , payloadSize(that.payloadSize)
{
}

//_________________________________________________________________________________________________
void AliceO2::Base::DataHeader::print() const
{
  printf("Data header version %i, flags: %i\n",headerVersion, flags);
  printf("  origin       : %s\n", dataOrigin);
  printf("  serialization: %s\n", payloadSerialization);
  printf("  description  : %s\n", dataDescription);
  printf("  sub spec.    : %lu\n", subSpecification);
  printf("  header size  : %i\n", headerSize);
  printf("  payloadSize  : %li\n", payloadSize);
}

//_________________________________________________________________________________________________
DataHeader& AliceO2::Base::DataHeader::operator=(const DataHeader& that)
{
  magicStringInt = that.magicStringInt;
  dataOriginInt = that.dataOriginInt;
  dataDescriptionInt[0] = that.dataDescriptionInt[0];
  dataDescriptionInt[1] = that.dataDescriptionInt[1];
  payloadSerializationInt = that.payloadSerializationInt;
  subSpecification = that.subSpecification;
  flags = that.flags;
  headerVersion = that.headerVersion;
  headerSize = that.headerSize;
  payloadSize = that.payloadSize;
  return *this;
}

//_________________________________________________________________________________________________
DataHeader& AliceO2::Base::DataHeader::operator=(const DataOrigin& that)
{
  dataOriginInt = that.dataOriginInt;
  return *this;
}

//_________________________________________________________________________________________________
DataHeader& AliceO2::Base::DataHeader::operator=(const DataDescription& that)
{
  dataDescriptionInt[0] = that.dataDescriptionInt[0];
  dataDescriptionInt[1] = that.dataDescriptionInt[1];
  return *this;
}

//_________________________________________________________________________________________________
DataHeader& AliceO2::Base::DataHeader::operator=(const PayloadSerialization& that)
{
  payloadSerializationInt = that.payloadSerializationInt;
  return *this;
}

//_________________________________________________________________________________________________
bool AliceO2::Base::DataHeader::operator==(const DataOrigin& that)
{
  return (that.dataOriginInt == gDataOriginAny.dataOriginInt ||
          that.dataOriginInt == dataOriginInt );
}

//_________________________________________________________________________________________________
bool AliceO2::Base::DataHeader::operator==(const DataDescription& that)
{
  return ((that.dataDescriptionInt[0] == gDataDescriptionAny.dataDescriptionInt[0] &&
	   that.dataDescriptionInt[1] == gDataDescriptionAny.dataDescriptionInt[1]) ||
          (that.dataDescriptionInt[0] == dataDescriptionInt[0] &&
	   that.dataDescriptionInt[1] == dataDescriptionInt[1] ));
}

//_________________________________________________________________________________________________
bool AliceO2::Base::DataHeader::operator==(const PayloadSerialization& that)
{
  return (that.payloadSerializationInt == gSerializationAny.payloadSerializationInt ||
          that.payloadSerializationInt == payloadSerializationInt );
}

//_________________________________________________________________________________________________
bool AliceO2::Base::DataHeader::operator==(const DataHeader& that)
{
  return( magicStringInt == that.magicStringInt &&
          dataOriginInt == that.dataOriginInt &&
          dataDescriptionInt == that.dataDescriptionInt &&
          subSpecification == that.subSpecification );
}

//_________________________________________________________________________________________________
AliceO2::Base::DataOrigin::DataOrigin() : dataOriginInt(gInvalidToken32) {}

//_________________________________________________________________________________________________
AliceO2::Base::DataOrigin::DataOrigin(const char* origin)
  : dataOriginInt(gInvalidToken32)
{
  if (origin) {
    strncpy(dataOrigin, origin, gSizeDataOriginString-1);
  }
}

//_________________________________________________________________________________________________
bool AliceO2::Base::DataOrigin::operator==(const AliceO2::Base::DataOrigin& other) const
{
  return dataOriginInt == other.dataOriginInt;
}

//_________________________________________________________________________________________________
void AliceO2::Base::DataOrigin::print() const
{
  printf("Data origin  : %s\n", dataOrigin);
}

//_________________________________________________________________________________________________
AliceO2::Base::DataDescription::DataDescription()
  : dataDescriptionInt()
{
  dataDescriptionInt[0] = gInvalidToken64;
  dataDescriptionInt[1] = gInvalidToken64<<8 | gInvalidToken64;
}

//_________________________________________________________________________________________________
AliceO2::Base::DataDescription::DataDescription(const char* desc)
  : dataDescription()
{
  *this = DataDescription(); // initialize by standard constructor
  if (desc) {
    strncpy(dataDescription, desc, gSizeDataDescriptionString-1);
  }
}

//_________________________________________________________________________________________________
bool AliceO2::Base::DataDescription::operator==(const AliceO2::Base::DataDescription& other) const {
  return (dataDescriptionInt[0] == other.dataDescriptionInt[0] &&
          dataDescriptionInt[1] == other.dataDescriptionInt[1]);
}

//_________________________________________________________________________________________________
void AliceO2::Base::DataDescription::print() const
{
  printf("Data descr.  : %s\n", dataDescription);
}

//_________________________________________________________________________________________________
AliceO2::Base::DataIdentifier::DataIdentifier()
  : dataDescription(), dataOrigin()
{
}

//_________________________________________________________________________________________________
AliceO2::Base::DataIdentifier::DataIdentifier(const char* desc, const char* origin)
  : dataDescription(), dataOrigin()
{
  dataDescription = AliceO2::Base::DataDescription(desc);
  dataOrigin = AliceO2::Base::DataOrigin(origin);
}

//_________________________________________________________________________________________________
bool AliceO2::Base::DataIdentifier::operator==(const AliceO2::Base::DataIdentifier& other) const {
  if (other.dataOrigin != gDataOriginAny && dataOrigin != other.dataOrigin) return false;
  if (other.dataDescription != gDataDescriptionAny && dataDescription != other.dataDescription) return false;
  return true;
}

//_________________________________________________________________________________________________
void AliceO2::Base::DataIdentifier::print() const
{
  dataOrigin.print();
  dataDescription.print();
}

//_________________________________________________________________________________________________
AliceO2::Base::PayloadSerialization::PayloadSerialization() : payloadSerializationInt(gInvalidToken64) {}

//_________________________________________________________________________________________________
AliceO2::Base::PayloadSerialization::PayloadSerialization(const char* serialization)
  : payloadSerializationInt(gInvalidToken32)
{
  if (serialization) {
    strncpy(payloadSerialization, serialization, gSizeSerializationString-1);
  }
}

//_________________________________________________________________________________________________
bool AliceO2::Base::PayloadSerialization::operator==(const AliceO2::Base::PayloadSerialization& other) const {
  return payloadSerializationInt == other.payloadSerializationInt;
}

//_________________________________________________________________________________________________
void AliceO2::Base::PayloadSerialization::print() const
{
  printf("Serialization: %s\n", payloadSerialization);
}
