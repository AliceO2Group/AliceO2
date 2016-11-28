#include "DataHeader.h"
#include <cstdio> // printf
#include <cstring> // strncpy

//the answer to life and everything
const uint32_t AliceO2::Base::BaseHeader::sMagicString = CharArr2uint32("O2O2");

using namespace AliceO2::Base;

//__________________________________________________________________________________________________
//possible data origins
const DataHeader::DataOrigin AliceO2::Base::gDataOriginAny    ("***");
const DataHeader::DataOrigin AliceO2::Base::gDataOriginInvalid("   ");
const DataHeader::DataOrigin AliceO2::Base::gDataOriginTPC    ("TPC");
const DataHeader::DataOrigin AliceO2::Base::gDataOriginTRD    ("TRD");
const DataHeader::DataOrigin AliceO2::Base::gDataOriginTOF    ("TOF");

//possible data types
const DataHeader::DataDescription AliceO2::Base::gDataDescriptionAny     ("***************");
const DataHeader::DataDescription AliceO2::Base::gDataDescriptionInvalid ("               ");
const DataHeader::DataDescription AliceO2::Base::gDataDescriptionRawData ("RAWDATA        ");
const DataHeader::DataDescription AliceO2::Base::gDataDescriptionClusters("CLUSTERS       ");
const DataHeader::DataDescription AliceO2::Base::gDataDescriptionTracks  ("TRACKS         ");

//possible serialization types
const BaseHeader::SerializationMethod AliceO2::Base::gSerializationMethodAny    ("*******");
const BaseHeader::SerializationMethod AliceO2::Base::gSerializationMethodInvalid("       ");
const BaseHeader::SerializationMethod AliceO2::Base::gSerializationMethodNone   ("NONE   ");
const BaseHeader::SerializationMethod AliceO2::Base::gSerializationMethodROOT   ("ROOT   ");
const BaseHeader::SerializationMethod AliceO2::Base::gSerializationMethodFlatBuf("FLATBUF");

//__________________________________________________________________________________________________
//static version numbers
const uint32_t BaseHeader::sVersion=1;

const BaseHeader::HeaderType DataHeader::sHeaderType = String2uint64("BaseHDER");
const BaseHeader::SerializationMethod DataHeader::sSerializationMethod = AliceO2::Base::gSerializationMethodNone;

const BaseHeader::HeaderType ROOTobjectHeader::sHeaderType = "ROOTmeta";
const BaseHeader::SerializationMethod ROOTobjectHeader::sSerializationMethod = gSerializationMethodNone;

//__________________________________________________________________________________________________
AliceO2::Base::BaseHeader::BaseHeader()
  : magicStringInt(sMagicString)
  , headerSize(sizeof(BaseHeader))
  , flags(0)
  , headerVersion(sVersion)
  , description(gInvalidToken64)
  , serialization(gInvalidToken64)
{
}

//__________________________________________________________________________________________________
AliceO2::Base::BaseHeader::BaseHeader(uint32_t size, HeaderType desc, SerializationMethod ser)
  : magicStringInt(sMagicString)
  , headerSize(size)
  , flags(0)
  , headerVersion(sVersion)
  , description(desc)
  , serialization(ser)
{
}

//__________________________________________________________________________________________________
AliceO2::Base::DataHeader::DataHeader()
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
void AliceO2::Base::DataHeader::print() const
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
DataHeader& AliceO2::Base::DataHeader::operator=(const DataHeader& that)
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
DataHeader& AliceO2::Base::DataHeader::operator=(const DataOrigin& that)
{
  dataOrigin = that;
  return *this;
}

//__________________________________________________________________________________________________
DataHeader& AliceO2::Base::DataHeader::operator=(const DataDescription& that)
{
  dataDescription = that;
  return *this;
}

//__________________________________________________________________________________________________
DataHeader& AliceO2::Base::DataHeader::operator=(const SerializationMethod& that)
{
  payloadSerializationMethod = that;
  return *this;
}

//__________________________________________________________________________________________________
bool AliceO2::Base::DataHeader::operator==(const DataOrigin& that)
{
  return (that == gDataOriginAny||
          that == dataOrigin );
}

//__________________________________________________________________________________________________
bool AliceO2::Base::DataHeader::operator==(const DataDescription& that)
{
  return ((that.itg[0] == gDataDescriptionAny.itg[0] &&
	   that.itg[1] == gDataDescriptionAny.itg[1]) ||
          (that.itg[0] == dataDescription.itg[0] &&
	   that.itg[1] == dataDescription.itg[1] ));
}

//__________________________________________________________________________________________________
bool AliceO2::Base::DataHeader::operator==(const SerializationMethod& that)
{
  return (that == gSerializationMethodAny||
          that == payloadSerializationMethod );
}

//__________________________________________________________________________________________________
bool AliceO2::Base::DataHeader::operator==(const DataHeader& that)
{
  return( magicStringInt == that.magicStringInt &&
          dataOrigin == that.dataOrigin &&
          dataDescription == that.dataDescription &&
          subSpecification == that.subSpecification );
}

//__________________________________________________________________________________________________
AliceO2::Base::DataHeader::DataOrigin::DataOrigin() : itg(gInvalidToken32) {}

//__________________________________________________________________________________________________
AliceO2::Base::DataHeader::DataOrigin::DataOrigin(const char* origin)
  : itg(gInvalidToken32)
{
  if (origin) {
    strncpy(str, origin, gSizeDataOriginString-1);
  }
}

//__________________________________________________________________________________________________
bool AliceO2::Base::DataHeader::DataOrigin::operator==(const DataOrigin& other) const
{
  return itg == other.itg;
}

//__________________________________________________________________________________________________
void AliceO2::Base::DataHeader::DataOrigin::print() const
{
  printf("Data origin  : %s\n", str);
}

//__________________________________________________________________________________________________
AliceO2::Base::DataHeader::DataDescription::DataDescription()
  : itg()
{
  itg[0] = gInvalidToken64;
  itg[1] = gInvalidToken64<<8 | gInvalidToken64;
}

//__________________________________________________________________________________________________
AliceO2::Base::DataHeader::DataDescription::DataDescription(const char* desc)
  : itg()
{
  *this = DataDescription(); // initialize by standard constructor
  if (desc) {
    strncpy(str, desc, gSizeDataDescriptionString-1);
  }
}

//__________________________________________________________________________________________________
bool AliceO2::Base::DataHeader::DataDescription::operator==(const DataDescription& other) const {
  return (itg[0] == other.itg[0] &&
          itg[1] == other.itg[1]);
}

//__________________________________________________________________________________________________
void AliceO2::Base::DataHeader::DataDescription::print() const
{
  printf("Data descr.  : %s\n", str);
}

//__________________________________________________________________________________________________
AliceO2::Base::DataIdentifier::DataIdentifier()
  : dataDescription(), dataOrigin()
{
}

//__________________________________________________________________________________________________
AliceO2::Base::DataIdentifier::DataIdentifier(const char* desc, const char* origin)
  : dataDescription(), dataOrigin()
{
  dataDescription = AliceO2::Base::DataHeader::DataDescription(desc);
  dataOrigin = AliceO2::Base::DataHeader::DataOrigin(origin);
}

//__________________________________________________________________________________________________
bool AliceO2::Base::DataIdentifier::operator==(const DataIdentifier& other) const {
  if (other.dataOrigin != gDataOriginAny && dataOrigin != other.dataOrigin) return false;
  if (other.dataDescription != gDataDescriptionAny &&
      dataDescription != other.dataDescription) return false;
  return true;
}

//__________________________________________________________________________________________________
void AliceO2::Base::DataIdentifier::print() const
{
  dataOrigin.print();
  dataDescription.print();
}

