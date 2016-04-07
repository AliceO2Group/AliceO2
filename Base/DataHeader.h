//The main O2 data header, v0.1
//origin: CWG4
//blame: Mikolaj Krzewicki, mkrzewic@cern.ch
//       Matthias Richter, Matthias.Richter@cern.ch
//       David Rohr, drohr@cern.ch

#ifndef ALICEO2_BASE_DATA_HEADER_
#define ALICEO2_BASE_DATA_HEADER_

#include <cstdint>

namespace AliceO2 {
namespace Base {

//____________________________________________________________________________
/// @defgroup aliceo2_dataformat_primitives Primitive data format definitions for ALICE O2
/// @brief This module collects information about all primitive data formats.
///
/// More to come

//____________________________________________________________________________
/// @defgroup aliceo2_dataformats_dataheader The Data Header
/// @brief A descriptive information for payload blocks
///
/// The format consists of header information and payload. The latter is not touched by the
/// framework. Each payload is described by the information in the header, for transport
/// a sequence of separate parts (header and payload in separate parts) is sent. The header
/// is described by structure \ref DataHeader
///
/// @ingroup aliceo2_dataformat_primitives

//____________________________________________________________________________
/// @defgroup dataheader_defines Length defines for DataHeader members
/// The header uses char fields for several members. This allows to define self
/// consistent unique identifiers. The identifiers are human readable in memory
/// and, rather than enumerators, independent of software versions. The string
/// is always zero terminated.
///
/// This section defines constant field lengths for char fields
/// @ingroup aliceo2_dataformats_dataheader

/// size of the magic string field @ingroup dataheader_defines
const uint32_t gSizeMagicString = 4;
/// size of the data origin field @ingroup dataheader_defines
const uint32_t gSizeDataOriginString = 4;
/// size of the payload serialization field @ingroup dataheader_defines
const uint32_t gSizeSerializationString = 8;
/// size of the data description field @ingroup dataheader_defines
const uint32_t gSizeDataDescriptionString = 16;
/// size of the header description field @ingroup dataheader_defines
const uint32_t gSizeHeaderDescriptionString = 8;
/// @}

//____________________________________________________________________________
struct DataHeader;
struct DataOrigin;
struct DataDescription;
struct DataIdentifier;
struct PayloadSerialization;

//____________________________________________________________________________
/// @struct BaseHeader
/// @brief the base header struct
///
/// @ingroup aliceo2_dataformats_dataheader
struct BaseHeader
{
  //other constants
  static const uint32_t sVersion = 1;
  static const char* sMagicString;

  //__the data layout:
  
  /// a magic string
  union {
    char     magicString[gSizeMagicString];
    uint32_t  magicStringInt;
  };

  /// size of this header (base + derived header)
  uint32_t    headerSize;

  /// flags, first bit indicates that a sub header follows
  uint32_t    flags;

  /// version of this header
  uint32_t    headerVersion;

  /// header contents description
  union {
    char headerDescription[gSizeHeaderDescriptionString];
    uint64_t headerDescriptionInt;
  };

  /// header serialization method
  union {
    char     headerSerialization[gSizeSerializationString];
    uint64_t  headerSerializationInt;
  };

  //___the functions:
  BaseHeader(); //ctor
  BaseHeader(const BaseHeader&); //copy ctor
};

//____________________________________________________________________________
/// @struct DataHeader
/// @brief the main header struct
///
/// @ingroup aliceo2_dataformats_dataheader
struct DataHeader : BaseHeader
{
  /// origin of the data (originating detector)
  union {
    char     dataOrigin[gSizeDataOriginString];
    uint32_t  dataOriginInt;
  };

  // need something for alignment, is there another field needed?
  uint32_t reserved;

  /// serialization method
  union {
    char     payloadSerialization[gSizeSerializationString];
    uint64_t  payloadSerializationInt;
  };
  
  /// data type descriptor
  union {
    char     dataDescription[gSizeDataDescriptionString];
    uint64_t  dataDescriptionInt[2];
  };

  /// sub specification (e.g. link number)
  uint64_t    subSpecification;

  /// size of the associated data
  uint64_t    payloadSize;

  //___NEVER MODIFY THE ABOVE
  //___NEW STUFF GOES BELOW

  //___the functions:
  DataHeader(); //ctor
  DataHeader(const DataHeader&); //copy ctor
  DataHeader& operator=(const DataHeader&); //assignment
  DataHeader& operator=(const DataOrigin&); //assignment
  DataHeader& operator=(const DataDescription&); //assignment
  DataHeader& operator=(const PayloadSerialization&); //assignment
  bool operator==(const DataHeader&); //comparison
  bool operator==(const DataOrigin&); //comparison
  bool operator==(const DataDescription&); //comparison
  bool operator==(const PayloadSerialization&); //comparison
  void print() const;
};

//____________________________________________________________________________
/// @struct DataOrigin
/// @brief Helper struct to encode origin of data.
///
/// The DataHeader stores the origin of data, e.g. detector in a dedicted field,
/// DataOrigin structure is used for assignment and comparison
///
/// @ingroup aliceo2_dataformats_dataheader
struct DataOrigin
{
  //origin of the data (originating detector)
  union {
    char     dataOrigin[gSizeDataOriginString];
    uint32_t  dataOriginInt;
  };
  DataOrigin();
  DataOrigin(const DataOrigin& other) : dataOriginInt(other.dataOriginInt) {}
  DataOrigin& operator=(const DataOrigin& other) {
    if (&other != this) dataOriginInt = other.dataOriginInt;
    return *this;
  }
  // note: no operator=(const char*) as this potentially runs into trouble with this
  // general pointer type, use: someorigin = DataOrigin("BLA")
  DataOrigin(const char* origin);
  bool operator==(const DataOrigin&) const;
  bool operator!=(const DataOrigin& other) const {return not this->operator==(other);}
  void print() const;
};

//____________________________________________________________________________
/// @struct DataDescription
/// @brief Helper struct to encode description of the data.
///
/// The DataHeader stores the description of data, e.g. raw, tracks, ... in a dedicted
/// field, DataDescription structure is used for assignment and comparison
///
/// @ingroup aliceo2_dataformats_dataheader
struct DataDescription
{
  //data type descriptor
  union {
    char     dataDescription[gSizeDataDescriptionString];
    uint64_t  dataDescriptionInt[2];
  };
  DataDescription();
  DataDescription(const DataDescription& other) : dataDescriptionInt() {*this = other;}
  DataDescription& operator=(const DataDescription& other) {
    if (&other != this) {
      dataDescriptionInt[0] = other.dataDescriptionInt[0];
      dataDescriptionInt[1] = other.dataDescriptionInt[1];
    }
    return *this;
  }
  // note: no operator=(const char*) as this potentially runs into trouble with this
  // general pointer type, use: somedesc = DataOrigin("SOMEDESCRIPTION")
  DataDescription(const char* desc);
  bool operator==(const DataDescription&) const;
  bool operator!=(const DataDescription& other) const {return not this->operator==(other);}
  void print() const;
};

//____________________________________________________________________________
/// @struct DataIdentifier
/// @brief Helper struct to encode origin and description of data.
///
/// The DataHeader stores origin and description of data in adedicted fields,
/// DataIdentifier structure is used for assignment and comparison
///
/// @ingroup aliceo2_dataformats_dataheader
struct DataIdentifier
{
  //a full data identifier combining origin and description
  DataDescription dataDescription;
  DataOrigin dataOrigin;
  DataIdentifier();
  DataIdentifier(const DataIdentifier&);
  DataIdentifier(const char* desc, const char* origin);
  bool operator==(const DataIdentifier&) const;
  void print() const;
};

//____________________________________________________________________________
/// @struct PayloadSerialization
/// @brief Helper struct to encode payload serialization method of the data.
///
/// The DataHeader stores the payload serialization method in a dedicted field,
/// PayloadSerialization structure is used for assignment and comparison
///
/// @ingroup aliceo2_dataformats_dataheader
struct PayloadSerialization
{
  //serialization method
  union {
    char     payloadSerialization[gSizeSerializationString];
    uint64_t  payloadSerializationInt;
  };
  PayloadSerialization();
  // note: no operator=(const char*) as this potentially runs into trouble with this
  // general pointer type, use: sertype = DataOrigin("SERTYPE")
  PayloadSerialization(const char* serialization);
  bool operator==(const PayloadSerialization&) const;
  void print() const;
};

//____________________________________________________________________________
/// @defgroup data_description_defines Defines for data description
/// @ingroup aliceo2_dataformats_dataheader
/// @{

//____________________________________________________________________________
/// default int representation of 'invalid' token for 4-byte char field
const uint32_t gInvalidToken32 = 0x00202020;
/// default int representation of 'invalid' token for 8-byte char field
const uint64_t gInvalidToken64 = 0x0020202020202020;
/// invalid version
 const uint32_t gInvalidVersion = 0;

//____________________________________________________________________________
//possible data origins
extern const DataOrigin gDataOriginAny;
extern const DataOrigin gDataOriginInvalid;
extern const DataOrigin gDataOriginTPC;
extern const DataOrigin gDataOriginTRD;
extern const DataOrigin gDataOriginTOF;

//____________________________________________________________________________
//possible data types
extern const DataDescription gDataDescriptionAny;
extern const DataDescription gDataDescriptionInvalid;
extern const DataDescription gDataDescriptionRaw;
extern const DataDescription gDataDescriptionClusters;
extern const DataDescription gDataDescriptionTracks;

//____________________________________________________________________________
//possible serialization types
extern const PayloadSerialization gPayloadSerializationAny;
extern const PayloadSerialization gPayloadSerializationInvalid;
extern const PayloadSerialization gPayloadSerializationNone;
extern const PayloadSerialization gPayloadSerializationROOT;
extern const PayloadSerialization gPayloadSerializationFlatBuf;

/// @} // end of doxygen group

} //namespace Base
} //namespace AliceO2

#endif
