//The main O2 data header, v0.1
//origin: CWG4
//blame: Mikolaj Krzewicki, mkrzewic@cern.ch
//       Matthias Richter, Matthias.Richter@cern.ch
//       David Rohr, drohr@cern.ch

#ifndef ALICEO2_BASE_DATA_HEADER_
#define ALICEO2_BASE_DATA_HEADER_

#include <cstdint>
#include <stdio.h>

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
const uint32_t gSizeSerializationMethodString = 8;
/// size of the data description field @ingroup dataheader_defines
const uint32_t gSizeDataDescriptionString = 16;
/// size of the header description field @ingroup dataheader_defines
const uint32_t gSizeHeaderDescriptionString = 8;
/// @}

//____________________________________________________________________________
struct DataHeader;
struct DataIdentifier;

//intializers
constexpr uint64_t String2uint64(char c1, char c2, char c3, char c4, char c5, char c6, char c7)
{
	return((uint64_t) c1 | (uint64_t) c2 << 8 | (uint64_t) c3 << 16 | (uint64_t) c4 << 24 | (uint64_t) c5 << 32 | (uint64_t) c6 << 40 | (uint64_t) c7 << 48);
}

constexpr uint64_t String2uint64(const char* str)
{
	return((uint64_t) str[0] | (str[0] ? ((uint64_t) str[1] << 8 | (str[1] ? ((uint64_t) str[2] << 16 | (str[2] ? ((uint64_t) str[3] << 24 | (str[3] ? ((uint64_t) str[4] << 32 | (str[4] ? ((uint64_t) str[5] << 40 | (str[5] ? ((uint64_t) str[6] << 48 ) : 0)) : 0)) : 0)) : 0)) : 0)) : 0));
}

constexpr uint32_t String2uint32(char c1, char c2, char c3)
{
	return((uint32_t) c1 | (uint32_t) c2 << 8 | (uint32_t) c3 << 16);
}

constexpr uint32_t String2uint32(const char* str)
{
	return((uint32_t) str[0] | (str[0] ? ((uint32_t) str[1] << 8 | (str[1] ? ((uint32_t) str[2] << 16 ) : 0)) : 0));
}

constexpr uint32_t CharArr2uint32(char c1, char c2, char c3, char c4)
{
	return((uint32_t) c1 | (uint32_t) c2 << 8 | (uint32_t) c3 << 16 | (uint32_t) c4 << 24);
}

constexpr uint32_t CharArr2uint32(const char* str)
{
	return((uint32_t) str[0] | (str[0] ? ((uint32_t) str[1] << 8 | (str[1] ? ((uint32_t) str[2] << 16 | (str[2] ? ((uint32_t) str[3] << 24) : 0)) : 0)) : 0));
}

//____________________________________________________________________________
/// @struct BaseHeader
/// @brief the base header struct
///
/// @ingroup aliceo2_dataformats_dataheader
struct BaseHeader
{
  //other constants
  static const uint32_t sVersion;
  static const uint32_t sMagicString;

  //__the data layout:

  /// a magic string
  union {
    char     magicString[gSizeMagicString];
    uint32_t  magicStringInt;
  };

  /// size of this header (base + derived header)
  uint32_t    headerSize;

  /// flags, first bit indicates that a sub header follows
  union {
    uint32_t    flags;
    struct {
      uint32_t  flagsNextHeader :1, //do we have a next header after this one?
                flagsUnused :31;    //currently unused
    };
  };

  /// version of this header
  uint32_t    headerVersion;

  /// header contents description
  struct Description {
    union {
      char     str[gSizeHeaderDescriptionString];
      uint64_t itg;
    };
    Description(uint64_t v) {itg=v;}
    Description(const char* s) {itg=String2uint64(s);}
    Description(const Description& that) {itg=that.itg;}
    bool operator==(const Description& that) const {return that.itg==itg;}
    bool operator==(const uint64_t& that) const {return that==itg;}
  } description;

  /// header serialization method
  struct SerializationMethod {
    union {
      char      str[gSizeSerializationMethodString];
      uint64_t  itg;
    };
    SerializationMethod(uint64_t v) {itg=v;}
    SerializationMethod(const char* s) {itg=String2uint64(s);}
    SerializationMethod(const SerializationMethod& that) {itg=that.itg;}
    bool operator==(const SerializationMethod& that) const {return that.itg==itg;}
    bool operator==(const uint64_t& that) const {return that==itg;}
  } serialization;

  //___the functions:
  BaseHeader(); //ctor
  BaseHeader(uint32_t size, Description description, SerializationMethod serialization); //ctor for use in derived types
  BaseHeader(const BaseHeader&); //copy ctor
  static BaseHeader* Get(void* b) {
    return (*(reinterpret_cast<uint32_t*>(b))==sMagicString)?static_cast<BaseHeader*>(b):nullptr;
  }
  inline BaseHeader* NextHeader() {return (flagsNextHeader)?reinterpret_cast<BaseHeader*>(reinterpret_cast<unsigned char*>(this)+headerSize):nullptr;}
};

//____________________________________________________________________________
/// @struct DataHeader
/// @brief the main header struct
///
/// @ingroup aliceo2_dataformats_dataheader
struct DataHeader : public BaseHeader
{
  static const Description sDescription;
  static const SerializationMethod sSerializationMethod;

  /// data type descriptor
  struct DataDescription {
    union {
      char     str[gSizeDataDescriptionString];
      uint64_t itg[2];
    };
    DataDescription();
    DataDescription(const DataDescription& other) : itg() {*this = other;}
    DataDescription& operator=(const DataDescription& other) {
      if (&other != this) {
        itg[0] = other.itg[0];
        itg[1] = other.itg[1];
      }
      return *this;
    }
    // note: no operator=(const char*) as this potentially runs into trouble with this
    // general pointer type, use: somedesc = DataOrigin("SOMEDESCRIPTION")
    DataDescription(const char* desc);
    bool operator==(const DataDescription&) const;
    bool operator!=(const DataDescription& other) const {return not this->operator==(other);}
    void print() const;
  } dataDescription;

  /// origin of the data (originating detector)
  struct DataOrigin {
    union {
      char     str[gSizeDataOriginString];
      uint32_t itg;
    };
    DataOrigin();
    DataOrigin(const DataOrigin& other) : itg(other.itg) {}
    DataOrigin& operator=(const DataOrigin& other) {
      if (&other != this) itg = other.itg;
      return *this;
    }
    // note: no operator=(const char*) as this potentially runs into trouble with this
    // general pointer type, use: someorigin = DataOrigin("BLA")
    DataOrigin(const char* origin);
    bool operator==(const DataOrigin&) const;
    bool operator!=(const DataOrigin& other) const {return not this->operator==(other);}
    void print() const;
  } dataOrigin;

  // need something for alignment, is there another field needed?
  uint32_t reserved;

  /// serialization method
  BaseHeader::SerializationMethod payloadSerializationMethod;

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
  DataHeader& operator=(const SerializationMethod&); //assignment
  bool operator==(const DataHeader&); //comparison
  bool operator==(const DataOrigin&); //comparison
  bool operator==(const DataDescription&); //comparison
  bool operator==(const SerializationMethod&); //comparison
  void print() const;
  static const DataHeader* Get(const BaseHeader* baseHeader) {
    return (baseHeader->description==DataHeader::sDescription)?
    static_cast<const DataHeader*>(baseHeader):nullptr;
  }
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
  DataHeader::DataDescription dataDescription;
  DataHeader::DataOrigin dataOrigin;
  DataIdentifier();
  DataIdentifier(const DataIdentifier&);
  DataIdentifier(const char* desc, const char* origin);
  bool operator==(const DataIdentifier&) const;
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
extern const DataHeader::DataOrigin gDataOriginAny;
extern const DataHeader::DataOrigin gDataOriginInvalid;
extern const DataHeader::DataOrigin gDataOriginTPC;
extern const DataHeader::DataOrigin gDataOriginTRD;
extern const DataHeader::DataOrigin gDataOriginTOF;

//____________________________________________________________________________
//possible data types
extern const DataHeader::DataDescription gDataDescriptionAny;
extern const DataHeader::DataDescription gDataDescriptionInvalid;
extern const DataHeader::DataDescription gDataDescriptionRawData;
extern const DataHeader::DataDescription gDataDescriptionClusters;
extern const DataHeader::DataDescription gDataDescriptionTracks;

//____________________________________________________________________________
//possible serialization types
extern const BaseHeader::SerializationMethod gSerializationMethodAny;
extern const BaseHeader::SerializationMethod gSerializationMethodInvalid;
extern const BaseHeader::SerializationMethod gSerializationMethodNone;
extern const BaseHeader::SerializationMethod gSerializationMethodROOT;
extern const BaseHeader::SerializationMethod gSerializationMethodFlatBuf;

/// @} // end of doxygen group

} //namespace Base
} //namespace AliceO2

//____________________________________________________________________________
//helper function to print a hex/ASCII dump of some memory
void hexDump (const char *desc, void *addr, int len) {
  int i;
  unsigned char buff[17];       // stores the ASCII data
  unsigned char *pc = static_cast<unsigned char*>(addr);     // cast to make the code cleaner.

  // Output description if given.
  if (desc != NULL)
    printf ("%s:\n", desc);

  // In case of null pointer addr
  if (addr==nullptr) {printf("  nullptr\n"); return;}

  // Process every byte in the data.
  for (i = 0; i < len; i++) {
    // Multiple of 16 means new line (with line offset).
    if ((i % 16) == 0) {
      // Just don't print ASCII for the zeroth line.
      if (i != 0)
        printf ("  %s\n", buff);

      // Output the offset.
      //printf ("  %04x ", i);
      printf ("  %p ", &pc[i]);
    }

    // Now the hex code for the specific character.
    printf (" %02x", pc[i]);

    // And store a printable ASCII character for later.
    if ((pc[i] < 0x20) || (pc[i] > 0x7e))
      buff[i % 16] = '.';
    else
      buff[i % 16] = pc[i];
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

#endif
