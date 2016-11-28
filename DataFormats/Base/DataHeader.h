//The main O2 data header, v0.1
//origin: CWG4
//blame: Mikolaj Krzewicki, mkrzewic@cern.ch
//       Matthias Richter, Matthias.Richter@cern.ch
//       David Rohr, drohr@cern.ch

#ifndef ALICEO2_BASE_DATA_HEADER_
#define ALICEO2_BASE_DATA_HEADER_

#include <cstdint>
#include <stdio.h>
#include <iostream>


// define a byte type
//enum class byte : unsigned char
//{
//};
using byte = unsigned char;

namespace AliceO2 {
namespace Base {

//__________________________________________________________________________________________________
/// @defgroup aliceo2_dataformat_primitives Primitive data format definitions for ALICE O2
/// @brief This module collects information about all primitive data formats.
///
/// More to come

//__________________________________________________________________________________________________
/// @defgroup aliceo2_dataformats_dataheader The Data Header
/// @brief A descriptive information for payload blocks
///
/// The format consists of header information and payload. The latter is not touched by the
/// framework. Each payload is described by the information in the header, for transport
/// a sequence of separate parts (header and payload in separate parts) is sent. The header
/// is described by structure \ref DataHeader
///
/// @ingroup aliceo2_dataformat_primitives

//__________________________________________________________________________________________________
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

//__________________________________________________________________________________________________
struct DataHeader;
struct DataIdentifier;

//__________________________________________________________________________________________________
//intializers
constexpr uint64_t String2uint64(char c1, char c2, char c3, char c4, char c5, char c6, char c7)
{
	return((uint64_t) c1 | (uint64_t) c2 << 8 | (uint64_t) c3 << 16 | (uint64_t) c4 << 24 |
      (uint64_t) c5 << 32 | (uint64_t) c6 << 40 | (uint64_t) c7 << 48);
}

constexpr uint64_t String2uint64(const char* str)
{
	return((uint64_t) str[0] |
         (str[0] ? ((uint64_t) str[1] << 8 |
         (str[1] ? ((uint64_t) str[2] << 16 |
         (str[2] ? ((uint64_t) str[3] << 24 |
         (str[3] ? ((uint64_t) str[4] << 32 |
         (str[4] ? ((uint64_t) str[5] << 40 |
         (str[5] ? ((uint64_t) str[6] << 48 )
          : 0)) : 0)) : 0)) : 0)) : 0)) : 0));
}

constexpr uint32_t String2uint32(char c1, char c2, char c3)
{
	return((uint32_t) c1 | (uint32_t) c2 << 8 | (uint32_t) c3 << 16);
}

constexpr uint32_t String2uint32(const char* str)
{
	return((uint32_t) str[0] |
         (str[0] ? ((uint32_t) str[1] << 8 |
         (str[1] ? ((uint32_t) str[2] << 16 )
          : 0)) : 0));
}

constexpr uint32_t CharArr2uint32(char c1, char c2, char c3, char c4)
{
	return((uint32_t) c1 |
         (uint32_t) c2 << 8 |
         (uint32_t) c3 << 16 |
         (uint32_t) c4 << 24);
}

constexpr uint32_t CharArr2uint32(const char* str)
{
	return((uint32_t) str[0] |
         (str[0] ? ((uint32_t) str[1] << 8 |
         (str[1] ? ((uint32_t) str[2] << 16 |
         (str[2] ? ((uint32_t) str[3] << 24)
          : 0)) : 0)) : 0));
}

//__________________________________________________________________________________________________
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

  /// header type description
  struct HeaderType {
    union {
      char     str[gSizeHeaderDescriptionString];
      uint64_t itg;
    };
    HeaderType(uint64_t v) {itg=v;}
    HeaderType(const char* s) {itg=String2uint64(s);}
    HeaderType(const HeaderType& that) {itg=that.itg;}
    bool operator==(const HeaderType& that) const {return that.itg==itg;}
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

  //ctor
  BaseHeader();
  BaseHeader(const BaseHeader&) = default;
  //ctor for use in derived types
  BaseHeader(uint32_t size, HeaderType description, SerializationMethod serialization);

  //access
  inline static BaseHeader* get(char* b, size_t /*len*/=0) {
    //this is to guess if the buffer starting at b looks like a header
    return (*(reinterpret_cast<uint32_t*>(b))==sMagicString)?reinterpret_cast<BaseHeader*>(b):nullptr;
  }

  inline uint32_t size() const noexcept { return headerSize; }
  inline const char* data() const noexcept { return reinterpret_cast<const char*>(this); }

  inline BaseHeader* next() {
    return (flagsNextHeader)?
      reinterpret_cast<BaseHeader*>(reinterpret_cast<unsigned char*>(this)+headerSize):nullptr;
  }

  // find a header with specified description (every header type must define one, usually a global)
  inline BaseHeader* get(const HeaderType desc) {
    BaseHeader* current = this;
    if (current->description==desc) return current;
    while ((current = current->next())) {
      if (current->description==desc) return current;
    }
    return nullptr;
  }

  // methods to construct a multi-header
  // intended use:
  //   compose(const T& header1, const T& header2, ...)
  //   - T are derived from BaseHeader, arbirtary number of arguments/headers
  //   - return is a unique_ptr holding the serialized buffer ready to be shipped.
  using MemorySpan = std::unique_ptr<char[]>;
  using Buffer = std::pair<MemorySpan,size_t>;

  //TODO: this should maybe return a std::vector<char>
  // - nope, there is no way we can release the actual data to zmq using vector
  //TODO: maybe add a specialization requiring first arg to be DataHeader
  template<typename... Args>
  static Buffer compose(const Args... args) {
    size_t bufferSize = size(args...);
    MemorySpan memory(new char[bufferSize]);
    inject(memory.get(), args...);
    return std::make_pair(std::move(memory),bufferSize);
  }

private:
  template<typename T>
  static size_t size(const T& h) noexcept {
    return h.size();
  }

  template<typename T, typename... Args>
  static size_t size(const T& h, const Args... args) noexcept {
    return size(h) + size(args...);
  }

  template<typename T>
  static char* inject(char* here, const T& h) noexcept {
    std::copy(h.data(), h.data()+h.size(), here);
    return here + h.size();
  }

  template<typename T, typename... Args>
  static char* inject(char* here, const T& h, const Args... args) noexcept {
    auto alsohere = inject(here, h);
    (reinterpret_cast<BaseHeader*>(here))->flagsNextHeader = true;
    return inject(alsohere, args...);
  }
};

//__________________________________________________________________________________________________
/// @struct ROOTobjectHeader
/// @brief an example data header containing a name of a ROOT object
struct ROOTobjectHeader : public BaseHeader {
  static const HeaderType sHeaderType;
  static const SerializationMethod sSerializationMethod;
  ROOTobjectHeader() :
    BaseHeader(sizeof(ROOTobjectHeader), sHeaderType, sSerializationMethod)
  , name()
  {
    //memset(static_cast<void*>(&name[0]),'Z',64);
    memset(&name[0],'Z',64);
  }
  char name[64];
};

//__________________________________________________________________________________________________
/// @struct DataHeader
/// @brief the main header struct
///
/// @ingroup aliceo2_dataformats_dataheader
struct DataHeader : public BaseHeader
{
  static const HeaderType sHeaderType;
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
  DataHeader(const DataHeader&) = default;
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
    return (baseHeader->description==DataHeader::sHeaderType)?
    static_cast<const DataHeader*>(baseHeader):nullptr;
  }
};

//__________________________________________________________________________________________________
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

//__________________________________________________________________________________________________
/// @defgroup data_description_defines Defines for data description
/// @ingroup aliceo2_dataformats_dataheader
/// @{

//__________________________________________________________________________________________________
/// default int representation of 'invalid' token for 4-byte char field
const uint32_t gInvalidToken32 = 0x00202020;
/// default int representation of 'invalid' token for 8-byte char field
const uint64_t gInvalidToken64 = 0x0020202020202020;
/// invalid version
const uint32_t gInvalidVersion = 0;

//__________________________________________________________________________________________________
//possible data origins
extern const DataHeader::DataOrigin gDataOriginAny;
extern const DataHeader::DataOrigin gDataOriginInvalid;
extern const DataHeader::DataOrigin gDataOriginTPC;
extern const DataHeader::DataOrigin gDataOriginTRD;
extern const DataHeader::DataOrigin gDataOriginTOF;

//__________________________________________________________________________________________________
//possible data types
extern const DataHeader::DataDescription gDataDescriptionAny;
extern const DataHeader::DataDescription gDataDescriptionInvalid;
extern const DataHeader::DataDescription gDataDescriptionRawData;
extern const DataHeader::DataDescription gDataDescriptionClusters;
extern const DataHeader::DataDescription gDataDescriptionTracks;

//__________________________________________________________________________________________________
//possible serialization types
extern const BaseHeader::SerializationMethod gSerializationMethodAny;
extern const BaseHeader::SerializationMethod gSerializationMethodInvalid;
extern const BaseHeader::SerializationMethod gSerializationMethodNone;
extern const BaseHeader::SerializationMethod gSerializationMethodROOT;
extern const BaseHeader::SerializationMethod gSerializationMethodFlatBuf;

/// @} // end of doxygen group

} //namespace Base
} //namespace AliceO2

//__________________________________________________________________________________________________
//helper function to print a hex/ASCII dump of some memory
void hexDump (const char* desc, const void* voidaddr, size_t len) {
  size_t i;
  unsigned char buff[17];       // stores the ASCII data
  const byte* addr = reinterpret_cast<const byte*>(voidaddr);

  // Output description if given.
  if (desc != NULL)
    printf ("%s, ", desc);
  printf("%zu bytes:\n", len);

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

#endif
