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

/// @brief O2 data header classes and API, v0.1
///
/// origin: CWG4
/// @author Mikolaj Krzewicki, mkrzewic@cern.ch
/// @author Matthias Richter, Matthias.Richter@cern.ch
/// @author David Rohr, drohr@cern.ch

#ifndef ALICEO2_BASE_DATA_HEADER_
#define ALICEO2_BASE_DATA_HEADER_

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>

// enum class byte : unsigned char
// {
// };
/// @typedef define a byte type
using byte = unsigned char;

namespace o2 {
namespace Header {

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
constexpr uint32_t gSizeMagicString = 4;
/// size of the data origin field @ingroup dataheader_defines
constexpr uint32_t gSizeDataOriginString = 4;
/// size of the payload serialization field @ingroup dataheader_defines
constexpr uint32_t gSizeSerializationMethodString = 8;
/// size of the data description field @ingroup dataheader_defines
constexpr uint32_t gSizeDataDescriptionString = 16;
/// size of the header description field @ingroup dataheader_defines
constexpr uint32_t gSizeHeaderDescriptionString = 8;

/// @}

//__________________________________________________________________________________________________
struct DataHeader;
struct DataIdentifier;

//__________________________________________________________________________________________________
// internal implementations
/// @ingroup aliceo2_dataformat_tools
namespace Internal {
// terminating initializer implementation
template <typename T>
constexpr T String2__()
{
  return 0;
}
// recursive initializer implementation
template <typename T, typename... Targs>
constexpr T String2__(char c, Targs... Fargs)
{
  return (T) c | String2__<T>(Fargs...) << 8;
}

/// compile time evaluation of the number of arguments in argument pack
template<typename T, typename... Targs>
struct getnargs {
  static int const value = getnargs<Targs...>::value + 1;
};
template<typename T>
struct getnargs<T> {
  static int const value = 1;
};

/// get the number of active bits (set to 1) in a bitfield
template <unsigned int N>
struct NumberOfActiveBits {
  static int const value = NumberOfActiveBits<(N >> 1)>::value + (N & 0x1);
};
template <>
struct NumberOfActiveBits<0> {
  static int const value = 0;
};

/// evaluate the array size necessary to hold a N-byte number with type T
template <int N, typename T>
struct ArraySize {
  static_assert(N >= sizeof(T), "get this code to work first");
  static int const value = 1;
  //static int const value = std::conditional<(N > sizeof(T)), ArraySize<(N - sizeof(T)), T>, ArraySize<0, T>>::value + 1;
};
template <typename T>
struct ArraySize<0, T> {
  static int const value = 0;
};

/// select uint type depending on size, default is uint64_t
template <int N>
struct TraitsIntType {
  typedef uint64_t Type;
};
template <>
struct TraitsIntType<1> {
  typedef uint8_t Type;
};
template <>
struct TraitsIntType<2> {
  typedef uint16_t Type;
};
template <>
struct TraitsIntType<4> {
  typedef uint32_t Type;
};

struct defaultPrinter {
  void operator()(const char* str) const {}
};

/// helper wrapper to define a type from an integer, which can than
/// be used as template argument
template <int N>
struct intWrapper {
  static const int value = N;
};

/// compile time evaluation of a string length, which is either N - 1 or
/// shorter if one character in the array has been set to 0.
template <int N>
constexpr std::size_t strLength(const char (&str)[N], std::size_t pos = 0) {
  return ((pos >= N || str[pos] == 0) ? 0 : 1 + strLength(str, pos + 1));
}
};

//__________________________________________________________________________________________________
/// constexpr intializer, evaluated at compile time
/// generic intializer for variable char argument list
template <typename T, typename... Targs>
constexpr T String2(char c, Targs... Fargs)
{
  // number of arguments has either to match the size of the type, or the last element is treated
  // as '0' if missing
  // TODO: this is a bit of a question, do we allow the number of characters to
  // be smaller than the size of the type and just pad with 0 like in the case
  // of a char array argument?
  static_assert(Internal::getnargs<T, Targs...>::value == sizeof(T) ||
		Internal::getnargs<T, Targs...>::value == sizeof(T) - 1,
		"number of arguments does not match the uint type width"
		);
  return Internal::String2__<T>(c, Fargs...);
}

/// constexpr intializer, evaluated at compile time
/// generic initializer, convert a string to unsigned integer of different width
/// Example usage: String2<uint64_t>("IDENTIFY")
/// @ingroup aliceo2_dataformat_tools
template <typename T, std::size_t N, std::size_t pos = 0, bool suppressAssert = false>
constexpr T String2(const char (&str)[N])
{
  static_assert(std::is_integral<T>::value, "Non integral types not compatible with String2<type>");
  static_assert(N >= pos, "Position is larger than the length of string");
  static_assert(suppressAssert || N - pos <= sizeof(T) + 1,
		"String parameter is longer than the size of the data type"
		);
  return((T) str[0+pos] |
         (str[0+pos] && sizeof(T) >= 2 ? ((T) str[1+pos] << (sizeof(T) >= 2 ? 8  : 0) |
         (str[1+pos] && sizeof(T) >= 4 ? ((T) str[2+pos] << (sizeof(T) >= 4 ? 16 : 0) |
         (str[2+pos] && sizeof(T) >= 4 ? ((T) str[3+pos] << (sizeof(T) >= 4 ? 24 : 0) |
         (str[3+pos] && sizeof(T) >= 8 ? ((T) str[4+pos] << (sizeof(T) >= 8 ? 32 : 0) |
         (str[4+pos] && sizeof(T) >= 8 ? ((T) str[5+pos] << (sizeof(T) >= 8 ? 40 : 0) |
         (str[5+pos] && sizeof(T) >= 8 ? ((T) str[6+pos] << (sizeof(T) >= 8 ? 48 : 0) |
         (str[6+pos] && sizeof(T) >= 8 ? ((T) str[7+pos] << (sizeof(T) >= 8 ? 56 : 0) )
          : 0)) : 0)) : 0)) : 0)) : 0)) : 0)) : 0));
}

//__________________________________________________________________________________________________
/// generic descriptor class faturing the union of a char and a uint element
/// of the same size
/// this is currently working only for up to 8 bytes aka uint64_t, the general
/// solution is working also for multiples of 64 bit, but then the itg member needs
/// to be an array for all. This has not been enabled yet, first the implications
/// have to be studied.
template <int N, typename PrinterPolicy = Internal::defaultPrinter>
struct Descriptor {
  static_assert(Internal::NumberOfActiveBits<N>::value == 1 && N <= 8,
		"size is required to be 1 or a multiple of 2");
  static int const size = N;
  static int const bitcount = size*8;
  static int const arraySize = 1; //Internal::ArraySize<size, uint64_t>::value;
  typedef typename Internal::TraitsIntType<N>::Type ItgType;

  union {
    char     str[N];
    ItgType  itg; // for extension > 64 bit: [arraySize];
  };
  constexpr Descriptor() : itg(0) {};
  constexpr Descriptor(ItgType initializer) : itg(initializer) {};
  constexpr Descriptor(const Descriptor& other) : itg(other.itg) {}
  Descriptor& operator=(const Descriptor& other) {
    if (&other != this) itg = other.itg;
    return *this;
  }
  // Note: don't need to define operator=(ItgType v) because the compiler
  // can use Descriptor(ItgType initializer) for conversion

  // type cast operator for simplified usage of the descriptor's integer member
  operator ItgType() const {return itg;}

  /// constructor from a compile-time string
  template<std::size_t NN>
  constexpr Descriptor(const char (&origin)[NN]) : itg(String2<ItgType>(origin)) {};

  /// Init descriptor from string at runtime
  /// In contrast to all other functions which are fixed at compile time, this is
  /// available at runtime and must be used with care
  ///
  /// Note: no assignment operator operator=(const char*) as this potentially runs
  /// into trouble with this general pointer type.
  void runtimeInit(const char* string) {
    char* target = str;
    char* targetEnd = target + N;
    const char* source = string;
    for ( ; source != nullptr && target < targetEnd && *source !=0; ++target, ++source) *target = *source;
    for ( ; target < targetEnd; ++target) *target = 0;
    // require the string to be maximum the descriptor size
    assert(source != nullptr && *source == 0);
  }

  bool operator==(const Descriptor& other) const {return itg == other.itg;}
  bool operator!=(const Descriptor& other) const {return not this->operator==(other);}
  // print function needs to be implemented for every derivation
  PrinterPolicy printer;
  void print() const {
    // eventually terminate string before printing
    printer(str);
  };
};

//__________________________________________________________________________________________________
/// default int representation of 'invalid' token for 4-byte char field
const uint32_t gInvalidToken32 = 0xFFFFFFFF;
/// default int representation of 'invalid' token for 8-byte char field
const uint64_t gInvalidToken64 = 0xFFFFFFFFFFFFFFFF;

struct HeaderType {
  union {
    char     str[gSizeHeaderDescriptionString];
    uint64_t itg;
  };
  constexpr HeaderType(uint64_t v) : itg{v} {}
  template<std::size_t N>
  constexpr HeaderType(const char (&s)[N]) : itg{String2<uint64_t>(s)} {}
  constexpr HeaderType(const HeaderType& that) : itg{that.itg} {}
  bool operator==(const HeaderType& that) const {return that.itg==itg;}
  bool operator==(const uint64_t& that) const {return that==itg;}
};

struct SerializationMethod {
  union {
    char      str[gSizeSerializationMethodString];
    uint64_t  itg;
  };
  constexpr SerializationMethod(uint64_t v) : itg{v} {}
  template<std::size_t N>
  constexpr SerializationMethod(const char (&s)[N]) : itg{String2<uint64_t>(s)} {}
  constexpr SerializationMethod(const SerializationMethod& that) : itg{that.itg} {}
  bool operator==(const SerializationMethod& that) const {return that.itg==itg;}
  bool operator==(const uint64_t& that) const {return that==itg;}
};

//possible serialization types
extern const o2::Header::SerializationMethod gSerializationMethodAny;
extern const o2::Header::SerializationMethod gSerializationMethodInvalid;
extern const o2::Header::SerializationMethod gSerializationMethodNone;
extern const o2::Header::SerializationMethod gSerializationMethodROOT;
extern const o2::Header::SerializationMethod gSerializationMethodFlatBuf;

//__________________________________________________________________________________________________
/// @struct BaseHeader
/// @brief the base header struct
/// Every header type must begin (i.e. derive) with this.
/// Don't use this struct directly.
/// The derived header must initialize BaseHeader fields with appropriate values
/// - headerSize must be initialized to sizeof(DerivedHeader)
/// 3 static members MUST be defined in each derived type:
/// - sHeaderType; description must be initialized with a static const member of type
///   BaseHeader::HeaderType with a gloablly unique value.
/// - sSerializationMethod; the contents of a derived header (content defined after the BaseHeader)
///   can be serialized using any method. The serialization member must be initialized
///   with a gloablly unique static const BaseHeader::SerializationMethod value defined
///   in the Derived header.
/// - sVersion; the version of the header described by this base header.
/// Call the special BaseHeader constructor in the DerivedHeader ctor to easily achieve
/// the above.
/// @ingroup aliceo2_dataformats_dataheader
struct BaseHeader
{
  // static definitions
  static const uint32_t sMagicString;

  static const uint32_t sVersion;
  static const o2::Header::HeaderType sHeaderType;
  static const o2::Header::SerializationMethod sSerializationMethod;

  //__the data layout:

  /// a magic string, used to identify an O2 header in a raw stream of bytes
  union {
    char     magicString[gSizeMagicString];
    uint32_t  magicStringInt;
  };

  /// size of the header that starts with this sequence (base + derived header)
  /// set by the derived header
  uint32_t    headerSize;

  /// flags, first bit indicates that a sub header follows
  union {
    uint32_t    flags;
    struct {
      uint32_t  flagsNextHeader :1, //do we have a next header after this one?
                flagsUnused :31;    //currently unused
    };
  };

  /// version of the entire header, set by the derived header
  uint32_t    headerVersion;

  /// header type description, set by derived header
  o2::Header::HeaderType description;

  /// header serialization method, set by derived header
  o2::Header::SerializationMethod serialization;

  //___the functions:

  /// dont construct directly
  BaseHeader();
  BaseHeader(const BaseHeader&) = default;
  /// Special ctor for initialization in derived types
  BaseHeader(uint32_t mySize, HeaderType description,
             SerializationMethod serialization, uint32_t version);

  /// @brief access header in buffer
  ///
  /// this is to guess if the buffer starting at b looks like a header
  inline static const BaseHeader* get(const byte* b, size_t /*len*/=0) {
    return (*(reinterpret_cast<const uint32_t*>(b))==sMagicString) ?
      reinterpret_cast<const BaseHeader*>(b) :
      nullptr;
  }

  inline uint32_t size() const noexcept { return headerSize; }
  inline const byte* data() const noexcept { return reinterpret_cast<const byte*>(this); }

  /// get the next header if any
  inline const BaseHeader* next() const noexcept {
    return (flagsNextHeader) ?
      reinterpret_cast<const BaseHeader*>(reinterpret_cast<const byte*>(this)+headerSize) :
      nullptr;
  }

};

/// find a header of type HeaderType in a buffer
/// use like this:
/// HeaderType* h = get<HeaderType>(buffer)
template<typename HeaderType>
const HeaderType* get(const byte* buffer, size_t /*len*/=0) {
  const BaseHeader* current = BaseHeader::get(buffer);
  if (!current) return nullptr;
  if (current->description==HeaderType::sHeaderType)
    return reinterpret_cast<const HeaderType*>(current);
  while ((current = current->next())) {
    if (current->description==HeaderType::sHeaderType)
      return reinterpret_cast<const HeaderType*>(current);
  }
  return nullptr;
}

template<typename HeaderType>
const HeaderType* get(const void* buffer, size_t len=0) {
  return get<HeaderType>(reinterpret_cast<const byte *>(buffer), len);
}

//__________________________________________________________________________________________________
/// @struct Stack
/// @brief a move-only header stack with serialized headers
/// This is the flat buffer where all the headers in a multi-header go.
/// This guy knows how to move the serialized content to FairMQ
/// and inform it how to release when all is sent.
/// methods to construct a multi-header
/// intended use:
///   - as a variadic intializer list (as an argument to a function)
///
///   One can also use Stack::compose(const T& header1, const T& header2, ...)
///   - T are derived from BaseHeader, arbirtary number of arguments/headers
///   - return is a unique_ptr holding the serialized buffer ready to be shipped.
struct Stack {

  // This is ugly and needs fixing BUT:
  // we need a static deleter for fairmq.
  // TODO: template the class with allocators
  //
  using Buffer = std::unique_ptr<byte[]>;
  static std::default_delete<byte[]> sDeleter;
  static void freefn(void* /*data*/, void* hint) {
    Stack::sDeleter(static_cast<byte*>(hint));
  }

  size_t bufferSize;
  Buffer buffer;

  byte* data() const {return buffer.get();}
  size_t size() const {return bufferSize;}

  ///The magic constructor: takes arbitrary number of arguments and serialized them
  /// into the buffer.
  /// Intended use: produce a temporary via an initializer list.
  /// TODO: maybe add a static_assert requiring first arg to be DataHeader
  /// or maybe even require all these to be derived form BaseHeader
  template<typename... Headers>
  Stack(const Headers&... headers)
    : bufferSize{size(headers...)}
    , buffer{new byte[bufferSize]}
  {
    inject(buffer.get(), headers...);
  }
  Stack() = default;
  Stack(Stack&&) = default;
  Stack(Stack&) = delete;
  Stack& operator=(Stack&) = delete;
  Stack& operator=(Stack&&) = default;

  /// the magic compose - serialize (almost) anything into the buffer
  /// (works with headers, strings and arrays)
  template<typename... Args>
  static Stack compose(const Args&... args) {
    Stack b;
    b.bufferSize = size(args...);
    b.buffer.reset(new byte[b.bufferSize]);
    inject(b.buffer.get(), args...);
    return b;
  }

  template<typename T, typename... Args>
  static size_t size(const T& h, const Args... args) noexcept {
    return size(h) + size(args...);
  }

private:
  template<typename T>
  static size_t size(const T& h) noexcept {
    return h.size();
  }

  template<typename T>
  static byte* inject(byte* here, const T& h) noexcept {
    std::copy(h.data(), h.data()+h.size(), here);
    return here + h.size();
  }

  template<typename T, typename... Args>
  static byte* inject(byte* here, const T& h, const Args... args) noexcept {
    auto alsohere = inject(here, h);
    (reinterpret_cast<BaseHeader*>(here))->flagsNextHeader = true;
    return inject(alsohere, args...);
  }
};

//__________________________________________________________________________________________________
/// @struct NameHeader
/// @brief an example data header containing a name of an object as a null terminated char arr.
/// this is a template! at instantiation the template parameter determines the
/// size of the held string array.
/// a caveat with decoding is you have to use Header::get<NameHeader<0>>(buffer)
/// to get it out of a buffer. May improve in the future if enough people complain.
/// @ingroup aliceo2_dataformats_dataheader
template <size_t N>
struct NameHeader : public BaseHeader {
  static const uint32_t sVersion;
  static const o2::Header::HeaderType sHeaderType;
  static const o2::Header::SerializationMethod sSerializationMethod;
  NameHeader()
  : BaseHeader(sizeof(NameHeader), sHeaderType, sSerializationMethod, sVersion)
  , name()
  {
    memset(&name[0],'\0',N);
  }

  NameHeader(std::string in)
  : BaseHeader(sizeof(NameHeader), sHeaderType, sSerializationMethod, sVersion)
  , name()
  {
    //std::copy(in.begin(), in.begin()+N, name);
    // here we actually wnat a null terminated string
    strncpy(name,in.c_str(),N);
    name[N-1] = '\0';
  }

  NameHeader& operator=(const std::string string) {
    std::copy(string.begin(), string.begin()+N, name);
    return *this;
  }
private:
  char name[N];
};

template <size_t N>
const o2::Header::HeaderType NameHeader<N>::sHeaderType = "NameHead";

// dirty trick to always have access to the headertypeID of a templated header type
// TODO: find out if this can be done in a nicer way + is this realy necessary?
template <>
const o2::Header::HeaderType NameHeader<0>::sHeaderType;

template <size_t N>
const SerializationMethod NameHeader<N>::sSerializationMethod = gSerializationMethodNone;

template <size_t N>
const uint32_t NameHeader<N>::sVersion = 1;

//__________________________________________________________________________________________________
/// this 128 bit type for a header field describing the payload data type
struct DataDescription {
  static const std::size_t N = gSizeDataDescriptionString;
  union {
    char     str[N];
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
  // general pointer type, use: somedesc = DataDescription("SOMEDESCRIPTION")
  template<std::size_t NN>
  constexpr DataDescription(const char (&desc)[NN]) : itg{String2<uint64_t, NN, 0, true>(desc), (Internal::strLength(desc) > 8 ? String2<uint64_t, NN, std::conditional< (NN > 8), Internal::intWrapper<8>, Internal::intWrapper<0>>::type::value >(desc) : 0)} {
    // the initializer implements a number of compile time checks
    // - for the conversion of the first part of the string to the first 64bit field
    //   the check for the string length needs to be disabled as it is naturally longer
    //   then the type size
    // - the second field only has to be filled if the string is longer than 8 chars, padding
    //   with 0 otherwize, this catches the rare case that one character within the array
    //   has been explicitely set to 0, but the compile time length is longer
    // - depending on the length of the char field, either the offset 8 can be chosen or the
    //   offset is set to the maximum, which will lead also to 0
    // - it's probably given that N > 0
    static_assert(N > 0, "Strange error, that should not happen at all");
  }

  /// Init from string at runtime
  /// In contrast to all other functions which are fixed at compile time, this is
  /// available at runtime and must be used with care
  ///
  /// Note: no assignment operator operator=(const char*) as this potentially runs
  /// into trouble with this general pointer type.
  void runtimeInit(const char* string) {
    char* target = str;
    char* targetEnd = target + N;
    const char* source = string;
    for ( ; source != nullptr && target < targetEnd && *source !=0; ++target, ++source) *target = *source;
    for ( ; target < targetEnd; ++target) *target = 0;
    // require the string to be maximum the descriptor size
    assert(source != nullptr && *source == 0);
  }

  bool operator==(const DataDescription&) const;
  bool operator!=(const DataDescription& other) const {return not this->operator==(other);}
  void print() const;
};

//__________________________________________________________________________________________________
// 32bit (4 characters) for data origin, ex. the detector or subsystem name
struct printDataOrigin {
  void operator()(const char* str) const;
};
typedef Descriptor<gSizeDataOriginString, printDataOrigin> DataOrigin;

//__________________________________________________________________________________________________
/// @struct DataHeader
/// @brief the main header struct
///
/// The main O2 data header struct. All messages should have it, preferably at the beginning
/// of the header stack.
/// It contains the fields that describe the buffer size, data type,
/// origin and serialization method used to construct the buffer.
/// The member subSpecification might be defined differently for each type/origin,
/// interpretation of this field is up to the specific subsystem involved.
///
/// @ingroup aliceo2_dataformats_dataheader
struct DataHeader : public BaseHeader
{
  // allows DataHeader::SubSpecificationType to be used as generic type in the code
  typedef uint64_t SubSpecificationType;

  //static data for this header type/version
  static const uint32_t sVersion;
  static const o2::Header::HeaderType sHeaderType;
  static const o2::Header::SerializationMethod sSerializationMethod;

  ///
  /// data type descriptor
  ///
  DataDescription dataDescription;

  ///
  /// origin of the data (originating detector)
  ///
  DataOrigin dataOrigin;

  ///
  /// need something for alignment, is there another field needed?
  /// how about a hash?
  ///
  uint32_t reserved;

  ///
  /// serialization method
  ///
  SerializationMethod payloadSerializationMethod;

  ///
  /// sub specification (e.g. link number)
  ///
  SubSpecificationType subSpecification;

  ///
  /// size of the associated data
  ///
  uint64_t    payloadSize;

  //___NEVER MODIFY THE ABOVE
  //___NEW STUFF GOES BELOW

  //___the functions:
  DataHeader(); ///ctor
  DataHeader(const DataHeader&) = default;
  DataHeader& operator=(const DataHeader&); //assignment
  DataHeader& operator=(const DataOrigin&); //assignment
  DataHeader& operator=(const DataDescription&); //assignment
  DataHeader& operator=(const SerializationMethod&); //assignment
  bool operator==(const DataHeader&) const; //comparison
  bool operator==(const DataOrigin&) const; //comparison
  bool operator==(const DataDescription&) const; //comparison
  bool operator==(const SerializationMethod&) const; //comparison
  void print() const; ///pretty print the contents

  static const DataHeader* Get(const BaseHeader* baseHeader) {
    return (baseHeader->description==DataHeader::sHeaderType)?
    static_cast<const DataHeader*>(baseHeader):nullptr;
  }
};

//__________________________________________________________________________________________________
/// @defgroup data_description_defines Defines for data description
/// @ingroup aliceo2_dataformats_dataheader
/// @{

//__________________________________________________________________________________________________
//possible data origins
extern const o2::Header::DataOrigin gDataOriginAny;
extern const o2::Header::DataOrigin gDataOriginInvalid;
extern const o2::Header::DataOrigin gDataOriginTPC;
extern const o2::Header::DataOrigin gDataOriginTRD;
extern const o2::Header::DataOrigin gDataOriginTOF;

//possible data types
extern const o2::Header::DataDescription gDataDescriptionAny;
extern const o2::Header::DataDescription gDataDescriptionInvalid;
extern const o2::Header::DataDescription gDataDescriptionRawData;
extern const o2::Header::DataDescription gDataDescriptionClusters;
extern const o2::Header::DataDescription gDataDescriptionTracks;
extern const o2::Header::DataDescription gDataDescriptionConfig;
extern const o2::Header::DataDescription gDataDescriptionInfo;
/// @} // end of doxygen group

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
  DataDescription dataDescription;
  DataOrigin dataOrigin;
  DataIdentifier();
  DataIdentifier(const DataIdentifier&);
  template<std::size_t N, std::size_t M>
  DataIdentifier(const char (&desc)[N], const char (&origin)[M])
    : dataDescription(desc), dataOrigin(origin)
  {
  }

  bool operator==(const DataIdentifier&) const;
  void print() const;
};

//__________________________________________________________________________________________________
///helper function to print a hex/ASCII dump of some memory
void hexDump (const char* desc, const void* voidaddr, size_t len, size_t max=0);

} //namespace Header
} //namespace AliceO2

#endif
