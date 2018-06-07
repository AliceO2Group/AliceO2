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

/// @brief O2 data header classes and API, v0.1
///
/// origin: CWG4
/// @author Mikolaj Krzewicki, mkrzewic@cern.ch
/// @author Matthias Richter, Matthias.Richter@cern.ch
/// @author David Rohr, drohr@cern.ch

#ifndef ALICEO2_BASE_DATA_HEADER_
#define ALICEO2_BASE_DATA_HEADER_

#include <cstdint>
#include <memory>
#include <cassert>
#include <cstring> //needed for memcmp
#include <algorithm> // std::min
#include <stdexcept>
#include "MemoryResources/MemoryResources.h"

using byte = unsigned char;

namespace o2 {
namespace header {

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
///helper function to print a hex/ASCII dump of some memory
void hexDump(const char* desc, const void* voidaddr, size_t len, size_t max = 0);

//__________________________________________________________________________________________________
// internal implementations
/// @ingroup aliceo2_dataformat_tools
namespace internal {
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
template <typename T, int N>
constexpr int ArraySize() {
  return N / sizeof(T) + ((N%sizeof(T))?1:0);
}

/// select uint type depending on size, default is uint64_t
template <int N>
struct TraitsIntType {
  using Type = uint64_t;
};
template <>
struct TraitsIntType<1> {
  using Type = uint8_t;
};
template <>
struct TraitsIntType<2> {
  using Type = uint16_t;
};
template <>
struct TraitsIntType<4> {
  using Type = uint32_t;
};

struct defaultPrinter {
  void operator()(const char* str) const {}
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
  static_assert(sizeof...(Targs) == sizeof(T) + 1 ||
                sizeof...(Targs) == sizeof(T),
                "number of arguments does not match the uint type width"
                );
  return internal::String2__<T>(c, Fargs...);
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
/// helper traits to efficiently compare descriptors
/// the default implementation with memcmp is basically never used
/// specializations handle the two relevant cases
template<int S>
struct DescriptorCompareTraits {
  template<typename T, typename Length>
  static bool compare(const T &lh, const T &rh, Length N) {
    return std::memcmp(lh.str, rh.str, N) == 0;
  }
};
template<>
struct DescriptorCompareTraits<1> {
  template<typename T, typename Length>
  static bool compare(const T &lh, const T &rh, Length) {
    return lh.itg[0] == rh.itg[0];
  }
};
template<>
struct DescriptorCompareTraits<2> {
  template<typename T, typename Length>
  static bool compare(const T &lh, const T &rh, Length) {
    return (lh.itg[0] == rh.itg[0]) && (lh.itg[1] == rh.itg[1]);
  }
};

//__________________________________________________________________________________________________
/// generic descriptor class faturing the union of a char and a uint element
/// of the same size
/// this is currently working only for up to 8 bytes aka uint64_t, the general
/// solution is working also for multiples of 64 bit, but then the itg member needs
/// to be an array for all. This has not been enabled yet, first the implications
/// have to be studied.
template <std::size_t N, typename PrinterPolicy = internal::defaultPrinter>
struct Descriptor {
  static_assert(internal::NumberOfActiveBits<N>::value == 1,
                "Descriptor size is required to be a power of 2");
  using self_type = Descriptor<N>;
  static int const size = N;
  static int const bitcount = size*8;
  static constexpr int arraySize = internal::ArraySize<uint64_t, size>();
  using ItgType = typename internal::TraitsIntType<N>::Type;

  union {
    char     str[N];
    ItgType  itg[arraySize];
  };
  constexpr Descriptor() : itg{0} {};
  constexpr Descriptor(ItgType initializer) : itg{initializer} {};
  constexpr Descriptor(const Descriptor& other) = default;
  Descriptor& operator=(const Descriptor& other) = default;

  // Note: don't need to define operator=(ItgType v) because the compiler
  // can use Descriptor(ItgType initializer) for conversion

  // type cast operator for simplified usage of the descriptor's integer member
  // TODO: this is sort of a hack, takes the first element.
  //       we should rethink these implicit conversions
  operator ItgType() const
  {
    static_assert(arraySize == 1, "casting Descriptor to ItgType only allowed for N<=8");
    return itg[0];
  }

  /// constructor from a compile-time string
  template<std::size_t L>
  constexpr Descriptor(const char (&in)[L]) : str{0}
  {
    static_assert(L <= N + 1, "initializer string must not exceed descriptor size");
    unsigned i = 0;
    for (; in[i] && i < std::min(N, L) ; ++i) {
      str[i] = in[i];
    }
  }

  /// Init descriptor from string at runtime
  /// In contrast to all other functions which are fixed at compile time, this is
  /// available at runtime and must be used with care
  ///
  /// Note: no assignment operator operator=(const char*) as this potentially runs
  /// into trouble with this general pointer type.
  void runtimeInit(const char* string, short length = -1) {
    char* target = str;
    char* targetEnd = target;
    if (length >= 0 && length < (int)N) targetEnd += length;
    else targetEnd += N;
    const char* source = string;
    for ( ; source != nullptr && target < targetEnd && *source !=0; ++target, ++source) *target = *source;
    targetEnd = str + N;
    for ( ; target < targetEnd; ++target) *target = 0;
    // require the string to be not longer than the descriptor size
    assert(source != nullptr && (*source == 0 || (length >= 0 && length <= (int)N)));
  }

  bool operator==(const Descriptor& other) const {return DescriptorCompareTraits<arraySize>::compare(*this,other, N);}
  bool operator!=(const Descriptor& other) const {return not this->operator==(other);}

  // explicitly forbid comparison with e.g. const char* strings
  // use: value == Descriptor<N>("DESC") for the appropriate
  // template instantiation instead
  template<typename T>
  bool operator==(const T*) const = delete;
  template<typename T>
  bool operator!=(const T*) const = delete;
  // print function needs to be implemented for every derivation
  void print() const {
    // eventually terminate string before printing
    PrinterPolicy printer;
    printer(str);
  };
};

//__________________________________________________________________________________________________
/// default int representation of 'invalid' token for 4-byte char field
const uint32_t gInvalidToken32 = 0xFFFFFFFF;
/// default int representation of 'invalid' token for 8-byte char field
const uint64_t gInvalidToken64 = 0xFFFFFFFFFFFFFFFF;

using HeaderType = Descriptor<gSizeHeaderDescriptionString>;
using SerializationMethod = Descriptor<gSizeSerializationMethodString>;

//possible serialization types
constexpr o2::header::SerializationMethod gSerializationMethodAny{ "*******" };
constexpr o2::header::SerializationMethod gSerializationMethodInvalid{ "INVALID" };
constexpr o2::header::SerializationMethod gSerializationMethodNone{ "NONE" };
constexpr o2::header::SerializationMethod gSerializationMethodROOT{ "ROOT" };
constexpr o2::header::SerializationMethod gSerializationMethodFlatBuf{ "FLATBUF" };

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
  static const o2::header::HeaderType sHeaderType;
  static const o2::header::SerializationMethod sSerializationMethod;

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
  o2::header::HeaderType description;

  /// header serialization method, set by derived header
  o2::header::SerializationMethod serialization;

  //___the functions:

  /// dont construct directly
  BaseHeader() = delete;
  BaseHeader(const BaseHeader&) = default;
  /// Special ctor for initialization in derived types
  BaseHeader(uint32_t mySize, HeaderType description,
             SerializationMethod serialization, uint32_t version);

  /// @brief access header in buffer
  ///
  /// this is to guess if the buffer starting at b looks like a header
  inline static const BaseHeader* get(const byte* b, size_t /*len*/ = 0)
  {
    return (b != nullptr && *(reinterpret_cast<const uint32_t*>(b)) == sMagicString)
             ? reinterpret_cast<const BaseHeader*>(b)
             : nullptr;
  }

  /// @brief access header in buffer
  ///
  /// this is to guess if the buffer starting at b looks like a header
  inline static BaseHeader* get(byte* b, size_t /*len*/ = 0)
  {
    return (b != nullptr && *(reinterpret_cast<uint32_t*>(b)) == sMagicString) ? reinterpret_cast<BaseHeader*>(b)
                                                                               : nullptr;
  }

  inline uint32_t size() const noexcept { return headerSize; }
  inline const byte* data() const noexcept { return reinterpret_cast<const byte*>(this); }

  /// get the next header if any (const version)
  inline const BaseHeader* next() const noexcept {
    return (flagsNextHeader) ?
      reinterpret_cast<const BaseHeader*>(reinterpret_cast<const byte*>(this)+headerSize) :
      nullptr;
  }

  /// get the next header if any (non-const version)
  inline BaseHeader* next() noexcept
  {
    return (flagsNextHeader) ? reinterpret_cast<BaseHeader*>(reinterpret_cast<byte*>(this) + headerSize) : nullptr;
  }
};

/// find a header of type HeaderType in a buffer
/// use like this:
/// HeaderType* h = get<HeaderType*>(buffer)
template <typename HeaderType, typename std::enable_if_t<std::is_pointer<HeaderType>::value, int> = 0>
auto get(const byte* buffer, size_t /*len*/ = 0)
{
  using HeaderConstPtrType = const typename std::remove_pointer<HeaderType>::type*;
  using HeaderValueType = typename std::remove_pointer<HeaderType>::type;

  const BaseHeader* current = BaseHeader::get(buffer);
  if (!current)
    return HeaderConstPtrType{ nullptr };
  if (current->description == HeaderValueType::sHeaderType)
    return reinterpret_cast<HeaderConstPtrType>(current);
  while ((current = current->next())) {
    if (current->description == HeaderValueType::sHeaderType)
      return reinterpret_cast<HeaderConstPtrType>(current);
  }
  return HeaderConstPtrType{ nullptr };
}

template <typename HeaderType, typename std::enable_if_t<std::is_pointer<HeaderType>::value, int> = 0>
auto get(const void* buffer, size_t len = 0)
{
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
///   One can also use the ctor directly:
//    Stack::Stack(const T& header1, const T& header2, ...)
//    - arguments can be headers, or stacks, all will be concatenated in a new Stack
///   - returns a Stack ready to be shipped.
struct Stack {

 private:
  static void freefn(void* data, void* hint)
  {
    boost::container::pmr::memory_resource* resource = static_cast<boost::container::pmr::memory_resource*>(hint);
    resource->deallocate(data, 0, 0);
  }

  struct freeobj {
    freeobj(boost::container::pmr::memory_resource* mr) : resource(mr) {}
    boost::container::pmr::memory_resource* resource{ nullptr };
    void operator()(byte* ptr) { Stack::freefn(ptr, resource); }
  };

 public:
  using allocator_type = boost::container::pmr::polymorphic_allocator<byte>;
  using value_type = byte;
  using BufferType = std::unique_ptr<value_type[], freeobj>;

  Stack() = default;
  Stack(Stack&&) = default;
  Stack(Stack&) = delete;
  Stack& operator=(Stack&) = delete;
  Stack& operator=(Stack&&) = default;

  value_type* data() const { return buffer.get(); }
  size_t size() const { return bufferSize; }
  allocator_type get_allocator() const { return allocator; }

  //
  boost::container::pmr::memory_resource* getFreefnHint() const noexcept { return allocator.resource(); }
  static auto getFreefn() noexcept { return &freefn; }

  /// The magic constructors: take arbitrary number of headers and serialize them
  /// into the buffer buffer allocated by the specified polymorphic allocator. By default
  /// allocation is done using new_delete_resource.
  /// In the final stack the first header must be DataHeader.
  /// all headers must derive from BaseHeader, in addition also other stacks can be passed to ctor.
  template <typename FirstArgType, typename... Headers,
            typename std::enable_if_t<
              !std::is_convertible<FirstArgType, boost::container::pmr::polymorphic_allocator<byte>>::value, int> = 0>
  Stack(FirstArgType&& firstHeader, Headers&&... headers)
    : Stack(boost::container::pmr::new_delete_resource(), std::forward<FirstArgType>(firstHeader),
            std::forward<Headers>(headers)...)
  {
  }

  template <typename... Headers>
  Stack(const allocator_type allocatorArg, Headers&&... headers)
    : allocator{ allocatorArg },
      bufferSize{ calculateSize(std::forward<Headers>(headers)...) },
      buffer{ static_cast<byte*>(allocator.resource()->allocate(bufferSize, alignof(std::max_align_t))),
              freeobj{ getFreefnHint() } }
  {
    inject(buffer.get(), std::forward<Headers>(headers)...);
  }

 private:
  allocator_type allocator{ boost::container::pmr::new_delete_resource() };
  size_t bufferSize{ 0 };
  BufferType buffer{ nullptr, freeobj{ getFreefnHint() } };

  template <typename T, typename... Args>
  static size_t calculateSize(T&& h, Args&&... args) noexcept
  {
    return calculateSize(std::forward<T>(h)) + calculateSize(std::forward<Args>(args)...);
  }

  template <typename T>
  static size_t calculateSize(T&& h) noexcept
  {
    return h.size();
  }

  //recursion terminator
  constexpr static size_t calculateSize() { return 0; }

  template <typename T>
  static byte* inject(byte* here, T&& h) noexcept
  {
    using headerType = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    static_assert(
      std::is_base_of<BaseHeader, headerType>::value == true || std::is_same<Stack, headerType>::value == true,
      "header stack parameters are restricted to stacks and headers derived from BaseHeader");
    std::copy(h.data(), h.data() + h.size(), here);
    return here + h.size();
    // somehow could not trigger copy elision for placed construction, TODO: check out if this is possible here
    // headerType* placed = new (here) headerType(std::forward<T>(h));
    // return here + placed->size();
  }

  template <typename T, typename... Args>
  static byte* inject(byte* here, T&& h, Args&&... args) noexcept
  {
    auto alsohere = inject(here, h);
    // the type might be a stack itself, loop through headers and set the flag in the last one
    if (h.size() > 0) {
      BaseHeader* next = BaseHeader::get(here);
      while (next->flagsNextHeader) {
        next = next->next();
      }
      next->flagsNextHeader = hasNonEmptyArg(args...);
    }
    return inject(alsohere, args...);
  }

  // helper function to check if there is at least one non-empty header/stack in the argument pack
  template <typename T, typename... Args>
  static bool hasNonEmptyArg(const T& h, const Args&... args) noexcept
  {
    if (h.size() > 0) {
      return true;
    }
    return hasNonEmptyArg(args...);
  }

  template <typename T>
  static bool hasNonEmptyArg(const T& h) noexcept
  {
    if (h.size() > 0) {
      return true;
    }
    return false;
  }
};

//__________________________________________________________________________________________________
/// this 128 bit type for a header field describing the payload data type
struct printDataDescription {
  void operator()(const char* str) const;
};

using DataDescription = Descriptor<gSizeDataDescriptionString, printDataDescription>;

//__________________________________________________________________________________________________
// 32bit (4 characters) for data origin, ex. the detector or subsystem name
struct printDataOrigin {
  void operator()(const char* str) const;
};
using DataOrigin = Descriptor<gSizeDataOriginString, printDataOrigin>;

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
  using SubSpecificationType = uint64_t;

  //static data for this header type/version
  static const uint32_t sVersion;
  static const o2::header::HeaderType sHeaderType;
  static const o2::header::SerializationMethod sSerializationMethod;

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
  uint32_t reserved = 0;

  ///
  /// serialization method
  ///
  SerializationMethod payloadSerializationMethod = SerializationMethod(gInvalidToken64);

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
  explicit DataHeader(DataDescription desc, DataOrigin origin, SubSpecificationType subspec, uint64_t size); /// ctor

  DataHeader(const DataHeader&) = default;
  DataHeader& operator=(const DataHeader&) = default; //assignment

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
constexpr o2::header::DataOrigin gDataOriginAny{ "***" };
constexpr o2::header::DataOrigin gDataOriginInvalid{ "NIL" };
constexpr o2::header::DataOrigin gDataOriginFLP{ "FLP" };
constexpr o2::header::DataOrigin gDataOriginACO{ "ACO" };
constexpr o2::header::DataOrigin gDataOriginCPV{ "CPV" };
constexpr o2::header::DataOrigin gDataOriginCTP{ "CTP" };
constexpr o2::header::DataOrigin gDataOriginEMC{ "EMC" };
constexpr o2::header::DataOrigin gDataOriginFIT{ "FIT" };
constexpr o2::header::DataOrigin gDataOriginHMP{ "HMP" };
constexpr o2::header::DataOrigin gDataOriginITS{ "ITS" };
constexpr o2::header::DataOrigin gDataOriginMCH{ "MCH" };
constexpr o2::header::DataOrigin gDataOriginMFT{ "MFT" };
constexpr o2::header::DataOrigin gDataOriginMID{ "MID" };
constexpr o2::header::DataOrigin gDataOriginPHS{ "PHS" };
constexpr o2::header::DataOrigin gDataOriginTOF{ "TOF" };
constexpr o2::header::DataOrigin gDataOriginTPC{ "TPC" };
constexpr o2::header::DataOrigin gDataOriginTRD{ "TRD" };
constexpr o2::header::DataOrigin gDataOriginZDC{ "ZDC" };

//possible data types
constexpr o2::header::DataDescription gDataDescriptionAny{ "***************" };
constexpr o2::header::DataDescription gDataDescriptionInvalid{ "INVALID_DESC" };
constexpr o2::header::DataDescription gDataDescriptionRawData{ "RAWDATA" };
constexpr o2::header::DataDescription gDataDescriptionClusters{ "CLUSTERS" };
constexpr o2::header::DataDescription gDataDescriptionTracks{ "TRACKS" };
constexpr o2::header::DataDescription gDataDescriptionConfig{ "CONFIGURATION" };
constexpr o2::header::DataDescription gDataDescriptionInfo{ "INFORMATION" };
constexpr o2::header::DataDescription gDataDescriptionROOTStreamers{ "ROOT STREAMERS" };
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
///compile time checks for the basic structures
/// use hardcoded numbers as these are fundamental assumption
static_assert(sizeof(HeaderType) == 8,
              "HeaderType struct must be of size 8");
static_assert(sizeof(SerializationMethod) == 8,
              "SerializationMethod struct must be of size 8");
static_assert(sizeof(BaseHeader) == 32,
              "BaseHeader struct must be of size 32");
static_assert(sizeof(DataOrigin) == 4,
              "DataOrigin struct must be of size 4");
static_assert(sizeof(DataHeader) == 80,
              "DataHeader struct must be of size 80");
static_assert(gSizeMagicString == sizeof(BaseHeader::magicStringInt),
              "Size mismatch in magic string union");
static_assert(sizeof(BaseHeader::sMagicString) == sizeof(BaseHeader::magicStringInt),
              "Inconsitent size of global magic identifier");

} //namespace header

} //namespace o2

#endif
