// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_UTILS_RAWPARSER_H
#define FRAMEWORK_UTILS_RAWPARSER_H

/// @file   RawParser.h
/// @author Matthias Richter
/// @since  2019-10-15
/// @brief  Generic parser for consecutive raw pages

#include "Headers/RAWDataHeader.h"
#include "Framework/VariantHelpers.h" // definition of `overloaded`
#include "Framework/Logger.h"
#include <functional>
#include <memory>
#include <variant>
#include <stdexcept>
#include <string>
#include <type_traits>

// FIXME: probably moved somewhere else
namespace o2::framework
{

/// workspace for raw parser internal utilities
namespace raw_parser
{

/// specifier for printout
enum struct FormatSpec {
  Info,          // basic info: version
  TableHeader,   // table header
  Entry,         // table entry, i.e. RDH at current position
  FullTable,     // full table with header and all entiries
  FullTableInfo, // info and full table with header and all entries
};

template <typename T>
struct RDHFormatter {
  using type = T;
  static void apply(std::ostream&, type const&, FormatSpec, const char* = "")
  {
  }
};

template <>
struct RDHFormatter<header::RAWDataHeaderV7> {
  using type = header::RAWDataHeaderV7;
  static const char* sFormatString;
  static void apply(std::ostream&, type const&, FormatSpec, const char* = "");
};

template <>
struct RDHFormatter<header::RAWDataHeaderV6> {
  using type = header::RAWDataHeaderV6;
  static const char* sFormatString;
  static void apply(std::ostream&, type const&, FormatSpec, const char* = "");
};

template <>
struct RDHFormatter<header::RAWDataHeaderV5> {
  using type = header::RAWDataHeaderV5;
  static const char* sFormatString;
  static void apply(std::ostream&, type const&, FormatSpec, const char* = "");
};

template <>
struct RDHFormatter<header::RAWDataHeaderV4> {
  using type = header::RAWDataHeaderV4;
  static const char* sFormatString;
  static void apply(std::ostream&, type const&, FormatSpec, const char* = "");
};

struct RawParserHelper {
  static int sErrorMode;            // 0: no error checking, 1: print error message, 2: throw exception. To be set via O2_DPL_RAWPARSER_ERRORMODE.
  static int sCheckIncompleteHBF;   // Check if HBFs are incomplete, set to 2 to throw in case sErrorMode = 2.
  static unsigned long sErrors;     // Obviously this would need to be atomic to be fully correct, but a race condition is unlikely and would only lead to one extra log message printed.
  static unsigned long sErrorLimit; // Exponentially downscale error reporting after sErrorLimit errors.
  static unsigned long sErrorScale; // Exponentionally downscale verbosity.

  static bool checkPrintError(size_t& localCounter);
};

/// @class ConcreteRawParser
/// Raw parser implementation for a particular version of RAWDataHeader.
/// Parses a contiguous sequence of raw pages in a raw buffer.
///
/// Template parameters:
///     HeaderType    the raw data header type
///     MAX_SIZE      maximum page size
///     BOUNDS_CHECKS check buffer bounds while navigating RDHs
///
/// We expect the actual page size to be variable up to the maximum size,
/// actual size given by fields of the RAWDataHeader.
///
template <typename HeaderType, size_t MAX_SIZE, bool BOUNDS_CHECKS>
class ConcreteRawParser
{
 public:
  using buffer_type = unsigned char;
  using header_type = HeaderType;
  using self_type = ConcreteRawParser;
  static constexpr size_t max_size = MAX_SIZE;

  ConcreteRawParser() = delete;

  /// Constructor
  /// Raw buffer is provided by pointer and size, read-only access
  template <typename T>
  ConcreteRawParser(T const* buffer, size_t size)
    : mRawBuffer(reinterpret_cast<buffer_type const*>(buffer)), mSize(size)
  {
    static_assert(sizeof(T) == sizeof(buffer_type), "buffer required to be byte-type");
    if (size < sizeof(header_type)) { // TODO: should be included in sErrorMode treatment, but would need to existing behavior in detail
      std::runtime_error("buffer too small to fit at least the page header");
    }
    next();
  }
  // FIXME: add constructor for container type buffer

  /// copy constructor, we allow for copies of the parser because acces to the
  /// underlying buffer is read-only, the copy starts at the same state
  /// mainly used for the iterator
  ConcreteRawParser(ConcreteRawParser const& other) = default;

  /// Get header at current position
  header_type const& header() const
  {
    return *reinterpret_cast<header_type const*>(mPosition);
  }

  /// Get size of payload at current position
  size_t size() const
  {
    if (mPosition == mRawBuffer + mSize) {
      return 0;
    }
    header_type const& h = header();
    if (h.memorySize >= h.headerSize) {
      return h.memorySize - h.headerSize;
    }
    return max_size - h.headerSize;
  }

  /// Get size of header + payload at current position
  size_t sizeTotal() const
  {
    if (mPosition == mRawBuffer + mSize) {
      return 0;
    }
    header_type const& h = header();
    if (h.memorySize >= h.headerSize) {
      return h.memorySize;
    }
    return max_size;
  }

  /// Get pointer to payload data at current position
  buffer_type const* data() const
  {
    size_t size = this->size();
    if (size == 0) {
      return nullptr;
    }
    header_type const& h = header();
    return mPosition + h.headerSize;
  }

  /// Get pointer to raw buffer at current position
  buffer_type const* raw() const
  {
    if (mPosition < mRawBuffer + mSize) {
      return mPosition;
    }
    return nullptr;
  }

  /// Get offset of payload in the raw buffer at current position
  size_t offset() const
  {
    if (mPosition < mRawBuffer + mSize) {
      header_type const& h = header();
      return h.headerSize;
    }
    return 0;
  }

  bool checkPageInBuffer() const
  {
    if (mPosition + sizeof(header_type) > mRawBuffer + mSize) {
      return false;
    }
    header_type const& h = header();
    if (h.memorySize > MAX_SIZE || h.headerSize > MAX_SIZE) {
      return false;
    }
    size_t pageSize = h.memorySize >= h.headerSize ? h.memorySize : MAX_SIZE;
    return mPosition + pageSize <= mRawBuffer + mSize;
  }

  /// Parse the complete buffer
  /// For each page, the processor function is called with the payload buffer and size,
  /// processor has signature
  ///     void(unsigned char const*, size_t)
  template <typename Processor>
  void parse(Processor&& processor)
  {
    if (!reset()) {
      return;
    }
    if constexpr (BOUNDS_CHECKS) {
      if (RawParserHelper::sErrorMode && !checkPageInBuffer()) {
        if (RawParserHelper::sErrorMode >= 2) {
          throw std::runtime_error("Corrupt RDH - RDH parsing ran out of raw data buffer");
        }
        if (RawParserHelper::checkPrintError(mNErrors)) {
          LOG(error) << "RAWPARSER: Corrupt RDH - RDH parsing ran out of raw data buffer (" << RawParserHelper::sErrors << " total RawParser errors)";
        }
      }
    }
    // auto deleter = [](buffer_type*) {};
    do {
      processor(data(), size());
      // processor(std::unique_ptr<buffer_type, decltype(deleter)>(data(), deleter), size());
    } while (next());
  }

  /// Move to next page start
  bool next()
  {
    int lastPacketCounter = -1;
    int lastHBFPacketCounter = -1;
    unsigned int lastFEEID = -1;
    if (mPosition == nullptr) {
      mPosition = mRawBuffer;
      if (mSize == 0) {
        return false;
      }
    } else {
      auto offset = header().offsetToNext;
      if ((mPosition + offset + sizeof(header_type) > mRawBuffer + mSize) || (offset < sizeof(header_type))) {
        mPosition = mRawBuffer + mSize;
        return false;
      }
      if (RawParserHelper::sCheckIncompleteHBF) {
        lastPacketCounter = header().packetCounter;
        lastHBFPacketCounter = header().pageCnt;
        lastFEEID = header().feeId;
      }
      mPosition += offset;
    }
    if constexpr (BOUNDS_CHECKS) {
      if (RawParserHelper::sErrorMode && !checkPageInBuffer()) {
        if (RawParserHelper::sErrorMode >= 2) {
          throw std::runtime_error("Corrupt RDH - RDH parsing ran out of raw data buffer");
        }
        if (RawParserHelper::checkPrintError(mNErrors)) {
          LOG(error) << "RAWPARSER: Corrupt RDH - RDH parsing ran out of raw data buffer (" << RawParserHelper::sErrors << " total RawParser errors)";
        }
        mPosition = mRawBuffer + mSize;
        return false;
      }
    }
    if (RawParserHelper::sErrorMode) {
      if (header().version != HeaderType().version) {
        if (RawParserHelper::sErrorMode >= 2) {
          throw std::runtime_error("Corrupt RDH - Invalid RDH version");
        }
        if (RawParserHelper::checkPrintError(mNErrors)) {
          LOG(error) << "RAWPARSER: Corrupt RDH - Invalid RDH Version " << header().version << " (expected " << HeaderType().version << ") (" << RawParserHelper::sErrors << " total RawParser errors)";
        }
        mPosition = mRawBuffer + mSize;
        return false;
      }
      // FIXME: The && should be ||, since both packetCounter and pageCnt should be incremental. But currently, UL data with multiple links does not have incremental packetCounter.
      if (lastPacketCounter != -1 && lastFEEID == header().feeId && ((unsigned char)(lastPacketCounter + 1) != header().packetCounter && (unsigned short)(lastHBFPacketCounter + 1) != header().pageCnt)) {
        if (RawParserHelper::sErrorMode >= 2 && RawParserHelper::sCheckIncompleteHBF >= 2) {
          throw std::runtime_error("Incomplete HBF - jump in packet counter");
        }
        if (RawParserHelper::checkPrintError(mNErrors)) {
          LOG(error) << "RAWPARSER: Incomplete HBF - jump in packet counter " << lastPacketCounter << " to " << header().packetCounter << " (" << RawParserHelper::sErrors << " total RawParser errors)";
        }
        mPosition = mRawBuffer + mSize;
        return false;
      }
    }
    return true;
  }

  /// Reset the parser, set position to beginning of buffer
  bool reset()
  {
    mPosition = mRawBuffer;
    mNErrors = 0;
    return mSize != 0;
  }

  /// Advance position by number of steps
  /// Positions are given by RAWDataHeaders, either at fixed page size or variable length.
  /// @param step   Step number to advance, if -1 position is set beyond the buffer (end iterator)
  void advance(int step)
  {
    if (step < 0) {
      // set beyond the buffer
      mPosition = mRawBuffer + mSize;
    } else {
      while ((step-- > 0) && next()) {
      };
    }
  }

  size_t getNErrors() const
  {
    return mNErrors;
  }

  /// Comparison: instances are equal if they serve the same buffer and are in the same
  /// state, i.e. at same position
  template <typename T = self_type>
  bool operator==(T const& other) const
  {
    if constexpr (std::is_same<T, self_type>::value == true) {
      return mRawBuffer == other.mRawBuffer && mPosition == other.mPosition;
    } else {
      throw std::runtime_error(std::string("incompatible types for comparison ") + typeid(T).name() + "/" + typeid(self_type).name());
    }
  }

  void format(std::ostream& os, FormatSpec choice = FormatSpec::Entry, const char* delimiter = "\n") const
  {
    RDHFormatter<header_type>::apply(os, header(), choice, delimiter);
  }

 private:
  buffer_type const* mRawBuffer;
  buffer_type const* mPosition = nullptr;
  size_t mSize;
  size_t mNErrors = 0;
};

using V7 = header::RAWDataHeaderV7;
using V6 = header::RAWDataHeaderV6;
using V5 = header::RAWDataHeaderV5;
using V4 = header::RAWDataHeaderV4;
// FIXME v3 and v4 are basically the same with v4 defining a few more fields in the otherwise reserved parts
// needs to be defined in the header, have to check if we need to support this
// using V3 = header::RAWDataHeaderV3;

template <size_t N, bool BOUNDS_CHECKS>
using V7Parser = ConcreteRawParser<header::RAWDataHeaderV7, N, BOUNDS_CHECKS>;
template <size_t N, bool BOUNDS_CHECKS>
using V6Parser = ConcreteRawParser<header::RAWDataHeaderV6, N, BOUNDS_CHECKS>;
template <size_t N, bool BOUNDS_CHECKS>
using V5Parser = ConcreteRawParser<header::RAWDataHeaderV5, N, BOUNDS_CHECKS>;
template <size_t N, bool BOUNDS_CHECKS>
using V4Parser = ConcreteRawParser<header::RAWDataHeaderV4, N, BOUNDS_CHECKS>;
// template <size_t N, bool BOUNDS_CHECKS>
// using V3Parser = ConcreteRawParser<header::RAWDataHeaderV3, N, BOUNDS_CHECKS>;

/// Parser instance type for the raw parser main class, all supported versions of
/// RAWDataHeader are handled in a variant
template <size_t N, bool BOUNDS_CHECKS>
using ConcreteParserVariants = std::variant<V7Parser<N, BOUNDS_CHECKS>, V6Parser<N, BOUNDS_CHECKS>, V5Parser<N, BOUNDS_CHECKS>, V4Parser<N, BOUNDS_CHECKS>>;

/// create a raw parser depending on version of RAWDataHeader found at beginning of data
template <size_t PageSize, bool BOUNDS_CHECKS, typename T>
ConcreteParserVariants<PageSize, BOUNDS_CHECKS> create(T const* buffer, size_t size)
{
  // we use v5 for checking the matching version
  if (buffer == nullptr || size < sizeof(header::RAWDataHeaderV5)) {
    throw std::runtime_error("can not create RawParser: invalid buffer");
  }

  V5 const* v5 = reinterpret_cast<V5 const*>(buffer);
  if (v5->version == 5) {
    return ConcreteRawParser<V5, PageSize, BOUNDS_CHECKS>(buffer, size);
  } else if (v5->version == 7) {
    return ConcreteRawParser<V7, PageSize, BOUNDS_CHECKS>(buffer, size);
  } else if (v5->version == 6) {
    return ConcreteRawParser<V6, PageSize, BOUNDS_CHECKS>(buffer, size);
  } else if (v5->version == 4) {
    return ConcreteRawParser<V4, PageSize, BOUNDS_CHECKS>(buffer, size);
    //} else if (v5->version == 3) {
    //  return ConcreteRawParser<V3, PageSize, BOUNDS_CHECKS>(buffer, size);
  }
  throw std::runtime_error("can not create RawParser: invalid version " + std::to_string(v5->version));
}

/// iteratively walk through the available instances and parse with instance
/// specified by index
template <size_t N, typename T, typename P>
void walk_parse(T& instances, P&& processor, size_t index)
{
  if constexpr (N > 0) {
    if (index == N - 1) {
      std::get<N - 1>(instances).parse(processor);
    }

    walk_parse<N - 1>(instances, processor, index);
  }
}

template <typename U, typename T, size_t N = std::variant_size_v<T>>
U const* get_if(T& instances)
{
  if constexpr (N > 0) {
    auto* parser = std::get_if<N - 1>(&instances);
    if (parser) {
      // we are in the active instance, return header if type matches
      using parser_type = typename std::variant_alternative<N - 1, T>::type;
      using header_type = typename parser_type::header_type;
      if constexpr (std::is_same<U, header_type>::value == true) {
        return &(parser->header());
      }
    } else {
      // continue walking through instances until active one is found
      return get_if<U, T, N - 1>(instances);
    }
  }
  return nullptr;
}

} // namespace raw_parser

/// @class RawParser parser for the O2 raw data
/// O2 raw data is organized in pages of a fixed size, each page starts with the RAWDataHeader
/// The page size may actually be smaller than the maximum size, depending on the header
/// fields.
///
/// The parser class works on a contiguous sequence of raw pages in a raw buffer.
/// Multiple versions of RAWDataHeader are supported transparently and selected depending
/// on the version field of the header.
///
/// \par Usage:
///
///     // option 1: parse method
///     RawParser parser(buffer, size);
///     auto processor = [&count](auto data, size_t size) {
///       std::cout << "Processing block of size " << size << std::endl;
///     };
///     parser.parse(processor);
///
///     // option 2: iterator
///     RawParser parser(buffer, size);
///     for (auto it = parser.begin(), end = parser.end(); it != end; ++it, ++count) {
///       std::cout << "Iterating block of size " << it.size() << std::endl;
///       auto dataptr = it.data();
///     }
///
/// TODO:
/// - iterators are not independent at the moment and this can cause conflicts, this must be
///   improved
template <size_t MAX_SIZE = 8192, bool BOUNDS_CHECKS = true>
class RawParser
{
 public:
  using buffer_type = unsigned char;
  static size_t const max_size = MAX_SIZE;
  using self_type = RawParser<MAX_SIZE, BOUNDS_CHECKS>;

  RawParser() = delete;

  /// Constructor, raw buffer provided by pointer and size
  template <typename T>
  RawParser(T const* buffer, size_t size)
    : mParser(raw_parser::create<MAX_SIZE, BOUNDS_CHECKS>(buffer, size))
  {
    static_assert(sizeof(T) == sizeof(buffer_type), "buffer required to be byte-type");
  }
  // FIXME: constructor for container type

  /// Parse complete raw buffer and call processor on payload data for each page
  template <typename Processor>
  void parse(Processor&& processor)
  {
    constexpr size_t NofAlternatives = std::variant_size_v<decltype(mParser)>;
    static_assert(NofAlternatives == 4); // Change this if a new RDH version is added
    raw_parser::walk_parse<NofAlternatives>(mParser, processor, mParser.index());
    // it turned out that using a iterative function is faster than using std::visit
    // std::visit([&processor](auto& parser) { return parser.parse(processor); }, mParser);
  }

  /// Reset parser and set position to beginning of buffer
  bool reset()
  {
    return std::visit([](auto& parser) { return parser.reset(); }, mParser);
  }

  /// @struct RawDataHeaderInfo the smallest common part of all RAWDataHeader versions
  /// This struct is used as iterator value type and is a common header which can be returned
  /// for all versions of RAWDataHeader
  struct RawDataHeaderInfo {
    uint8_t version = 0;
    uint8_t headerSize = 0;
  };

  /// Iterator implementation
  /// Supports the following operations:
  /// - increment (there is no decrement, its not a bidirectional parser)
  /// - dereference operator returns @a RawDataHeaderInfo as common header
  /// - member function data() returns pointer to payload at current position
  /// - member function size() return size of payload at current position
  template <typename T, typename ParentType>
  class Iterator
  {
   public:
    using parent_type = ParentType;
    using self_type = Iterator;
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using reference = T&;
    using pointer = T*;

    Iterator() = delete;

    Iterator(parent_type parser, int start = 0)
      : mParser(parser)
    {
      std::visit([&start](auto& parser) {parser.reset(); parser.advance(start); }, mParser);
    }
    ~Iterator() = default;

    // prefix increment
    self_type& operator++()
    {
      std::visit([](auto& parser) { parser.next(); }, mParser);
      return *this;
    }
    // postfix increment
    self_type operator++(int /*unused*/)
    {
      self_type copy(*this);
      operator++();
      return copy;
    }
    // return reference
    reference operator*()
    {
      return std::visit([](auto& parser) { return static_cast<reference>(parser.header()); }, mParser);
    }
    // comparison
    bool operator==(const self_type& other) const
    {
      return std::visit([&other](auto& parser) { return std::visit([&parser](auto& otherParser) { return parser == otherParser; },
                                                                   other.mParser); },
                        mParser);
    }

    bool operator!=(const self_type& rh) const
    {
      return not operator==(rh);
    }

    /// get pointer to raw block at current position, rdh starts here
    buffer_type const* raw() const
    {
      return std::visit([](auto& parser) { return parser.raw(); }, mParser);
    }

    /// get pointer to payload at current position
    buffer_type const* data() const
    {
      return std::visit([](auto& parser) { return parser.data(); }, mParser);
    }

    /// offset of payload at current position
    size_t offset() const
    {
      return std::visit([](auto& parser) { return parser.offset(); }, mParser);
    }

    /// get size of payload at current position
    size_t size() const
    {
      return std::visit([](auto& parser) { return parser.size(); }, mParser);
    }

    /// get size of payload + header at current position
    size_t sizeTotal() const
    {
      return std::visit([](auto& parser) { return parser.sizeTotal(); }, mParser);
    }

    /// get header as specific type
    /// Normal usage is get_if<T>() but in some rare cases the type can also be passed by parameter
    /// get_if((T*)nullptr), the parameter is ignored
    /// @return pointer to header of the specified type, or nullptr if type does not match to actual type
    template <typename U>
    U const* get_if(U const* = nullptr) const
    {
      return raw_parser::get_if<U>(mParser);
    }

    friend std::ostream& operator<<(std::ostream& os, self_type const& it)
    {
      std::visit([&os](auto& parser) { return parser.format(os, raw_parser::FormatSpec::Entry, ""); }, it.mParser);
      return os;
    }

   private:
    parent_type mParser;
  };

  // only define the const_iterator because the parser will allow read-only access
  using const_iterator = Iterator<RawDataHeaderInfo const, raw_parser::ConcreteParserVariants<MAX_SIZE, BOUNDS_CHECKS>>;

  const_iterator begin() const
  {
    return const_iterator(mParser);
  }

  const_iterator end() const
  {
    return const_iterator(mParser, -1);
  }

  friend std::ostream& operator<<(std::ostream& os, self_type const& parser)
  {
    std::visit([&os](auto& parser) { return parser.format(os, raw_parser::FormatSpec::Info, "\n"); }, parser.mParser);
    std::visit([&os](auto& parser) { return parser.format(os, raw_parser::FormatSpec::TableHeader); }, parser.mParser);
    // FIXME: need to decide what kind of information we want to have in the printout
    // for the moment its problematic, because the parser has only one variable determining the position and all
    // iterators work with the same instance which is asking for conflicts
    // this needs to be changed in order to have fully independent iterators over the same constant buffer
    // for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    //  os << "\n" << it;
    //}
    return os;
  }

  size_t getNErrors() const
  {
    return std::visit([](auto& parser) { return parser.getNErrors(); }, mParser);
  }

  static void setCheckIncompleteHBF(bool v)
  {
    raw_parser::RawParserHelper::sCheckIncompleteHBF = v;
  }

 private:
  raw_parser::ConcreteParserVariants<MAX_SIZE, BOUNDS_CHECKS> mParser;
};

} // namespace o2::framework

#endif // FRAMEWORK_UTILS_RAWPARSER_H
