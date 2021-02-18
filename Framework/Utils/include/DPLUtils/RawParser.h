// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

/// @class ConcreteRawParser
/// Raw parser implementation for a particular version of RAWDataHeader.
/// Parses a contiguous sequence of raw pages in a raw buffer.
///
/// Template parameters:
///     HeaderType  the raw data header type
///     MAX_SIZE    maximum page size
///
/// We expect the actual page size to be variable up to the maximum size,
/// actual size given by fields of the RAWDataHeader.
///
template <typename HeaderType, size_t MAX_SIZE>
class ConcreteRawParser
{
 public:
  using buffer_type = unsigned char;
  using header_type = HeaderType;
  using self_type = ConcreteRawParser;
  size_t max_size = MAX_SIZE;

  ConcreteRawParser() = delete;

  /// Constructor
  /// Raw buffer is provided by pointer and size, read-only access
  template <typename T>
  ConcreteRawParser(T const* buffer, size_t size)
    : mRawBuffer(reinterpret_cast<buffer_type const*>(buffer)), mSize(size)
  {
    static_assert(sizeof(T) == sizeof(buffer_type), "buffer required to be byte-type");
    if (size < sizeof(header_type)) {
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

  /// Get length of payload at current position
  size_t length() const
  {
    if (mPosition == mRawBuffer + mSize) {
      return 0;
    }
    header_type const& h = header();
    // FIXME: block length disappeared in V5, check what can be used
    return max_size - h.headerSize;
  }

  /// Get pointer to payload data at current position
  buffer_type const* data() const
  {
    size_t len = length();
    if (len == 0) {
      return nullptr;
    }
    header_type const& h = header();
    if (mPosition + len + h.headerSize > mRawBuffer + mSize) {
      throw std::runtime_error("not enough data at position " + std::to_string(mPosition - mRawBuffer));
    }
    return mPosition + h.headerSize;
  }

  /// Parse the complete buffer
  /// For each page, the processor function is called with the payload buffer and size,
  /// processor has signature
  ///     void(unsigned char const*, size_t)
  template <typename Processor>
  void parse(Processor&& processor)
  {
    reset();
    //auto deleter = [](buffer_type*) {};
    do {
      processor(data(), length());
      //processor(std::unique_ptr<buffer_type, decltype(deleter)>(data(), deleter), size());
    } while (next());
  }

  /// Move to next page start
  bool next()
  {
    if (mPosition == nullptr) {
      mPosition = mRawBuffer;
    } else {
      auto offset = header().offsetToNext;
      if ((mPosition + offset + sizeof(header_type) > mRawBuffer + mSize) || (offset < sizeof(header_type))) {
        // FIXME: should check if there is unread data left of if we just reached
        // the end of the buffer by parsing all pages
        mPosition = mRawBuffer + mSize;
        return false;
      }
      mPosition += offset;
    }
    // FIXME: check page header validity: version, size
    return true;
  }

  /// Reset the parser, set position to beginning of buffer
  void reset()
  {
    mPosition = mRawBuffer;
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
};

using V5 = header::RAWDataHeaderV5;
using V4 = header::RAWDataHeaderV4;
// FIXME v3 and v4 are basically the same with v4 defining a few more fields in the otherwise reserved parts
// needs to be defined in the header, have to check if we need to support this
//using V3 = header::RAWDataHeaderV3;

template <size_t N>
using V5Parser = ConcreteRawParser<header::RAWDataHeaderV5, N>;
template <size_t N>
using V4Parser = ConcreteRawParser<header::RAWDataHeaderV4, N>;
//template <size_t N>
//using V3Parser = ConcreteRawParser<header::RAWDataHeaderV3, N>;

/// Parser instance type for the raw parser main class, all supported versions of
/// RAWDataHeader are handled in a variant
template <size_t N>
using ConcreteParserVariants = std::variant<V5Parser<N>, V4Parser<N>>;

/// create a raw parser depending on version of RAWDataHeader found at beginning of data
template <size_t PageSize, typename T>
ConcreteParserVariants<PageSize> create(T const* buffer, size_t size)
{
  // we use v5 for checking the matching version
  if (buffer == nullptr || size < sizeof(header::RAWDataHeaderV5)) {
    throw std::runtime_error("can not create RawParser: invalid buffer");
  }

  V5 const* v5 = reinterpret_cast<V5 const*>(buffer);
  if (v5->version == 5) {
    return ConcreteRawParser<V5, PageSize>(buffer, size);
  } else if (v5->version == 4) {
    return ConcreteRawParser<V4, PageSize>(buffer, size);
    //} else if (v5->version == 3) {
    //  return ConcreteRawParser<V3, PageSize>(buffer, size);
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
///     auto processor = [&count](auto data, size_t length) {
///       std::cout << "Processing block of length " << length << std::endl;
///     };
///     parser.parse(processor);
///
///     // option 2: iterator
///     RawParser parser(buffer, size);
///     for (auto it = parser.begin(), end = parser.end(); it != end; ++it, ++count) {
///       std::cout << "Iterating block of length " << it.length() << std::endl;
///       auto dataptr = it.data();
///     }
///
/// TODO:
/// - iterators are not independent at the moment and this can cause conflicts, this must be
///   improved
template <size_t MAX_SIZE = 8192>
class RawParser
{
 public:
  using buffer_type = unsigned char;
  size_t max_size = MAX_SIZE;
  using self_type = RawParser<MAX_SIZE>;

  RawParser() = delete;

  /// Constructor, raw buffer provided by pointer and size
  template <typename T>
  RawParser(T const* buffer, size_t size)
    : mParser(raw_parser::create<MAX_SIZE>(buffer, size))
  {
    static_assert(sizeof(T) == sizeof(buffer_type), "buffer required to be byte-type");
  }
  // FIXME: constructor for container type

  /// Parse complete raw buffer and call processor on payload data for each page
  template <typename Processor>
  void parse(Processor&& processor)
  {
    constexpr size_t NofAlternatives = std::variant_size_v<decltype(mParser)>;
    static_assert(NofAlternatives == 2);
    raw_parser::walk_parse<NofAlternatives>(mParser, processor, mParser.index());
    // it turned out that using a iterative function is faster than using std::visit
    //std::visit([&processor](auto& parser) { return parser.parse(processor); }, mParser);
  }

  /// Reset parser and set position to beginning of buffer
  void reset()
  {
    std::visit([](auto& parser) { parser.reset(); }, mParser);
  }

  /// @struct RawDataHeaderInfo the smallest common part of all RAWDataHeader versions
  /// This struct is used as iterator value type and is a common header which can be returned
  /// for all versions of RAWDataHeader
  struct RawDataHeaderInfo {
    uint8_t version = 0;
    uint8_t headerSize = 0;
  };

  template <typename T>
  using IteratorBase = std::iterator<std::forward_iterator_tag, T>;

  /// Iterator implementation
  /// Supports the following operations:
  /// - increment (there is no decrement, its not a bidirectional parser)
  /// - dereference operator returns @a RawDataHeaderInfo as common header
  /// - member function data() returns pointer to payload at current position
  /// - member function length() return size of payload at current position
  template <typename T, typename ParentType>
  class Iterator : public IteratorBase<T>
  {
   public:
    using parent_type = ParentType;
    using self_type = Iterator;
    using value_type = typename IteratorBase<T>::value_type;
    using reference = typename IteratorBase<T>::reference;
    using pointer = typename IteratorBase<T>::pointer;

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

    /// get pointer to payload at current position
    buffer_type const* data() const
    {
      return std::visit([](auto& parser) { return parser.data(); }, mParser);
    }

    /// get length of payload at current position
    size_t length() const
    {
      return std::visit([](auto& parser) { return parser.length(); }, mParser);
    }

    /// get header as specific type
    /// @return pointer to header of the specified type, or nullptr if type does not match to actual type
    template <typename U>
    U const* get_if() const
    {
      return raw_parser::get_if<U>(mParser);
    }

    friend std::ostream& operator<<(std::ostream& os, self_type const& it)
    {
      std::visit([&os](auto& parser) { return parser.format(os); }, it.mParser);
      return os;
    }

   private:
    parent_type mParser;
  };

  // only define the const_iterator because the parser will allow read-only access
  using const_iterator = Iterator<RawDataHeaderInfo const, raw_parser::ConcreteParserVariants<MAX_SIZE>>;

  const_iterator begin()
  {
    return const_iterator(mParser);
  }

  const_iterator end()
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
    //for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    //  os << "\n" << it;
    //}
    return os;
  }

 private:
  raw_parser::ConcreteParserVariants<MAX_SIZE> mParser;
};

} // namespace o2::framework

#endif // FRAMEWORK_UTILS_RAWPARSER_H
