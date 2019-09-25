// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALGORITHM_PARSER_H
#define ALGORITHM_PARSER_H

/// @file   Parser.h
/// @author Matthias Richter
/// @since  2017-09-20
/// @brief  Utilities for parsing of data sequences

#include <functional>
#include <vector>

namespace o2
{

namespace algorithm
{

/// helper function returning size of type with a specialization for
/// void returning 0
template <typename T>
struct typesize {
  static const size_t size = sizeof(T);
};
// specialization for void
template <>
struct typesize<void> {
  static const size_t size = 0;
};

/**
 * @class ForwardParser
 * Parser for a sequence of frames with header, trailer and variable payload.
 * The size is expected to be part of the header.
 *
 * Trailer type can be void, which is also the default template parameter. That
 * allows to define a frame consisting of only header and data.
 *
 * Usage:
 * <pre>
 *   using SomeParser = ForwardParser<SomeHeaderType, SomeTrailerType>;
 *   SomeParser parser;
 *   std::vector<typename SomeParser::FrameInfo> frames;
 *   parser.parse(ptr, size,
 *                [] (const typename SomeParser::HeaderType& h) {
 *                  // check the header
 *                  return true;
 *                },
 *                [] (const typename SomeParser::TrailerType& t) {
 *                  // check the trailer
 *                  return true;
 *                },
 *                [] (const typename SomeParser::HeaderType& h) {
 *                  // get the size of the frame including payload
 *                  // and header and trailer size, e.g. payload size
 *                  // from a header member
 *                  return h.payloadSize + SomeParser::totalOffset;
 *                },
 *                [&frames] (typename SomeParser::FrameInfo& info) {
 *                  frames.emplace_back(info);
 *                  return true;
 *                }
 *                )
 *
 *   // a reduced version without trailer check callback
 *   using SomeParser = ForwardParser<SomeHeaderType>;
 *   SomeParser parser;
 *   std::vector<typename SomeParser::FrameInfo> frames;
 *   parser.parse(ptr, size,
 *                [] (const typename SomeParser::HeaderType& h) {
 *                  // check the header
 *                  return true;
 *                },
 *                [] (const typename SomeParser::HeaderType& h) {
 *                  // get the size of the frame including payload
 *                  // and header and trailer size, e.g. payload size
 *                  // from a header member
 *                  return h.payloadSize + SomeParser::totalOffset;
 *                },
 *                [&frames] (typename SomeParser::FrameInfo& info) {
 *                  frames.emplace_back(info);
 *                  return true;
 *                }
 *                )
 * </pre>
 */
template <typename HeaderT,
          typename TrailerT = void>
class ForwardParser
{
 public:
  using HeaderType = HeaderT;
  using TrailerType = TrailerT;
  using PayloadType = unsigned char;

  /// @struct FrameInfo
  /// a compound of header, data, and trailer
  struct FrameInfo {
    using PtrT = const PayloadType*;

    const HeaderType* header = nullptr;
    const TrailerType* trailer = nullptr;
    PtrT payload = nullptr;
    size_t length = 0;
  };

  /// the length offset due to header
  static const size_t headOffset = typesize<HeaderType>::size;
  /// the length offset due to trailer
  static const size_t tailOffset = typesize<TrailerType>::size;
  /// total length offset due to header and trailer
  static const size_t totalOffset = headOffset + tailOffset;

  /// alias for callback checking the header, return true if the object
  /// is a valid header
  using CheckHeaderFct = std::function<bool(const HeaderType&)>;

  /// alias for the argument type to be used in the CheckTrailer function
  /// have to forward to a valid type in case of void TrailerType in order
  /// to allow passing by reference
  using CheckTrailerFctArgumentT = typename std::conditional<
    !std::is_void<TrailerType>::value, TrailerType, int>::type;

  /// alias for callback checking the trailer, takes reference to trailer
  /// object if TrailerType is a valid type, no argument otherwise
  template <typename U>
  using CheckTrailerFct = typename std::conditional<
    !std::is_void<U>::value,
    std::function<bool(const CheckTrailerFctArgumentT&)>,
    std::function<bool()>>::type;

  /// alias for callback to get the complete frame size including header,
  /// trailer and the data
  using GetFrameSizeFct = std::function<size_t(const HeaderType&)>;

  /// function callback to insert/handle one frame into, sequentially called
  /// for all frames if the whole block has a valid format
  using InsertFct = std::function<bool(FrameInfo&)>;

  /// Parse buffer of size bufferSize, requires callbacks to check header
  /// trailer, the frame size, and insert callback to handle a FrameInfo
  /// object.
  template <typename InputType>
  int parse(const InputType* buffer, size_t bufferSize,
            CheckHeaderFct checkHeader,
            CheckTrailerFct<TrailerType> checkTrailer,
            GetFrameSizeFct getFrameSize,
            InsertFct insert)
  {
    static_assert(sizeof(InputType) == 1,
                  "ForwardParser currently only supports byte type buffer");
    if (buffer == nullptr || bufferSize == 0)
      return 0;
    size_t position = 0;
    std::vector<FrameInfo> frames;
    do {
      FrameInfo entry;

      // check the header
      if (sizeof(HeaderType) + position > bufferSize)
        break;
      entry.header = reinterpret_cast<const HeaderType*>(buffer + position);
      if (!checkHeader(*entry.header))
        break;

      // extract frame size from header, this is expected to be the
      // total frome size including header, payload and optional trailer
      auto frameSize = getFrameSize(*entry.header);
      if (frameSize + position > bufferSize)
        break;

      // payload starts right after the header
      entry.payload = reinterpret_cast<typename FrameInfo::PtrT>(entry.header + 1);
      entry.length = frameSize - totalOffset;

      // optionally extract and check trailer
      if (tailOffset > 0) {
        entry.trailer = nullptr;
      } else {
        auto trailerStart = buffer + position + frameSize - tailOffset;
        entry.trailer = reinterpret_cast<const TrailerType*>(trailerStart);
        if (!CheckTrailer(entry, checkTrailer))
          break;
      }

      // store the extracted frame info and continue with remaining buffer
      frames.emplace_back(entry);
      position += frameSize;
    } while (position < bufferSize);

    if (position == bufferSize) {
      // frames found and format consistent, insert entries to target
      // Note: the complete block must be consistent
      for (auto entry : frames) {
        if (!insert(entry))
          break;
      }
      return frames.size();
    } else if (frames.size() == 0) {
      // no frames found at all, the buffer does not contain any
      return 0;
    }

    // format error detected
    // TODO: decide about error policy
    return -1;
  }

  /// Parse buffer of size bufferSize, specialization skipping the trailer
  /// check, e.g. when its type is void, or when the integrity of the trailer
  /// is not relevant. Requires callbacks to check header, frame size, and
  /// insert callback to handle a FrameInfo object.
  template <typename InputType, typename U = TrailerType>
  typename std::enable_if<std::is_void<U>::value, int>::type
    parse(const InputType* buffer, size_t bufferSize,
          CheckHeaderFct checkHeader,
          GetFrameSizeFct getFrameSize,
          InsertFct insert)
  {
    auto checkTrailer = []() { return true; };
    return parse(buffer, bufferSize, checkHeader, checkTrailer, getFrameSize, insert);
  }

 private:
  /// internal function to check the trailer, distinguishes void and non-void
  /// trailer type.
  template <typename U = TrailerType>
  typename std::enable_if<!std::is_void<U>::value, bool>::type
    CheckTrailer(const FrameInfo& entry, CheckTrailerFct<TrailerType>& checkTrailer) const
  {
    return checkTrailer(*entry.trailer);
  }

  template <typename U = TrailerType>
  typename std::enable_if<std::is_void<U>::value, bool>::type
    CheckTrailer(const FrameInfo&, CheckTrailerFct<TrailerType>&) const
  {
    return true;
  }
};

/**
 * @class ReverseParser
 * Parser for a sequence of frames with header, trailer and variable payload.
 * The size is expected to be part of the trailer, the parsing is thus in
 * reverse direction. Also the insert callback is called with the entries
 * starting form the end of the buffer.
 * TODO: an easy extension can be to reverse the order of the inserts, meaning
 * that the entries are read from the beginning.
 *
 * Usage:
 * <pre>
 *   using SomeParser = ReverseParser<SomeHeaderType, SomeTrailerType>;
 *   SomeParser parser;
 *   std::vector<typename SomeParser::FrameInfo> frames;
 *   parser.parse(ptr, size,
 *                [] (const typename SomeParser::HeaderType& h) {
 *                  // check the header
 *                  return true;
 *                },
 *                [] (const typename SomeParser::TrailerType& t) {
 *                  // check the trailer
 *                  return true;
 *                },
 *                [] (const typename SomeParser::TrailerType& t) {
 *                  // get the size of the frame including payload
 *                  // and header and trailer size, e.g. payload size
 *                  // from a trailer member
 *                  return t.payloadSize + SomeParser::totalOffset;
 *                },
 *                [&frames] (typename SomeParser::FrameInfo& info) {
 *                  frames.emplace_back(info);
 *                  return true;
 *                }
 *                )
 * </pre>
 */
template <typename HeaderT, typename TrailerT>
class ReverseParser
{
 public:
  using HeaderType = HeaderT;
  using TrailerType = TrailerT;
  using PayloadType = unsigned char;

  /// @struct FrameInfo a compound of header, data, and trailer
  struct FrameInfo {
    using PtrT = const PayloadType*;

    const HeaderType* header = nullptr;
    const TrailerType* trailer = nullptr;
    PtrT payload = nullptr;
    size_t length = 0;
  };
  /// the length offset due to header
  static const size_t headOffset = typesize<HeaderType>::size;
  /// the length offset due to trailer
  static const size_t tailOffset = typesize<TrailerType>::size;
  /// total length offset due to header and trailer
  static const size_t totalOffset = headOffset + tailOffset;

  /// alias for callback checking the header, return true if the object
  /// is a valid header
  using CheckHeaderFct = std::function<bool(const HeaderType&)>;
  /// alias for callback checking the trailer
  using CheckTrailerFct = std::function<bool(const TrailerType&)>;
  /// alias for callback to get the complete frame size including header,
  /// trailer and the data
  using GetFrameSizeFct = std::function<size_t(const TrailerType&)>;
  /// function callback to insert/handle one frame into, sequentially called
  /// for all frames if the whole block has a valid format
  using InsertFct = std::function<bool(const FrameInfo&)>;

  /// Parse buffer of size bufferSize, requires callbacks to check header
  /// trailer, the frame size, and insert callback to handle a FrameInfo
  /// object.
  template <typename InputType>
  int parse(const InputType* buffer, size_t bufferSize,
            CheckHeaderFct checkHeader,
            CheckTrailerFct checkTrailer,
            GetFrameSizeFct getFrameSize,
            InsertFct insert)
  {
    static_assert(sizeof(InputType) == 1,
                  "ReverseParser currently only supports byte type buffer");
    if (buffer == nullptr || bufferSize == 0)
      return 0;
    auto position = bufferSize;
    std::vector<FrameInfo> frames;
    do {
      FrameInfo entry;

      // start from end, extract and check trailer
      if (sizeof(TrailerType) > position)
        break;
      entry.trailer = reinterpret_cast<const TrailerType*>(buffer + position - sizeof(TrailerType));
      if (!checkTrailer(*entry.trailer))
        break;

      // get the total frame size
      auto frameSize = getFrameSize(*entry.trailer);
      if (frameSize > position)
        break;

      // extract and check header
      auto headerStart = buffer + position - frameSize;
      entry.header = reinterpret_cast<const HeaderType*>(headerStart);
      if (!checkHeader(*entry.header))
        break;

      // payload immediately after header
      entry.payload = reinterpret_cast<typename FrameInfo::PtrT>(entry.header + 1);
      entry.length = frameSize - sizeof(HeaderType) - sizeof(TrailerType);
      frames.emplace_back(entry);
      position -= frameSize;
    } while (position > 0);

    if (position == 0) {
      // frames found and format consistent, the complete block must be consistent
      for (auto entry : frames) {
        if (!insert(entry))
          break;
      }
      return frames.size();
    } else if (frames.size() == 0) {
      // no frames found at all, the buffer does not contain any
      return 0;
    }

    // format error detected
    // TODO: decide about error policy
    return -1;
  }
};

} // namespace algorithm

} // namespace o2

#endif // ALGORITHM_PARSER_H
