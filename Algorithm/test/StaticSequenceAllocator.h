// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   StaticSequenceAllocator.h
/// @author Matthias Richter, based on work by Mikolaj Krzewicki
/// @since  2017-09-21
/// @brief  An allocator for static sequences of object types

namespace o2
{
namespace algorithm
{

/**
 * Helper struct to define a composite element from a header, some payload
 * and a trailer
 */
template <typename HeaderT, typename TrailerT = void>
struct Composite {
  using HeaderType = HeaderT;
  using TrailerType = TrailerT;
  size_t compositeLength = 0;
  size_t trailerLength = 0;
  size_t dataLength = 0;

  template <size_t N,
            typename U = TrailerType>
  constexpr Composite(const HeaderType h, const char (&d)[N],
                      typename std::conditional<!std::is_void<U>::value, const TrailerType, int>::type t,
                      typename std::enable_if<!std::is_void<U>::value>::type* = nullptr)
    : header(h), data(d), trailer(t)
  {
    dataLength = N;
    trailerLength = sizeof(TrailerType);
    compositeLength = sizeof(HeaderType) + dataLength + trailerLength;
  }

  template <size_t N,
            typename U = TrailerType>
  constexpr Composite(const HeaderType& h, const char (&d)[N],
                      typename std::enable_if<std::is_void<U>::value>::type* = nullptr)
    : header(h), data(d)
  {
    dataLength = N;
    trailerLength = 0;
    compositeLength = sizeof(HeaderType) + dataLength + trailerLength;
  }

  constexpr size_t getLength() const noexcept
  {
    return compositeLength;
  }

  constexpr size_t getDataLength() const noexcept
  {
    return dataLength;
  }

  template <typename BufferT>
  constexpr size_t insert(BufferT* buffer) const noexcept
  {
    static_assert(sizeof(BufferT) == 1, "buffer required to be of byte-type");
    size_t length = 0;
    memcpy(buffer + length, &header, sizeof(HeaderType));
    length += sizeof(HeaderType);
    memcpy(buffer + length, data, dataLength);
    length += dataLength;
    if (trailerLength > 0) {
      memcpy(buffer + length, &trailer, trailerLength);
      length += trailerLength;
    }
    return length;
  }

  const HeaderType header;
  const char* data = nullptr;
  typename std::conditional<!std::is_void<TrailerType>::value, const TrailerType, int>::type trailer;
};

/// recursively calculate the length of the sequence
/// object types are fixed at compile time and so is the total length of the
/// sequence. The function is recursively invoked for all arguments of the
// variable list
template <typename T, typename... TArgs>
constexpr size_t sequenceLength(const T& first, const TArgs... args) noexcept
{
  return sequenceLength(first) + sequenceLength(args...);
}

/// template secialization of sequence length calculation for one argument,
/// this is also the terminating instance for the last argument of the recursive
/// invocation of the function template.
template <typename T>
constexpr size_t sequenceLength(const T& first) noexcept
{
  return first.getLength();
}

/// recursive insert of variable number of objects
template <typename BufferT, typename T, typename... TArgs>
constexpr size_t sequenceInsert(BufferT* buffer, const T& first, const TArgs... args) noexcept
{
  static_assert(sizeof(BufferT) == 1, "buffer required to be of byte-type");
  auto length = sequenceInsert(buffer, first);
  length += sequenceInsert(buffer + length, args...);
  return length;
}

/// terminating template specialization, i.e. for the last element
template <typename BufferT, typename T>
constexpr size_t sequenceInsert(BufferT* buffer, const T& element) noexcept
{
  // TODO: make a general algorithm, at the moment this serves the
  // Composite class as a special case
  return element.insert(buffer);
}

/**
 * Allocator for a buffer of a static sequence of multiple objects.
 *
 * The sequence of object types is fixed at compile time and given as
 * a variable list of arguments to the constructor. The data of the objects
 * is runtime dependent.
 *
 * TODO: probably the Composite struct needs to be reworked to allow this
 * allocator to be more general
 */
struct StaticSequenceAllocator {
  using value_type = unsigned char;
  using BufferType = std::unique_ptr<value_type[]>;

  BufferType buffer;
  size_t bufferSize;

  size_t size() const { return bufferSize; }

  StaticSequenceAllocator() = delete;

  template <typename... Targs>
  StaticSequenceAllocator(Targs... args)
  {
    bufferSize = sequenceLength(args...);
    buffer = std::make_unique<value_type[]>(bufferSize);
    sequenceInsert(buffer.get(), args...);
  }
};

} // namespace algorithm
} // namespace o2
