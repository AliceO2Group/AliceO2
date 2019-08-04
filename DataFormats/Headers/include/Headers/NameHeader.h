// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief O2 data header classes, header example
///
/// origin: CWG4
/// @author Mikolaj Krzewicki, mkrzewic@cern.ch
/// @author Matthias Richter, Matthias.Richter@cern.ch
/// @author David Rohr, drohr@cern.ch

#ifndef NAMEHEADER_H
#define NAMEHEADER_H

#include <string>
#include "Headers/DataHeader.h"

namespace o2
{
namespace header
{

/// @struct NameHeader
/// @brief an example data header containing a name of an object as a null terminated char arr.
/// this is a template! at instantiation the template parameter determines the
/// size of the held string array.
/// a caveat with decoding is (if length of string is not known) you have to use header::get<NameHeader<0>>(buffer)
/// to get it out of a buffer. May improve in the future if enough people complain.
/// If the compiler decides to pad the end of struct, the length of padding is stored in the last byte
/// @ingroup aliceo2_dataformats_dataheader
template <size_t N>
struct NameHeader : public BaseHeader {
  static const uint32_t sVersion;
  static const o2::header::HeaderType sHeaderType;
  static const o2::header::SerializationMethod sSerializationMethod;
  NameHeader() : BaseHeader(sizeof(NameHeader), sHeaderType, sSerializationMethod, sVersion), name()
  {
    // set length of padding in the last byte
    uint8_t* lastByte = reinterpret_cast<uint8_t*>(this) + sizeof(NameHeader) - 1;
    *lastByte =
      reinterpret_cast<const uint8_t*>(this) + sizeof(NameHeader) - reinterpret_cast<const uint8_t*>(&name[N]);
    // zero string
    memset(&name[0], '\0', N);
  }

  NameHeader(std::string in) : BaseHeader(sizeof(NameHeader), sHeaderType, sSerializationMethod, sVersion), name()
  {
    // set length of padding in the last byte
    uint8_t* lastByte = reinterpret_cast<uint8_t*>(this) + sizeof(NameHeader) - 1;
    *lastByte =
      reinterpret_cast<const uint8_t*>(this) + sizeof(NameHeader) - reinterpret_cast<const uint8_t*>(&name[N]);
    // here we actually want a null terminated string
    strncpy(name, in.c_str(), N);
    name[N - 1] = '\0';
  }

  const char* getName() const { return name; }
  size_t getNameLength() const
  {
    const uint8_t* lastByte = reinterpret_cast<const uint8_t*>(this) + size() - 1;
    return (lastByte - reinterpret_cast<const uint8_t*>(name)) - *lastByte + 1;
  }

 private:
  char name[N];
};

template <size_t N>
const o2::header::HeaderType NameHeader<N>::sHeaderType = "NameHead";

// dirty trick to always have access to the headertypeID of a templated header type
// so when decoding an unknown length name use length 0
// TODO: find out if this can be done in a nicer way + is this realy necessary?
template <>
const o2::header::HeaderType NameHeader<0>::sHeaderType;

template <size_t N>
const SerializationMethod NameHeader<N>::sSerializationMethod = gSerializationMethodNone;

template <size_t N>
const uint32_t NameHeader<N>::sVersion = 1;
} // namespace header
} // namespace o2

#endif // NAMEHEADER_H
