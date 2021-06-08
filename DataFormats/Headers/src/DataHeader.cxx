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

#include "Headers/DataHeader.h"
#include <cstdio>  // printf
#include <cstring> // strncpy

//storage for BaseHeader static members, all invalid
const uint32_t o2::header::BaseHeader::sVersion = o2::header::gInvalidToken32;
const o2::header::HeaderType o2::header::BaseHeader::sHeaderType = o2::header::gInvalidToken64;
const o2::header::SerializationMethod o2::header::BaseHeader::sSerializationMethod = o2::header::gInvalidToken64;

using namespace o2::header;

//__________________________________________________________________________________________________
bool o2::header::BaseHeader::sanityCheck(uint32_t expectedVersion) const
{
  if (this->headerVersion != expectedVersion) {
    std::string errmsg = "header of type " + this->description.as<std::string>() + " with invalid ";
    errmsg += "version: " + std::to_string(this->headerVersion) + " (expected " + std::to_string(expectedVersion) + ")";
    // for the moment we throw, there is no support for multiple versions of a particular header
    // so we better spot non matching header stacks early, we migh change this later
    throw std::runtime_error(errmsg);
    return false;
  }
  return true;
}

//__________________________________________________________________________________________________
void o2::header::BaseHeader::throwInconsistentStackError() const
{
  throw std::runtime_error("inconsistent header stack, no O2 header at expected offset " + std::to_string(this->headerSize) + "for header of type " + this->description.as<std::string>());
}

//__________________________________________________________________________________________________
bool o2::header::DataHeader::operator==(const DataOrigin& that) const
{
  return (that == gDataOriginAny ||
          that == dataOrigin);
}

//__________________________________________________________________________________________________
bool o2::header::DataHeader::operator==(const DataDescription& that) const
{
  return ((that.itg[0] == gDataDescriptionAny.itg[0] &&
           that.itg[1] == gDataDescriptionAny.itg[1]) ||
          (that.itg[0] == dataDescription.itg[0] &&
           that.itg[1] == dataDescription.itg[1]));
}

//__________________________________________________________________________________________________
bool o2::header::DataHeader::operator==(const SerializationMethod& that) const
{
  return (that == gSerializationMethodAny ||
          that == payloadSerializationMethod);
}

//__________________________________________________________________________________________________
bool o2::header::DataHeader::operator==(const DataHeader& that) const
{
  return (magicStringInt == that.magicStringInt &&
          dataOrigin == that.dataOrigin &&
          dataDescription == that.dataDescription &&
          subSpecification == that.subSpecification);
}

//__________________________________________________________________________________________________
o2::header::DataIdentifier::DataIdentifier()
  : dataDescription(), dataOrigin()
{
}

//__________________________________________________________________________________________________
bool o2::header::DataIdentifier::operator==(const DataIdentifier& other) const
{
  if (other.dataOrigin != gDataOriginAny && dataOrigin != other.dataOrigin) {
    return false;
  }
  if (other.dataDescription != gDataDescriptionAny &&
      dataDescription != other.dataDescription) {
    return false;
  }
  return true;
}

//__________________________________________________________________________________________________
void o2::header::hexDump(const char* desc, const void* voidaddr, size_t len, size_t max)
{
  size_t i;
  unsigned char buff[17]; // stores the ASCII data
  memset(&buff[0], '\0', 17);
  const std::byte* addr = reinterpret_cast<const std::byte*>(voidaddr);

  // Output description if given.
  if (desc != nullptr) {
    printf("%s, ", desc);
  }
  printf("%zu bytes:", len);
  if (max > 0 && len > max) {
    len = max; //limit the output if requested
    printf(" output limited to %zu bytes\n", len);
  } else {
    printf("\n");
  }

  // In case of null pointer addr
  if (addr == nullptr) {
    printf("  nullptr, size: %zu\n", len);
    return;
  }

  // Process every byte in the data.
  for (i = 0; i < len; i++) {
    // Multiple of 16 means new line (with line offset).
    if ((i % 16) == 0) {
      // Just don't print ASCII for the zeroth line.
      if (i != 0) {
        printf("  %s\n", buff);
      }

      // Output the offset.
      //printf ("  %04x ", i);
      printf("  %p ", &addr[i]);
    }

    // Now the hex code for the specific character.
    printf(" %02x", (char)addr[i]);

    // And store a printable ASCII character for later.
    if (((char)addr[i] < 0x20) || ((char)addr[i] > 0x7e)) {
      buff[i % 16] = '.';
    } else {
      buff[i % 16] = (char)addr[i];
    }
    buff[(i % 16) + 1] = '\0';
    fflush(stdout);
  }

  // Pad out last line if not exactly 16 characters.
  while ((i % 16) != 0) {
    printf("   ");
    fflush(stdout);
    i++;
  }

  // And print the final ASCII bit.
  printf("  %s\n", buff);
  fflush(stdout);
}
