// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_SHA1_H_
#define O2_FRAMEWORK_SHA1_H_

/*
   Based on SHA-1 in C
   By Steve Reid <steve@edmweb.com>
   100% Public Domain
 */

#include <cstdint>

namespace o2::framework::internal
{

typedef struct
{
  uint32_t state[5];
  uint32_t count[2];
  unsigned char buffer[64];
} SHA1_CTX;

void SHA1Transform(uint32_t state[5], const unsigned char buffer[64]);
void SHA1Init(SHA1_CTX* context);
void SHA1Update(SHA1_CTX* context, const unsigned char* data, uint32_t len);
void SHA1Final(unsigned char digest[20], SHA1_CTX* context);
void SHA1(char* hash_out, const char* str, int len);
} // namespace o2::framework::internal

#endif // O2_FRAMEWORK_SHA1_H_
