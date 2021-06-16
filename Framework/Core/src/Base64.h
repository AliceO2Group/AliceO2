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

#ifndef O2_FRAMEWORK_BASE64_H_
#define O2_FRAMEWORK_BASE64_H_

namespace o2::framework::internal
{
int base64_encode(char* dest, int size, unsigned char* src, int slen);
char* base64_enc_malloc(unsigned char* src, int slen);
int base64_decode(unsigned char* dest, int size, char* src);
unsigned char* base64_dec_malloc(char* src);
} // namespace o2::framework::internal

#endif // O2_FRAMEWORK_BASE64_H_
