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

#ifndef O2_FRAMEWORK_ENDIAN_H_
#define O2_FRAMEWORK_ENDIAN_H_

// Lookup file for __BYTE_ORDER
#ifdef __APPLE__
#include <machine/endian.h>
#define swap16_ ntohs
#define swap32_ ntohl
#define swap64_ ntohll
#else
#include <endian.h>
#define swap16_ be16toh
#define swap32_ be32toh
#define ntohll be64toh
#define htonll htobe64
#define swap64_ ntohll
#endif
#define O2_HOST_BYTE_ORDER __BYTE_ORDER
#define O2_BIG_ENDIAN __BIG_ENDIAN
#define O2_LITTLE_ENDIAN __LITTLE_ENDIAN
#endif // O2_FRAMEWORK_ENDIAN_H_
