// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/*
 * File:   DeliveryType.h
 * Author: John Lång (john.larry.lang@cern.ch)
 *
 * Created on 17 July 2015, 9:54
 */

#ifndef O2_DCS_DELIVERY_TYPE
#define O2_DCS_DELIVERY_TYPE

#include <string>
#include <regex>
#include <stdexcept>
#include "DetectorsDCS/GenericFunctions.h"

namespace o2
{
namespace dcs
{
/**
     * This regular expression matches with strings representing payload types.
     */
static const std::regex REGEX_PT(
  "^(Raw|DPVAL)/(Int|Uint|Double|Bool|Char|String|Time|Binary)$");

/**
     * <p>DeliveryType is a piece of meta-information used for deducing types of
     * DPVAL payloads and DIM service description strings used with services
     * providing data to ADAPOS. DPCOMs use the DeliveryType of DPIDs to infer
     * the correct interpretation of payload data when being put into a
     * <tt>std::ofstream</tt>. Also, ADAPOS Engine uses DeliveryTypes for DIM
     * service subscription (and Load Generator for generating services).</p>
     * <p>Every DeliveryType, except <tt>VOID</tt>, has a raw and DPVAL
     * variant. A DIM service publishing payloads in DPVALs has a different
     * format and binary layout than a DIM service publishing only raw data, so
     * this distinction is critical to ADAPOS Engine.</p>
     *
     * @see ADAPRO::ADAPOS::DataPointCompositeObject
     * @see ADAPRO::ADAPOS::DataPointIdentifier
     * @see ADAPRO::ADAPOS::DataPointValue
     */
enum DeliveryType {
  /**
         * This DeliveryType is included only for testing and debugging
         * when the payload of a DPVAL is not going to be accessed. Accessing
         * the payload of a DPVAL with this DeliveryType results in a domain
         * error.
         */
  VOID = 0,

  /**
         * <tt>Binary</tt> is the general payload data type. This is the raw
         * variant.
         */
  RAW_BINARY = 64,

  /**
         * <tt>Binary</tt> is the general payload data type. This is the DPVAL
         * variant.
         */
  DPVAL_BINARY = 192,

  /**
         * <tt>Int</tt> stands for a 32-bit signed integer. This is the raw
         * variant. The numerical value of <tt>RAW_INT</tt> corresponds with the
         * WinCC constant <tt>DPEL_INT</tt>.
         */
  RAW_INT = 21,

  /**
         * <tt>Int</tt> stands for a 32-bit signed integer. This is the DPVAL
         * variant.
         */
  DPVAL_INT = 149,

  /**
         * <tt>Uint</tt> stands for a 32-bit unsigned integer. This is the raw
         * variant. The numerical value of <tt>RAW_UINT</tt> corresponds with
         * the WinCC constant <tt>DPEL_UINT</tt>.
         */
  RAW_UINT = 20,

  /**
         * <tt>Uint</tt> stands for a 32-bit insigned integer. This is the DPVAL
         * variant.
         */
  DPVAL_UINT = 148,

  /**
         * <tt>Double</tt> stands for IEEE 754 double precision floating point
         * number (64 bit). This is the raw variant. The numerical value of
         * <tt>RAW_INT</tt> corresponds with the WinCC constant
         * <tt>DPEL_FLOAT</tt>.
         */
  RAW_DOUBLE = 22,

  /**
         * <tt>Double</tt> stands for IEEE 754 double precision floating point
         * number (64 bit). This is the DPVAL variant.
         */
  DPVAL_DOUBLE = 150,

  /**
         * <tt>Bool</tt> stands for a boolean. This is the raw variant. The
         * numerical value of <tt>RAW_BOOL</tt> corresponds with the WinCC
         * constant <tt>DPEL_BOOL</tt>.
         */
  RAW_BOOL = 23,

  /**
         * <tt>Bool</tt> stands for a boolean. This is the DPVAL variant.
         */
  DPVAL_BOOL = 151,

  /**
         * <tt>Char</tt> stands for an ASCII character. This is the raw variant.
         * The numerical value of <tt>RAW_CHAR</tt> corresponds with the WinCC
         * constant <tt>DPEL_CHAR</tt>.
         */
  RAW_CHAR = 19,

  /**
         * <tt>Char</tt> stands for an ASCII character. This is the raw variant.
         */
  DPVAL_CHAR = 147,

  /**
         * <tt>String</tt> stands for a null-terminated array of ASCII
         * characters. This is the raw variant. The numerical value of
         * <tt>RAW_STRING</tt> corresponds with the WinCC constant
         * <tt>DPEL_STRING</tt>.
         */
  RAW_STRING = 25,

  /**
         * <tt>String</tt> stands for a null-terminated array of ASCII
         * characters. This is the DPVAL variant.
         */
  DPVAL_STRING = 153,

  /**
         * <tt>Time</tt> stands for two 32-bit (unsigned) integers containing
         * the milliseconds and secodns of a UNIX timestamp. This is the raw
         * variant. The numerical value of <tt>RAW_TIME</tt> corresponds with
         * the WinCC constant <tt>DPEL_DYN_UINT</tt>.
         */
  RAW_TIME = 4,

  /**
         * <tt>Time</tt> stands for two 32-bit (unsigned) integers containing
         * the milliseconds and secodns of a UNIX timestamp. This is the DPVAL
         * variant.
         */
  DPVAL_TIME = 132
};

template <>
inline DeliveryType read(const std::string& str)
{
  if (str == "Raw/Int") {
    return RAW_INT;
  } else if (str == "Raw/Uint") {
    return RAW_UINT;
  } else if (str == "Raw/Double") {
    return RAW_DOUBLE;
  } else if (str == "Raw/Bool") {
    return RAW_BOOL;
  } else if (str == "Raw/Char") {
    return RAW_CHAR;
  } else if (str == "Raw/String") {
    return RAW_STRING;
  } else if (str == "Raw/Binary") {
    return RAW_BINARY;
  } else if (str == "Raw/Time") {
    return RAW_TIME;
  } else if (str == "DPVAL/Int") {
    return DPVAL_INT;
  } else if (str == "DPVAL/Uint") {
    return DPVAL_UINT;
  } else if (str == "DPVAL/Double") {
    return DPVAL_DOUBLE;
  } else if (str == "DPVAL/Bool") {
    return DPVAL_BOOL;
  } else if (str == "DPVAL/Char") {
    return DPVAL_CHAR;
  } else if (str == "DPVAL/String") {
    return DPVAL_STRING;
  } else if (str == "DPVAL/Binary") {
    return DPVAL_BINARY;
  } else if (str == "DPVAL/Time") {
    return DPVAL_TIME;
  } else if (str == "Void") {
    return VOID;
  } else {
    throw std::domain_error("\"" + str +
                            "\" doesn't represent a DeliveryType.");
  }
}

template <>
inline std::string show(const DeliveryType type)
{
  switch (type) {
    case RAW_INT:
      return "Raw/Int";
    case RAW_UINT:
      return "Raw/Uint";
    case RAW_BOOL:
      return "Raw/Bool";
    case RAW_CHAR:
      return "Raw/Char";
    case RAW_DOUBLE:
      return "Raw/Double";
    case RAW_TIME:
      return "Raw/Time";
    case RAW_STRING:
      return "Raw/String";
    case RAW_BINARY:
      return "Raw/Binary";
    case DPVAL_INT:
      return "DPVAL/Int";
    case DPVAL_UINT:
      return "DPVAL/Uint";
    case DPVAL_BOOL:
      return "DPVAL/Bool";
    case DPVAL_CHAR:
      return "DPVAL/Char";
    case DPVAL_DOUBLE:
      return "DPVAL/Double";
    case DPVAL_TIME:
      return "DPVAL/Time";
    case DPVAL_STRING:
      return "DPVAL/String";
    case DPVAL_BINARY:
      return "DPVAL/Binary";
    case VOID:
      return "Void";
    default:
      throw std::domain_error("Illegal DeliveryType.");
  }
}

/**
     * Returns <tt>true</tt> if and only if the given DeliveryType is a DPVAL
     * variant.
     *
     * @param type  The DeliveryType to be checked.
     * @return      The result.
     * @throws std::domain_error If applied to an invalid value.
     */
inline bool DPVAL_variant(const DeliveryType type)
{
  switch (type) {
    case DPVAL_INT:
    case DPVAL_UINT:
    case DPVAL_BOOL:
    case DPVAL_CHAR:
    case DPVAL_DOUBLE:
    case DPVAL_TIME:
    case DPVAL_STRING:
    case DPVAL_BINARY:
      return true;
    case RAW_INT:
    case RAW_UINT:
    case RAW_BOOL:
    case RAW_CHAR:
    case RAW_DOUBLE:
    case RAW_TIME:
    case RAW_STRING:
    case RAW_BINARY:
    case VOID:
      return false;
    default:
      throw std::domain_error("Illegal DeliveryType.");
  }
}

/**
     * Returns the DIM description string for a DIM service with the given
     * DeliveryType.
     *
     * @param type  The DeliveryType.
     * @return      The corresponding DIM service description string.
     * @throws std::domain_error If applied to an invalid value.
     */
inline std::string dim_description(const DeliveryType type)
{

  switch (type) {
    case RAW_INT:
      return "I:1";
    case RAW_UINT:
      return "I:1";
    case RAW_BOOL:
      return "I:1";
    case RAW_CHAR:
      return "C:4";
    case RAW_DOUBLE:
      return "D:1";
    case RAW_TIME:
      return "I:2";
    case RAW_STRING:
      return "C:55";
    case RAW_BINARY:
      return "C:56";
    case DPVAL_INT:
      return "S:2;I:1;I:1";
    case DPVAL_UINT:
      return "S:2;I:1;I:1";
    case DPVAL_BOOL:
      return "S:2;I:1;I:1";
    case DPVAL_CHAR:
      return "S:2;I:1;C:4";
    case DPVAL_DOUBLE:
      return "S:2;I:1;D:1";
    case DPVAL_TIME:
      return "S:2;I:1;I:2";
    case DPVAL_STRING:
      return "S:2;I:1;C:55";
    case DPVAL_BINARY:
      return "S:2;I:1;C:56";
    case VOID:
    default:
      throw std::domain_error("Illegal DeliveryType.");
  }
}

/**
     * Returns the size of a buffer required to store the binary contents of a
     * DIM service of the given data type.
     *
     * @param type  The DeliveryType.
     * @return      Size of the payload.
     * @throws std::domain_error If applied to an invalid value.
     */
inline size_t dim_buffer_size(const DeliveryType type)
{
  switch (type) {
    case RAW_INT:
    case RAW_UINT:
    case RAW_BOOL:
    case RAW_CHAR:
      return 4;
    case RAW_DOUBLE:
    case RAW_TIME:
      return 8;
    case RAW_STRING:
      return 55;
    case RAW_BINARY:
      return 56;
    case DPVAL_INT:
    case DPVAL_UINT:
    case DPVAL_BOOL:
    case DPVAL_CHAR:
    case DPVAL_DOUBLE:
    case DPVAL_TIME:
    case DPVAL_STRING:
    case DPVAL_BINARY:
      return 64;
    case VOID:
      return 0;
    default:
      throw std::domain_error("Illegal DeliveryType.");
  }
}
} // namespace dcs

} // namespace o2

#endif /* O2_DCS_DELIVERY_TYPE */
