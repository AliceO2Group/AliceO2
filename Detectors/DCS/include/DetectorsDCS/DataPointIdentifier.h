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
 * File:   DataPointIdentifier.h
 * Author: John Lång (john.larry.lang@cern.ch)
 *
 * Created on 23 September 2016, 12:00
 */
#ifndef O2_DCS_DATAPOINT_IDENTIFIER_H
#define O2_DCS_DATAPOINT_IDENTIFIER_H

#include <cstring>
#include <string>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <functional>
#include <unordered_map>

#include "Rtypes.h"
#include "DetectorsDCS/StringUtils.h"
#include "DetectorsDCS/GenericFunctions.h"
#include "DetectorsDCS/DeliveryType.h"

namespace o2
{
namespace dcs
{
/**
     * DataPointIdentifier object is responsible for storing the alias and type
     * information of a data point.
     */
class alignas(64) DataPointIdentifier final
{
  const uint64_t pt1;
  const uint64_t pt2;
  const uint64_t pt3;
  const uint64_t pt4;
  const uint64_t pt5;
  const uint64_t pt6;
  const uint64_t pt7;
  const uint64_t pt8; // Contains the last 6 chars of alias and the type.

  DataPointIdentifier(
    const uint64_t pt1, const uint64_t pt2, const uint64_t pt3,
    const uint64_t pt4, const uint64_t pt5, const uint64_t pt6,
    const uint64_t pt7, const uint64_t pt8) noexcept : pt1(pt1), pt2(pt2), pt3(pt3), pt4(pt4), pt5(pt5), pt6(pt6), pt7(pt7), pt8(pt8) {}

 public:
  /**
         * The default constructor for DataPointIdentifier. Creates a DPID that
         * contains only zeroes. To fill it with alias and type, please use the
         * factory procedure <tt>fill</tt>.
         *
         * @see ADAPRO::ADAPOS::DataPointIdentifier::fill
         */
  DataPointIdentifier() noexcept : pt1(0), pt2(0), pt3(0), pt4(0), pt5(0), pt6(0), pt7(0), pt8(0)
  {
  }

  /**
         * A constructor for DataPointIdentifier. Copies the given arguments to
         * the memory segment owned by the object under construction.
         *
         * @param alias The alias of the DataPointIdentifier. Maximum length is
         * 62 characaters.
         * @param type  Type of the payload value associated with the service
         * identified by this DataPointIdentifier.
         */
  DataPointIdentifier(const std::string& alias, const DeliveryType type) noexcept : DataPointIdentifier()
  {
    strncpy((char*)this, alias.c_str(), 62);
    ((char*)&pt8)[7] = type;
  }

  /**
         * A copy constructor for DataPointIdentifier. 
         */
  DataPointIdentifier(const DataPointIdentifier& src) noexcept : DataPointIdentifier(src.pt1, src.pt2, src.pt3, src.pt4, src.pt5, src.pt6, src.pt7, src.pt8) {}

  DataPointIdentifier& operator=(const DataPointIdentifier& src) noexcept
  {
    if (&src != this) {
      memcpy(this, &src, sizeof(DataPointIdentifier));
    }
    return *this;
  }

  /**
         * This stati procedure fills the given DataPointIdentifier object with
         * the given parameters.
         *
         * @param dpid  The DataPointIdentifier to be filled (i.e. overwritten).
         * @param alias Alias of the data point. This is used for identifying
         * the DIM service that publishes updates to a data point.
         * @param type  Type of the data point payload value.
         */
  static inline void FILL(
    const DataPointIdentifier& dpid,
    const std::string& alias,
    const DeliveryType type) noexcept
  {
    strncpy((char*)&dpid, alias.c_str(), 62);
    ((char*)&dpid.pt8)[7] = type;
  }

  /**
         * This static procedure copies the given 64-byte binary segment into
         * the DataPointIdentifier object.
         *
         * @param dpid  The DataPointIdentifier to be filled (i.e. overwritten).
         * @param data  Beginning of the 64-byte binary segment.
         */
  static inline void FILL(
    const DataPointIdentifier& dpid,
    const uint64_t* const data) noexcept
  {
    std::strncpy((char*)&dpid, (char*)data, 62);
    ((char*)&dpid)[63] = ((data[7] & 0xFF00000000000000) >> 56);
  }

  /**
         * The equality comparison object of DPIDs.
         *
         * @param other The other DPID object for comparison.
         * @return      <tt>true</tt> if and only if the other DPID object
         * has exactly (bit-by-bit) the same state.
         */
  inline bool operator==(const DataPointIdentifier& other) const
  {
    return memcmp((char*)this, (char*)&other, 64) == 0;
  }

  /**
         * Negation of the equality comparison.
         *
         * @param other The second operand.
         * @return      <tt>true</tt> or <tt>false</tt>.
         */
  inline bool operator!=(const DataPointIdentifier& other) const
  {
    return !(*this == other);
  }

  /**
         * Appends DataPointIdentifier object to the given stream using CSV
         * format (i.e. <tt>&lt;alias&gt; ";" &lt;type&gt;</tt>).
         */
  friend inline std::ostream& operator<<(std::ostream& os,
                                         const DataPointIdentifier& dpid) noexcept
  {
    return os << std::string((char*)&dpid) << ";" << o2::dcs::show((DeliveryType)((dpid.pt8 & 0xFF00000000000000) >> 56));
  }

  /**
         * Returns the alias of the DPID object as a C-style string (i.e.
         * null-terminated char array).
         *
         * @return The alias.
         */
  inline const char* const get_alias() const noexcept
  {
    return (char*)&pt1;
  }

  /**
         * Getter for the payload type information.
         *
         * @return a DeliveryType object.
         */
  inline DeliveryType get_type() const noexcept
  {
    return (DeliveryType)((pt8 & 0xFF00000000000000) >> 56);
  }

  /**
         * Returns a hash code calculated from the alias. <em>Note that the
         * hash code is recalculated every time when this function is called.
         * </em>
         *
         * @return An unsigned integer.
         */
  inline size_t hash_code() const noexcept
  {
    return o2::dcs::hash_code(std::string((char*)&pt1));
  }

  /**
         * The destructor for DataPointIdentifier.
         */
  ~DataPointIdentifier() noexcept = default;
  ClassDefNV(DataPointIdentifier, 1);
};

/**
     * This simple function object is used for calculating the hash code for a
     * DataPointIdentifier object in a STL container.
     *
     * @param dpid  The DataPointIdentifier object, whose hash code is to be
     * calculated.
     * @return      The calculated hash code.
     */
struct DPIDHash {
  uint64_t operator()(const o2::dcs::DataPointIdentifier& dpid)
    const
  {
    return dpid.hash_code();
  }
};
} // namespace dcs

/// Defining DataPointIdentifier explicitly as messageable
namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::dcs::DataPointIdentifier> : std::true_type {
};
} // namespace framework

} // namespace o2

// specailized std::hash
namespace std
{
template <>
struct hash<o2::dcs::DataPointIdentifier> {
  std::size_t operator()(const o2::dcs::DataPointIdentifier& dpid) const
  {
    return std::hash<uint64_t>{}(dpid.hash_code());
  }
};

template <>
struct is_trivially_copyable<o2::dcs::DataPointIdentifier> : std::true_type {
};

} // namespace std

#endif /* O2_DCS_DATAPOINT_IDENTIFIER_H */
