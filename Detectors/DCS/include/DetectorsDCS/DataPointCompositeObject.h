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

/*
 * File:   DataPointCompositeObject.h
 * Author: John Lång (john.larry.lang@cern.ch)
 *
 * Created on 23 September 2016, 11:28
 */
#ifndef O2_DCS_DATAPOINT_COMPOSITE_OBJECT_H
#define O2_DCS_DATAPOINT_COMPOSITE_OBJECT_H

#include <cstdint>
#include <stdexcept>
#include <ostream>
#include <string>
#include "Rtypes.h"
#include "DetectorsDCS/StringUtils.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"

namespace o2
{
namespace dcs
{
/**
     * DataPointCompositeObject is a composition of a DataPointIdentifier and a
     * DataPointValue. It is the unit of data points that ADAPOS provides.
     *
     * @see ADAPRO::ADAPOS::DataPointIdentifier
     * @see ADAPRO::ADAPOS::DataPointValue
     */
struct DataPointCompositeObject final {
  /**
         * The DataPointIdentifier object, which occupies the first 64 bytes of
         * the DataPointCompositeObject. This object contains the immutable
         * alias and type information.
         *
         * @see ADAPRO::ADAPOS::DataPointIdentifier
         */
  const DataPointIdentifier id;

  /**
         * The DataPointValue object, which occupies the last 64 bytes of the
         * DataPointCompositeObject. This object contains the mutable parts of
         * DataPointCompositeObject. These parts are the ADAPOS flags,
         * timestamp, and the payload data.
         *
         * @see ADAPRO::ADAPOS::DataPointValue
         */
  DataPointValue data;

  /**
         * The default constructor for DataPointCompositeObject. Uses the
         * default constructors of its component, which means that their every
         * field will be filled with zeroes.
         *
        * @see ADAPRO::ADAPOS::DataPointIdentifier
        * @see ADAPRO::ADAPOS::DataPointValue
         */
  DataPointCompositeObject() noexcept : id(), data() {}

  /**
         * This constructor <em>copies</em> the given DataPointIdentifier and
         * DataPointValue into the fields <tt>id</tt> and <tt>data</tt>.
         *
         * @param id    The DPID component.
         * @param data  The DPVAL component.
         */
  DataPointCompositeObject(
    const DataPointIdentifier& id,
    const DataPointValue& data) noexcept : id(id), data(data) {}

  /**
         * Copy constructor
         */
  DataPointCompositeObject(const DataPointCompositeObject& src) noexcept : DataPointCompositeObject(src.id, src.data) {}

  DataPointCompositeObject& operator=(const DataPointCompositeObject& src) noexcept
  {
    if (&src != this) {
      memcpy(this, &src, sizeof(DataPointCompositeObject));
    }
    return *this;
  }

  /**
         * Bit-by bit equality comparison of DataPointCompositeObjects.
         *
         * @param other The right-hand operand of equality comparison.
         * @return      <tt>true</tt> or <tt>false</tt>.
         */
  inline bool operator==(const DataPointCompositeObject& other) const
    noexcept
  {
    return id == other.id && data == other.data;
  }

  /**
         * Negation of the <tt>==</tt> operator.
         *
         * @param other The right-hand side operand.
         * @return      <tt>true</tt> or <tt>false</tt>.
         */
  inline bool operator!=(const DataPointCompositeObject& other) const
    noexcept
  {
    return id != other.id || data != other.data;
  }

  /**
         * Overwrites a DataPointCompositeObject with the given <tt>id</tt> and
         * <tt>data</tt> values.
         *
         * @param id    The id value.
         * @param data  The data value.
         */
  inline void set(const DataPointIdentifier& id, DataPointValue& data) noexcept
  {
    DataPointIdentifier::FILL(this->id, (uint64_t*)&id);
    this->data = data;
  }

  /**
         * Overwrites a DataPointCompositeObject as a copy of the given 128-byte
         * segment (16 times <tt>sizeof(uint64_t)</tt>) of binary data.
         *
         * @param data Beginning of the data segment used for reading. The
         * length of the segment is assumed to be exactly 128 bytes and contain
         * a valid binary representation of a DataPointCompositeObject.
         */
  inline void set(const uint64_t* const data) noexcept
  {
    DataPointIdentifier::FILL(this->id, data);
    this->data.set(data + 8, id.get_type());
  }

  /**
         * Overwrites the state of the <tt>data</tt> field with the state of the
         * given DPVAL object, <em>except for the control flags, that will be
         * cleared out</em>.
         *
         * @param dpval A DPVAL object representing an update event to the
         * DPCOM in question.
         *
         * @see ADAPRO::ADAPOS::DataPointValue
         * @see ADAPRO::ADAPOS::DataPointValue::CONTROL_MASK
         */
  inline void update(const DataPointValue& dpval) noexcept
  {
    data.set(
      dpval.flags & ~DataPointValue::CONTROL_MASK,
      dpval.msec,
      dpval.sec,
      (uint64_t*)&dpval.payload_pt1,
      id.get_type());
  }

  /**
         * Returns a pointer to the beginning of the data part to be given to
         * DIM, which depends on the <tt>DeliveryType</tt> of this DPCOM. This
         * method is specific to ADAPOS Load Generator and ADAPOS Engine.
         *
         * @return  A <tt>(void*)</tt>.
         * @throws std::domain_error If the <tt>DeliveryType</tt> of this
         * DPCOM object was illegal (i.e. <tt>VOID</tt> or something else than
         * the enumerators of <tt>DeliveryType</tt>).
         * @see ADAPRO::ADAPOS::DeliveryType
         */
  inline void* dim_buffer() const
  {
    switch (id.get_type()) {
      case RAW_INT:
      case RAW_UINT:
      case RAW_BOOL:
      case RAW_CHAR:
      case RAW_DOUBLE:
      case RAW_TIME:
      case RAW_STRING:
      case RAW_BINARY:
      case RAW_FLOAT:
        return (void*)&data.payload_pt1;
      case DPVAL_INT:
      case DPVAL_UINT:
      case DPVAL_BOOL:
      case DPVAL_CHAR:
      case DPVAL_DOUBLE:
      case DPVAL_TIME:
      case DPVAL_STRING:
      case DPVAL_BINARY:
      case DPVAL_FLOAT:
        return (void*)&data;
      default:
      case VOID:
        throw std::domain_error("Illegal DeliveryType.");
    }
  }

  /**
         * Prints a CSV-representation of the DataPointCompositeObject into the
         * ostream. The format of the payload data depends on its type and is
         * handled automatically.
         *
         * @param os    The output stream.
         * @param dpcom The DataPointCompositeObject.
         * @return      Reference to the stream after the printing operation.
         */
  inline friend std::ostream& operator<<(std::ostream& os,
                                         const DataPointCompositeObject& dpcom) noexcept
  {
    std::string payload;
    os << dpcom.id << ";" << dpcom.data << ";";
    union Converter {
      uint64_t raw_data;
      float float_value;
      double double_value;
      uint32_t uint_value;
      int32_t int_value;
      char char_value;
      bool bool_value;
    } converter;
    converter.raw_data = dpcom.data.payload_pt1;
    switch (dpcom.id.get_type()) {
      case VOID:
        return os;
      case RAW_INT:
      case DPVAL_INT:
        return os << converter.int_value;
      case RAW_UINT:
      case DPVAL_UINT:
        return os << converter.uint_value;
      case RAW_BOOL:
      case DPVAL_BOOL:
        return os << (converter.bool_value ? "True" : "False");
      case RAW_CHAR:
      case DPVAL_CHAR:
        return os << converter.char_value;
      case RAW_FLOAT:
      case DPVAL_FLOAT:
        return os << converter.float_value;
      case RAW_DOUBLE:
      case DPVAL_DOUBLE:
        return os << converter.double_value;
      case RAW_STRING:
      case DPVAL_STRING:
        return os << std::string((char*)&dpcom.data.payload_pt1);
      case RAW_TIME:
      case DPVAL_TIME: {
        // I have no idea how to properly test the time_t backend.
        // It's probably even hardware/os/kernel specific...
#ifdef __linux__
        char buffer[17];
        std::time_t ts((dpcom.data.payload_pt1) >> 32);
        std::strftime(buffer, 32, "%FT%T", std::gmtime(&ts));
        return os << std::string(buffer) << "." << std::setw(3) << std::setfill('0') << std::to_string((dpcom.data.payload_pt1 & 0x00000000FFFF0000) >> 16)
                  << "Z";
#else
        return os << "[Timestamps not supported on this platform.]";
#endif
      }
      case RAW_BINARY:
      case DPVAL_BINARY:
      default:
        return os << o2::dcs::to_hex_little_endian(
                 (char*)&dpcom.data.payload_pt1, 56);
    }
  }

  /**
    * The destructor for DataPointCompositeObject so it is not deleted
    * and thus DataPointCompositeObject is trivially copyable
    */
  ~DataPointCompositeObject() noexcept = default;
  ClassDefNV(DataPointCompositeObject, 1);
};

/**
  * Return the value contained in the DataPointCompositeObject, if possible.
  *
  * @tparam T the expected type of the value
  *
  * @param dpcom the DataPointCompositeObject the value is extracted from
  *
  * @returns the value of the data point
  *
  * @throws if the DeliveryType of the data point is not compatible with T
  */
template <typename T>
T getValue(const DataPointCompositeObject& dpcom);

} // namespace dcs

/// Defining DataPointCompositeObject explicitly as messageable
namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::dcs::DataPointCompositeObject> : std::true_type {
};
} // namespace framework

} // namespace o2

/// Defining DataPointCompositeObject explicitly as copiable
namespace std
{
template <>
struct is_trivially_copyable<o2::dcs::DataPointCompositeObject> : std::true_type {
};
} // namespace std

#endif /* O2_DCS_DATAPOINT_COMPOSITE_OBJECT_H */
