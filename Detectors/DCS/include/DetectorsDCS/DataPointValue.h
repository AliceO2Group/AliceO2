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
 * File:   DataPointValue.h
 * Author: John Lång (john.larry.lang@cern.ch)
 *
 * Created on 23 September 2016, 12:20
 */
#ifndef O2_DCS_DATAPOINT_VALUE
#define O2_DCS_DATAPOINT_VALUE

#include <cstring>
#include <cstdint>
#include <ctime>
#include <atomic>
#include <string>
#include <ostream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include "Rtypes.h"
#include "DetectorsDCS/StringUtils.h"
#include "DetectorsDCS/Clock.h"
#include "DetectorsDCS/DeliveryType.h"

namespace o2
{
namespace dcs
{
/**
     * DataPointValue is the struct responsible for containing the flags,
     * timestamp, and payload value of a data point. It represents a discrete
     * update event to a data point. <em>It should be noted that the
     * constructors and setters of this struct, that take an array containing
     * binary data as parameter, don't check the length of the given array.</em>
     */
struct DataPointValue final {
  /**
         * <p>This flag is used for signaling that a DPVAL stream connection is
         * working, but there is no data to be sent.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>A</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t KEEP_ALIVE_FLAG = 0x0001;

  /**
         * <p>This flag is used for signaling the end of transmission.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>Z</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t END_FLAG = 0x0002;

  /**
         * <p>This flag is set in the ADAPOS FBI Master DPCOM object.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>M</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t FBI_FLAG = 0x0004;

  /**
         * <p>This flag is set by Producer to signal Consumer that this
         * DataPointValue object belongs to a new service, that doesn't yet have
         * its DataPointCompositeObject in the image.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>N</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t NEW_FLAG = 0x0008;

  /**
         * <p>This flag is reserved for the ADAPRO/ADAPOS Producer-Consumer
         * model. A DataPointValue is dirty if and only if it's stored in the
         * ring buffer but not been fully processed yet. This flag should always
         * be set when a DPCOM containing the DataPointValue object arrives from
         * ADAPOS Engine via TCP.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>D</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t DIRTY_FLAG = 0x0010;

  /**
         * <p>This flag is used as a part of the DPVAL mutual exclusion
         * mechanism. This flag should be set iff reader has the turn.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>U</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint64_t TURN_FLAG = 0x0020;

  /**
         * <p>This flag is used by the DPVAL mutual exclusion mechanism.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>W</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t WRITE_FLAG = 0x0040;

  /**
         * <p>This flag is used by the DPVAL mutual exclusion mechanism.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>R</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t READ_FLAG = 0x0080;

  /**
         * <p>This flag should be set if and only if there is a ring buffer
         * overflow happening. Ring buffer overflow is manifested as an
         * overwrite of a dirty DPVAL.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>O</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t OVERWRITE_FLAG = 0x0100;

  /**
         * <p>In case of buffer overflow, this flag can be used for indicating a
         * service that lost an update. <em>Not receiving this flag shouldn't be
         * taken as an indication that every update was successfully
         * transmitted.</em></p>
         * <p>If this flag is set, it will be denoted with letter <tt>V</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t VICTIM_FLAG = 0x0200;

  /**
         * <p>This flag should be set if and only if there is a DIM error
         * (namely, connection failure with the DIM service) in ADAPOS Engine
         * affecting this DataPointValue object.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>E</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t DIM_ERROR_FLAG = 0x0400;

  /**
         * <p>This flag should be set if and only if the DPID of the DPCOM
         * containing this DPVAL has incorrect or unexpected value.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>I</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t BAD_DPID_FLAG = 0x0800;

  /**
         * <p>This flag should be set if and only if the flags themselves have
         * bad or unexpedcted value.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>F</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t BAD_FLAGS_FLAG = 0x1000;

  /**
         * <p>This flag should be set if and only if the timestamp is
         * incorrect.</p>
         * <p>If this flag is set, it will be denoted with letter <tt>T</tt> in
         * the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t BAD_TIMESTAMP_FLAG = 0x2000;

  /**
         * <p>This flag should be set if and only if the payload value is bad.
         * </p><p>If this flag is set, it will be denoted with letter <tt>P</tt>
         * in the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t BAD_PAYLOAD_FLAG = 0x4000;

  /**
         * <p>This flag should be set if it a FBI contains duplicated DPCOMs
         * (i.e. DPCOMs with identical DPIDs). Also, partial FBI (with missing
         * DPCOMs) can be indicated with this flag.</p>
         * </p><p>If this flag is set, it will be denoted with letter <tt>B</tt>
         * in the CSV output generated from this DPVAL object.</p>
         */
  static constexpr uint16_t BAD_FBI_FLAG = 0x8000;

  /**
         * This mask covers the session flags: <tt>KEEP_ALIVE_FLAG</tt>,
         * <tt>END_FLAG</tt>, and <tt>FBI_MASTER_FLAG</tt>.
         *
         * @see ADAPRO::ADAPOS::DataPointValue::KEEPALIVE_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::END_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::FBI_FLAG
         */
  static constexpr uint16_t SESSION_MASK = 0x0007;

  /**
         * This mask covers the control flags: <tt>NEW_FLAG</tt>,
         * <tt>DIRTY_FLAG</tt>, <tt>TURN_FLAG</tt>, <tt>WRITE_FLAG</tt>, and
         * <tt>READ_FLAG</tt>.
         *
         * @see ADAPRO::ADAPOS::DataPointValue::NEW_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::DIRTY_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::TURN_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::WRITE_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::READ_FLAG
         */
  static constexpr uint16_t CONTROL_MASK = 0x00F8;

  /**
         * This mask covers the error flags: <tt>DIM_ERROR_FLAG</tt>,
         * <tt>OVERWRITE_FLAG</tt>, <tt>VICTIM_FLAG</tt>,
         * <tt>BAD_DPID_FLAG</tt>, <tt>BAD_FLAGS_FLAG</tt>,
         * <tt>BAD_TIMESTAMP_FLAG</tt>, <tt>BAD_PAYLOAD_FLAG</tt>, and
         * <tt>BAD_FBI_FLAG</tt>.
         *
         * @see ADAPRO::ADAPOS::DataPointValue::DIM_ERROR_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::OVERWRITE_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::VICTIM_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::BAD_DPID_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::BAD_FLAGS_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::BAD_TIMESTAMP_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::BAD_PAYLOAD_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::BAD_FBI_FLAG
         */
  static constexpr uint16_t ERROR_MASK = 0xFF00;

  /**
         * The ADAPOS flags, i.e. a bitmask of the 16 different DPVAL flags.
         *
         * @see ADAPRO::ADAPOS::DataPointValue::KEEPALIVE_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::END_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::FBI_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::NEW_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::DIRTY_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::TURN_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::WRITE_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::READ_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::DIM_ERROR_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::OVERWRITE_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::BAD_DPID_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::BAD_FLAGS_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::BAD_TIMESTAMP_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::BAD_PAYLOAD_FLAG
         * @see ADAPRO::ADAPOS::DataPointValue::BAD_FBI_FLAG
         */
  uint16_t flags = 0;

  /**
         * Milliseconds of the timestamp. This is the measured number of
         * milliseconds passed since the UNIX epoch
         * (1 January 1970, 01:00:00.000) modulo 1000. The purpose of this field
         * together with <tt>sec</tt> is to store a timestamp indicating the
         * moment of creation/modification of the DataPointValue object.
         *
         * @see ADAPRO::ADAPOS::DataPointValue::sec
         */
  uint16_t msec = 0;

  /**
         * Seconds of the timestamp. This is the measured number of seconds
         * passed since the UNIX epoch (1 January 1970, 01:00:00.000). The
         * purpose of this field together with <tt>msec</tt> is to store a
         * timestamp indicating the moment of creation/modification of the
         * DataPointValue object.
         */
  uint32_t sec = 0;

  /**
         * First part of the 56-byte binary payload value.
         */
  uint64_t payload_pt1 = 0;

  /**
         * Second part of the 56-byte binary payload value.
         */
  uint64_t payload_pt2 = 0;

  /**
         * Third part of the 56-byte binary payload value.
         */
  uint64_t payload_pt3 = 0;

  /**
         * Fourth part of the 56-byte binary payload value.
         */
  uint64_t payload_pt4 = 0;

  /**
         * Fifth part of the 56-byte binary payload value.
         */
  uint64_t payload_pt5 = 0;

  /**
         * Sixth part of the 56-byte binary payload value.
         */
  uint64_t payload_pt6 = 0;

  /**
         * Seventh part of the 56-byte binary payload value.
         */
  uint64_t payload_pt7 = 0;

  /**
         * The default constuctor of DataPointValue. Fills the object with
         * zeroes.
         */
  DataPointValue() = default;

  /**
         * The trivial copy constructor of DataPointValue.
         *
         * @param other The DataPointValue instance, whose state is to be copied
         * to the new object.
         */
  DataPointValue(const DataPointValue& other) = default;

  /**
         * Constructor for DataPointValue. <em>This constructor assumes that the
         * <tt>payload</tt> parameter points to a <tt>unit64_t</tt> array with
         * (at least) seven elements.</em> The data will be treated as binary
         * and will be written to the new DataPointValue object using
         * <tt>std::memcpy</tt>. Use of this constructor is discouraged, since
         * it might copy garbage data. Please consider using the constructor
         * with a <tt>DeliveryType</tt> parameter instead.
         *
         * @param flags         The ADAPOS flags.
         * @param milliseconds  Milliseconds of the timestamp.
         * @param seconds       Seconds of the timestamp.
         * @param payload       Pointer to a memory segment containing the
         * binary payload data. The next seven <tt>uint64_t</tt> values (i.e. 56
         * <tt>char</tt>s) will be copied to the DataPointValue object.
         * <em>payload</em> must not be <tt>nullptr</tt>. For an empty DPVAL,
         * use the default constructor.
         *
         * @deprecated Since ADAPRO 4.1.0, this constructor has become
         * deprecated. Please consider using the constructor with a
         * <tt>DeliveryType</tt> parameter instead.
         */
  [[deprecated]] DataPointValue(
    const uint16_t flags,
    const uint16_t milliseconds,
    const uint32_t seconds,
    const uint64_t* const payload) noexcept : flags(flags), msec(milliseconds), sec(seconds)
  {
    memcpy((void*)&payload_pt1, (void*)payload, 56);
  }

  /**
         * <p>Constructor for DataPointValue. Copies a data segment from the
         * array <tt>payload</tt>. Length of the copied data is determined by
         * the argument <tt>type</tt>. <em>No sanity checks on the given
         * array are performed.</em></p>
         * <p>If the given <tt>DeliveryType</t> is invalid, then the payload
         * of the new DataPointValue object will be filled with zeros.</p>
         *
         * @param flags         The ADAPOS flags.
         * @param milliseconds  Milliseconds of the timestamp.
         * @param seconds       Seconds of the timestamp.
         * @param payload       Pointer to a contiguous block of memory
         * containing the binary payload data.
         * @param type          Used for setting the payload correctly, using
         * <tt>set_payload</tt>. If this argument is <tt>VOID</tt>, then the
         * contents of the payload remain undefined.
         *
         * @throws std::domain_error If applied with an invalid DeliveryType.
         * @see ADAPRO::Data::DataPointValue::set_payload
         */
  DataPointValue(
    const uint16_t flags,
    const uint16_t milliseconds,
    const uint32_t seconds,
    const uint64_t* const payload,
    const DeliveryType type)
    : flags(flags), msec(milliseconds), sec(seconds)
  {
    set_payload(payload, type);
  }

  /**
         * Sets the given payload data to the DataPointValue. This setter avoids
         * copying garbage, because it uses the type information. However, it
         * assumes the given array to have at least the length determined by the
         * payload type.
         *
         * @param data  A binary segment containing the new payload data.
         * @param type  Type of the payload data.
         *
         * @throws std::domain_error If applied with an invalid DeliveryType.
         */
  inline void set_payload(
    const uint64_t* const data,
    const DeliveryType type)
  {
    switch (type) {
      case RAW_INT:
      case DPVAL_INT:
      case RAW_UINT:
      case DPVAL_UINT:
      case RAW_FLOAT:
      case DPVAL_FLOAT:
        this->payload_pt1 = data[0] & 0x00000000FFFFFFFF;
        break;
      case RAW_BOOL:
      case DPVAL_BOOL:
        this->payload_pt1 = (data[0] ? 1 : 0);
        break;
      case RAW_CHAR:
      case DPVAL_CHAR:
        this->payload_pt1 = data[0] & 0x00000000000000FF;
        break;
      case RAW_DOUBLE:
      case DPVAL_DOUBLE:
      case RAW_TIME:
      case DPVAL_TIME:
        this->payload_pt1 = data[0];
        break;
      case RAW_STRING:
      case DPVAL_STRING:
        std::strncpy((char*)&payload_pt1, (char*)data, 8);
        std::strncpy((char*)&payload_pt2, (char*)data + 8, 8);
        std::strncpy((char*)&payload_pt3, (char*)data + 16, 8);
        std::strncpy((char*)&payload_pt4, (char*)data + 24, 8);
        std::strncpy((char*)&payload_pt5, (char*)data + 32, 8);
        std::strncpy((char*)&payload_pt6, (char*)data + 40, 8);
        std::strncpy((char*)&payload_pt7, (char*)data + 48, 8);
        break;
      case RAW_BINARY:
      case DPVAL_BINARY:
        memcpy((void*)&payload_pt1, (void*)data, 56);
      case VOID:
        break;
      default:
        throw std::domain_error("Invalid DeliveryType.");
    }
  }

  /**
         * Returns the ADAPOS flags of the DataPointValue.
         *
         * @return ADAPOS flags.
         */
  inline uint16_t get_flags() const noexcept
  {
    return flags;
  }

  /**
         * Updates the timestamp of this DataPointValue object using system
         * clock.
         */
  inline void update_timestamp() noexcept
  {
    const uint64_t current_time(o2::dcs::epoch_time());
    msec = current_time % 1000;
    sec = current_time / 1000;
  }

  /**
         * <p>On Linux platforms, Returns a unique pointer to a string
         * containing an ISO 8601 timestamp. The value of the timestamp will be
         * calculated from the values of the fields <tt>msec</tt> and
         * <tt>sec</tt> and assuming the UTC timezone. The timestamp has the
         * following format:</p>
         * <p><tt>&lt;YYYY-MM-DD&gt; "T" &lt;HH:MM:SS.SSS&gt; "Z"</tt></p>
         *
         * @return An ISO 8601 compliant timestamp.
         */
  inline std::unique_ptr<std::string> get_timestamp() const noexcept
  {
#if defined(__linux__) || defined(__APPLE__)
    // time_t should be uint64_t (compatible) on 64-bit Linux platforms:
    char buffer[33];
    std::time_t ts((uint64_t)sec);
    std::strftime(buffer, 32, "%FT%T", std::gmtime(&ts));
    std::ostringstream oss;
    oss << std::string(buffer) << "." << std::setw(3) << std::setfill('0') << std::to_string(msec) << "Z";
    return std::make_unique<std::string>(
      std::move(oss.str()));
#else
    return std::make_unique<std::string>("Unsupported platform");
#endif
  }

  /**
         * Returns a 64-bit unsigned integer containing the timestamp value of
         * the DPVAL object.
         *
         * @return Milliseconds since the UNIX epoch (1.1.1970 01:00:00.000).
         */
  inline uint64_t get_epoch_time() const noexcept
  {
    return (((uint64_t)sec) * 1000) + msec;
  }

  /**
         * <p>The standard copy assignment operator. Performs a normal deep copy
         * operation overwriting this DataPointValue object with the other.</p>
         * <p><em>Note that this operator always copies a fixed size data
         * segment from the given DPVAL to this DPVAL regardless of the payload
         * value type. </em>Therefore, non-zero garbage data might be copied as
         * well. Using the setters with the <tt>type</tt> parameter is
         * recommended.</p>
         *
         * @param other The DataPointValue object, whose state will be copied to
         * this object.
         * @return <tt>*this</tt> after performing the copying.
         */
  DataPointValue& operator=(const DataPointValue& other) = default;

  /**
         * Bit-by bit equality comparison of DPVAL objects.
         *
         * @param other The second operand of equality comparison.
         * @return      <tt>true</tt> or <tt>false</tt>.
         */
  inline bool operator==(const DataPointValue& other) const noexcept
  {
    return memcmp((void*)this, (void*)&other, sizeof(DataPointValue)) == 0;
  }

  /**
         * Negation of the equality comparison.
         *
         * @param other The second operand.
         * @return      <tt>true</tt> or <tt>false</tt>.
         */
  inline bool operator!=(const DataPointValue& other) const noexcept
  {
    return !(*this == other);
  }

  /**
         * Sets the state of the DataPointValue. Copies a data segment from the
         * array <tt>payload</tt>. Length of the copied data is determined by
         * the argument <tt>type</tt>. <em>No sanity checks on the given array
         * are performed.</em>
         *
         * @param flags         New value for ADAPOS flags.
         * @param milliseconds  New value for milliseconds.
         * @param seconds       New value for seconds.
         * @param payload       New payload data.
         * @param type          Used for setting the payload correctly, using
         * <tt>set_payload</tt>.
         *
         * @throws std::domain_error If applied with an invalid DeliveryType.
         * @see ADAPRO::Data::DataPointValue::set_payload
         */
  inline void set(
    const uint16_t flags,
    const uint16_t milliseconds,
    const uint32_t seconds,
    const uint64_t* const payload,
    const DeliveryType type)
  {
    this->flags = flags;
    this->msec = milliseconds;
    this->sec = seconds;
    set_payload(payload, type);
  }

  /**
         * Sets the state of the DataPointValue, except for the flags. This
         * setter can be used for safely setting the state of the object without
         * interfering with the locking mechanism or invalidating the mutual
         * exclusion. This setter was added in <em>ADAPRO 2.4.0</em>.
         *
         * @param milliseconds  Milliseconds of the new timestamp.
         * @param seconds       Seconds of the new timestamp.
         * @param payload       New payload.
         * @param type          Used for setting the payload correctly, using
         * <tt>set_payload</tt>.
         *
         * @throws std::domain_error If applied with an invalid DeliveryType.
         * @see ADAPRO::Data::DataPointValue::set_payload
         */
  inline void set(
    const uint16_t milliseconds,
    const uint32_t seconds,
    const uint64_t* const payload,
    const DeliveryType type)
  {
    msec = milliseconds;
    sec = seconds;
    set_payload(payload, type);
  }

  /**
         * Sets the state of the DataPointValue. <em>No sanity checks on the
         * given array are performed.</em> The first array element is assumed to
         * contain data for the flags and the timestamp with their respective
         * sizes and order, while the rest of the array is assumed to contain
         * the payload data.
         *
         * @param data  New DPVAL contents.
         * @param type          Used for setting the payload correctly, using
         * <tt>set_payload</tt>.
         *
         * @throws std::domain_error If applied with an invalid DeliveryType.
         * @see ADAPRO::Data::DataPointValue::set_payload
         */
  inline void set(
    const uint64_t* const data,
    const DeliveryType type)
  {
    this->flags = data[0];
    this->msec = data[0] >> 16;
    this->sec = data[0] >> 32;
    set_payload(data + 1, type);
  }

  /**
         * Prints the flags and timestamp of the DataPointValue object into the
         * <tt>ostream</tt>, but not the payload data. The reason for omitting
         * the payload value is that DataPointValue lacks payload data typing
         * information, so conversion from binary to string is not meaningful.
         */
  friend inline std::ostream& operator<<(std::ostream& os,
                                         const DataPointValue& dpval) noexcept
  {
    return os << ((dpval.flags & KEEP_ALIVE_FLAG) ? "A" : "-")
              << ((dpval.flags & END_FLAG) ? "Z" : "-")
              << ((dpval.flags & FBI_FLAG) ? "M" : "-")
              << ((dpval.flags & NEW_FLAG) ? "N" : "-")
              << ((dpval.flags & DIRTY_FLAG) ? "D" : "-")
              << ((dpval.flags & TURN_FLAG) ? "U" : "-")
              << ((dpval.flags & WRITE_FLAG) ? "W" : "-")
              << ((dpval.flags & READ_FLAG) ? "R " : "- ")
              << ((dpval.flags & OVERWRITE_FLAG) ? "O" : "-")
              << ((dpval.flags & VICTIM_FLAG) ? "V" : "-")
              << ((dpval.flags & DIM_ERROR_FLAG) ? "E" : "-")
              << ((dpval.flags & BAD_DPID_FLAG) ? "I" : "-")
              << ((dpval.flags & BAD_FLAGS_FLAG) ? "F" : "-")
              << ((dpval.flags & BAD_TIMESTAMP_FLAG) ? "T" : "-")
              << ((dpval.flags & BAD_PAYLOAD_FLAG) ? "P" : "-")
              << ((dpval.flags & BAD_FBI_FLAG) ? "B;" : "-;")
              << *(dpval.get_timestamp());
  }

  /**
         * <p>Acquires the lock for writing into this DPVAL. This method is
         * based on Peterson's algorithm (with a spinlock). It can be used for
         * ensuring mutual exclusion between two threads accessing this DPVAL
         * object. <em>It is important to release the write lock by calling
         * <tt>release_write_lock()</tt> after use.</em></p>
         * <p><em>This method may or may not work due to CPU instruction
         * reordering, caching, and other optimizations.</em> Using
         * <tt>std::mutex</tt> instead is recommended.</p>
         *
         * @see ADAPRO::ADAPOS::DataPointValue::release_write_lock
         */
  inline void acquire_write_lock()
  {
    flags |= WRITE_FLAG;
    flags |= TURN_FLAG;
    while (flags & READ_FLAG && flags & TURN_FLAG) {
    }
  }

  /**
         * <p>Releases the lock for writing into this DPVAL. This function is
         * based on Peterson's algorithm (with a spinlock). It can be used for
         * ensuring mutual exclusion between two threads accessing this DPVAL
         * object.</p>
         * <p><em>This method may or may not work due to CPU instruction
         * reordering, caching, and other optimizations.</em> Using
         * <tt>std::mutex</tt> instead is recommended.</p>
         *
         * @see ADAPRO::ADAPOS::DataPointValue::acquire_write_lock
         */
  inline void release_write_lock()
  {
    flags &= ~WRITE_FLAG;
  }

  /**
         * <p>Acquires the lock for reading from this DPVAL. This function is
         * based on Peterson's algorithm (with a spinlock). It can be used for
         * ensuring mutual exclusion between two threads accessing this DPVAL
         * object. <em>It is important to release the read lock by calling
         * <tt>release_read_lock()</tt> after use.</em></p>
         * <p><em>This method may or may not work due to CPU instruction
         * reordering, caching, and other optimizations.</em> Using
         * <tt>std::mutex</tt> instead is recommended.</p>
         *
         * @see ADAPRO::ADAPOS::DataPointValue::release_read_lock
         */
  inline void acquire_read_lock()
  {
    flags |= READ_FLAG;
    flags &= ~TURN_FLAG;
    while (flags & WRITE_FLAG && !(flags & TURN_FLAG)) {
    }
  }

  /**
         * <p>Releases the lock for reading from this DPVAL. This function is
         * based on Peterson's algorithm (with a spinlock). It can be used for
         * ensuring mutual exclusion between two threads accessing this DPVAL
         * object.</p>
         * <p><em>This method may or may not work due to CPU instruction
         * reordering, caching, and other optimizations.</em> Using
         * <tt>std::mutex</tt> instead is recommended.</p>
         *
         * @see ADAPRO::ADAPOS::DataPointValue::acquire_read_lock
         */
  inline void release_read_lock()
  {
    flags &= ~READ_FLAG;
  }
  ClassDefNV(DataPointValue, 1);
};
} // namespace dcs
} // namespace o2

#endif /* O2_DCS_DATAPOINT_VALUE_H */
