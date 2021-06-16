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
 * File:   Clock.hpp
 * Author: John Lång (john.larry.lang@cern.ch)
 *
 * Created on 29 August 2016, 9:29
 */
#ifndef O2_DCS_CLOCK_H
#define O2_DCS_CLOCK_H

#include <ctime>
#include <cstdint>
#include <chrono>
#include <memory>
#include <string>

namespace o2
{
namespace dcs
{
/**
     * Returns a simple timestamp presenting the milliseconds of time passed
     * since the given time point.
     *
     * @param beginning The time point used as a reference for the time interval
     * calculation.
     * @return The amount of milliseconds passed since the given time point.
     */
inline uint64_t time_since(
  const std::chrono::steady_clock::time_point beginning) noexcept
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::steady_clock::now() - beginning)
    .count();
}

/**
     * Returns the measured number of milliseconds passed since UNIX epoch
     * (1 January 1970 01:00:00.000). This function uses system clock.
     *
     * @return Number of milliseconds since epoch.
     * @see ADAPRO::Library::now
     */
inline uint64_t epoch_time() noexcept
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

/**
     * Returns a timestamp using steady clock. This function is suitable for
     * measuring time intervals, but it's not meant to be used for calculating
     * dates.
     *
     * @return
     * @see ADAPRO::Library::epoch_time
     */
inline uint64_t now() noexcept
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

/**
     * Returns a timestamp of the current point of time in the local timezone.
     *
     * @return A simple ISO-8601-esque timestamp (<tt>YYYY-MM-DD HH:MM:SS</tt>).
     * Every decimal number in the date string has leading zeros and therefore
     * fixed length.
     * @see ADAPRO::Control::fs_timestamp
     */
inline std::string timestamp() noexcept
{
  char buffer[20];
  std::time_t now = std::time(nullptr);
  std::strftime(buffer, 32, "%F %T", std::localtime(&now));
  return std::string(buffer);
}

/**
     * Returns a timestamp of the current point of time in the local timezone.
     * The format of the timestamp is specified with the parameter
     * <tt>format</tt>. The format of the format string is the same as is used
     * by <tt>std::strftime</tt>.
     *
     * @return A simple timestamp in a format specified with the parameter
     * <tt>format</tt>.
     */
inline std::string timestamp(const std::string& format) noexcept
{
  char buffer[20];
  std::time_t now = std::time(nullptr);
  std::strftime(buffer, 32, format.c_str(), std::localtime(&now));
  return std::string(buffer);
}

/**
     * Generates a simple timestamp usable file paths. This function is like
     * <tt>ADAPRO::Control::timestamp</tt>, but with spaces replaced with
     * underscores and colons with dots in order to ensure compatibility with
     * (Linux) filesystems. This function uses local timezone.
     *
     * @return A simple ISO-8601-esque timestamp (<tt>YYYY-MM-DD_HH.MM.SS</tt>).
     * Every decimal number in the date string has leading zeros and therefore
     * fixed length.
     * @see ADAPRO::Control::timestamp
     */
inline std::string fs_timestamp() noexcept
{
  char buffer[20];
  std::time_t now = std::time(nullptr);
  std::strftime(buffer, 32, "%F_%H.%M.%S", std::localtime(&now));
  return std::string(buffer);
}
} // namespace dcs
} // namespace o2

#endif /* O2_DCS_CLOCK_H */
