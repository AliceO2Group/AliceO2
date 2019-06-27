// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file bitfield.h
/// \author David Rohr

#ifndef Q_BITFIELD_H
#define Q_BITFIELD_H

#ifdef GPUCA_NOCOMPAT_ALLOPENCL
#include <type_traits>
#endif

template <class T, class S>
class bitfield
{
 public:
  bitfield(T v) : bits((S)v) {}
  bitfield(S v = 0) : bits(v) {}
  bitfield(const bitfield&) = default;
  bitfield& operator=(const bitfield&) = default;
  bitfield operator|(const bitfield v) const { return bits | v.bits; }
  bitfield& operator|=(const bitfield v)
  {
    bits |= v.bits;
    return *this;
  }
  bitfield operator&(const bitfield v) const { return bits & v.bits; }
  bitfield operator&(const T v) const { return bits & static_cast<S>(v); }
  bitfield& operator&=(const bitfield v)
  {
    bits &= v.bits;
    return *this;
  }
  bitfield operator~() const { return ~bits; }
  bool operator==(const bitfield v) { return bits == v.bits; }
  bool operator==(const T v) { return bits == static_cast<S>(v); }
  bool operator!=(const bitfield v) { return bits != v.bits; }
  bool operator!=(const T v) { return bits != static_cast<S>(v); }
  bitfield& setBits(const bitfield v, bool w)
  {
    if (w) {
      bits |= v.bits;
    } else {
      bits &= ~v.bits;
    }
    return *this;
  }
  void set(S v) { bits = v; }
  void set(T v) { bits = static_cast<S>(v); }
  template <typename... Args>
  void set(T v, Args... args)
  {
    this->set(args...);
    bits |= static_cast<S>(v);
  }
  S get() const { return bits; }
  operator bool() const { return bits; }
  operator S() const { return bits; }
  bool isSet(const bitfield& v) const { return *this & v; }
  bool isSet(const S v) const { return bits & v; }

#ifdef GPUCA_NOCOMPAT_ALLOPENCL
  static_assert(std::is_integral<S>::value, "Storage type non integral");
  static_assert(sizeof(S) >= sizeof(T), "Storage type has insufficient capacity");
#endif

 private:
  S bits;
};

#endif
