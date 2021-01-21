// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AbstractRefAccessor.h
/// \brief Accessor for objects of the same base class located in different containers
/// \author ruben.shahoyan@cern.ch

#ifndef O2_POLYACCESSOR_H
#define O2_POLYACCESSOR_H

#include <array>
#include <gsl/span>

namespace o2
{
namespace dataformats
{

/*
  AbstractRefAccessor allows to register multiple containers of objects convertible to type T and 
  to access them by providing a global index indicating the registered source ID and the
  index within the original container
*/

template <typename T, int N>
class AbstractRefAccessor
{
 public:
  /// register container in the accessor
  template <typename C>
  void registerContainer(const C& cont, int src)
  {
    mSizeOfs[src] = sizeof(typename std::remove_reference<decltype(cont)>::type::value_type);
    mSizes[src] = cont.size();
    mContainerPtr[src] = reinterpret_cast<const char*>(cont.data());
  }

  /// get object as user provided type from explicitly provided source, index
  template <typename U>
  const U& get_as(int src, int idx) const
  {
    return *reinterpret_cast<const U*>(getPtr(src, idx));
  }

  /// get object as user provided type from explicitly provided source, index
  template <typename U, typename I>
  const U& get_as(const I globIdx) const
  {
    return get_as<U>(globIdx.getSource(), globIdx.getIndex());
  }

  /// get object from explicitly provided source, index
  const T& get(int src, int idx) const
  {
    return get_as<T>(src, idx);
  }

  /// get object from the global index
  template <typename I>
  const T& get(const I globIdx) const
  {
    return get_as<T>(globIdx.getSource(), globIdx.getIndex());
  }

  /// access particula source container as a span
  template <typename U>
  auto getSpan(int src) const
  {
    return gsl::span<const U>(reinterpret_cast<const U*>(getPtr(src, 0), getSize(src)));
  }

  size_t getSize(int src) const
  {
    return mSizes[src];
  }

 private:
  auto getPtr(int src, int idx) const { return mContainerPtr[src] + mSizeOfs[src] * idx; }

  std::array<size_t, N> mSizeOfs{};           // sizeof for all containers elements
  std::array<size_t, N> mSizes{};             // size of eack container
  std::array<const char*, N> mContainerPtr{}; // pointers on attached containers
};

} // namespace dataformats
} // namespace o2

#endif
