// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FLATTERRESTORE_H
#define FLATTERRESTORE_H

/// @file   FlattenRestore.h
/// @author Matthias Richter
/// @since  2020-04-05
/// @brief  Utilities to copy complex objects to flat buffer and restore

#include <type_traits>

namespace o2::algorithm
{
namespace flatten
{

/// Calculate cumulative value size of a variable number of arguments
/// The function takes parameters by reference and calculates the memory size of
/// all parameters together. The pointer attribute is removed.
///
/// Example:
///   char* array1;
///   int* array2;
///   float* array3;
///   size = value_size(array1, array2, array3);
///   // size is sizeof(char) + sizeof(int) + sizeof(float)
template <typename ValueType, typename... Args>
constexpr size_t value_size(ValueType const& value, Args&&... args)
{
  size_t size = sizeof(typename std::remove_pointer<typename std::remove_reference<ValueType>::type>::type);
  if constexpr (sizeof...(Args) > 0) {
    size += value_size(std::forward<Args>(args)...);
  }
  return size;
}

/// Copy the content of variable number of arrays with the same extent to a buffer
/// The target pointer is passed be reference and incremented while copying
/// @param wrtptr write pointer
/// @param count  extent of the arrays
/// @param args   a variable number of pointers to arrays
/// @return copied size in bytes
template <typename TargetType, typename ValueType, typename... Args>
static size_t copy_to(TargetType& wrtptr, size_t count, ValueType* array, Args&&... args)
{
  static_assert(std::is_pointer<TargetType>::value == true, "need reference to pointer");
  static_assert(sizeof(typename std::remove_pointer<TargetType>::type) == 1, "need char-like pointer");

  size_t copySize = 0;
  if (array != nullptr) {
    copySize = count * value_size(array);
    memcpy(wrtptr, array, copySize);
    wrtptr += copySize;
  } else if (count > 0) {
    throw std::runtime_error("invalid nullptr to array of " + std::to_string(count) + " element(s)");
  }
  if constexpr (sizeof...(Args) > 0) {
    copySize += copy_to(wrtptr, count, std::forward<Args>(args)...);
  }
  return copySize;
}

/// Set pointers to regions in source buffer
/// A variable number of pointer arguments of consecutive arrays in a buffer are set according
/// to the type size and the extent of arrays
/// The read pointer is passed be reference and incremented while copying
/// @param readptr the source buffer
/// @param count   extent of the arrays
/// @param args    a variable number of references of pointers to arrays
/// @return handled raw size in bytes
template <typename BufferType, typename ValueType, typename... Args>
static size_t set_from(BufferType& readptr, size_t count, ValueType& array, Args&&... args)
{
  static_assert(std::is_pointer<typename std::remove_reference<ValueType>::type>::value == true, "need reference to pointer");
  array = reinterpret_cast<typename std::remove_reference<ValueType>::type>(readptr);
  size_t readSize = count * value_size(array);
  readptr += readSize;
  if constexpr (sizeof...(Args) > 0) {
    readSize += set_from(readptr, count, std::forward<Args>(args)...);
  }
  return readSize;
}

/// Calculate the total size of a sequence of arrays with the same extent
/// the first argument is a dummy argument to have the same signature as copy_to
/// and set_from
/// @param dummyptr unused
/// @param count    extent of the arrays
/// @param args     a variable number of references of pointers to arrays
/// @return total size
template <typename BufferType, typename... Args>
static size_t calc_size(BufferType const& dummyptr, size_t count, Args&&... args)
{
  return count * value_size(std::forward<Args>(args)...);
}

} // namespace flatten
} // namespace o2::algorithm

#endif
