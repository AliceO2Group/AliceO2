// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALGORITHM_HEADERSTACK_H
#define ALGORITHM_HEADERSTACK_H

/// @file   HeaderStack.h
/// @author Matthias Richter
/// @since  2017-09-19
/// @brief  Utilities for the O2 header stack

// the implemented functionality relies on o2::header::get defined in
// DataHeader.h; all O2 headers inherit from BaseHeader, this is required
// to check the consistency of the header stack, also the next-header-flag
// is part of the BaseHeader
#include "Headers/DataHeader.h" // for o2::header::get

namespace o2
{

namespace algorithm
{

/**
 * Generic utility for the O2 header stack, redirect to header specific callbacks
 *
 * The O2 header stack consists of one or more headers. The DataHeader is the
 * first one and mandatory. Other optional headers can be recursively extracted
 * from a buffer with one call to this utility function. For each header a pair
 * of type and callback has to be provided. The header type can be provided as
 * a dummy parameter, the callback with a lambda or any callable object.
 *
 * Usage:
 *   dispatchHeaderStackCallback(ptr, size,
 *                               MyHeader(),
 *                               [] (const auto & h) {
 *                                 // do something with h
 *                               }
 */
template <
  typename PtrType,
  typename SizeType,
  typename HeaderType,
  typename HeaderCallbackType,
  typename... MoreTypes>
void dispatchHeaderStackCallback(PtrType ptr,
                                 SizeType size,
                                 HeaderType header,
                                 HeaderCallbackType onHeader,
                                 MoreTypes&&... types);

// template specialization: handler for one pair of type and callback
template <
  typename PtrType,
  typename SizeType,
  typename HeaderType,
  typename HeaderCallbackType>
void dispatchHeaderStackCallback(PtrType ptr,
                                 SizeType size,
                                 HeaderType /*dummy*/,
                                 HeaderCallbackType onHeader)
{
  const HeaderType* h = o2::header::get<HeaderType*>(ptr, size);
  if (h) {
    onHeader(*h);
  }
}

// an empty function in case no StackTypes have been provided to call
template <typename PtrType, typename SizeType>
void dispatchHeaderStackCallback(PtrType ptr, SizeType size)
{
}

// actual implementation
template <
  typename PtrType,
  typename SizeType,
  typename HeaderType,
  typename HeaderCallbackType,
  typename... MoreTypes>
void dispatchHeaderStackCallback(PtrType ptr,
                                 SizeType size,
                                 HeaderType header,
                                 HeaderCallbackType onHeader,
                                 MoreTypes&&... types)
{
  // call for current
  dispatchHeaderStackCallback(ptr, size, header, onHeader);
  // call for next
  dispatchHeaderStackCallback(ptr, size, types...);
}

/**
 * Generic utility for the O2 header stack, extract headers
 *
 * The O2 header stack consists of one or more headers. The DataHeader is the
 * first one and mandatory. Other optional headers can be recursively extracted
 * from a buffer with one call to this utility function. For each header to be
 * extracted, a variable can be passed be reference. If a header of corresponding
 * type is in the stack, its content will be assigned to the variable.
 *
 * Usage:
 *   DataHeader dataheader;
 *   TriggerHeader triggerheader
 *   parseHeaderStack(ptr, size,
 *                    dataheader,
 *                    triggerheader
 */
template <
  typename PtrType,
  typename SizeType,
  typename HeaderType,
  typename... MoreTypes>
void parseHeaderStack(PtrType ptr,
                      SizeType size,
                      HeaderType& header,
                      MoreTypes&&... types);

// template specialization: handler for one type
template <
  typename PtrType,
  typename SizeType,
  typename HeaderType>
void parseHeaderStack(PtrType ptr,
                      SizeType size,
                      HeaderType& header)
{
  const HeaderType* h = o2::header::get<HeaderType*>(ptr, size);
  if (h) {
    header = *h;
  }
}

// an empty function in case no StackTypes have been provided to call
template <typename PtrType, typename SizeType>
void parseHeaderStack(PtrType ptr, SizeType size)
{
}

// generic implementation
template <
  typename PtrType,
  typename SizeType,
  typename HeaderType,
  typename... MoreTypes>
void parseHeaderStack(PtrType ptr,
                      SizeType size,
                      HeaderType& header,
                      MoreTypes&&... types)
{
  // call for current
  parseHeaderStack(ptr, size, header);
  // call for next
  parseHeaderStack(ptr, size, types...);
}

}; // namespace algorithm

}; // namespace o2

#endif //ALGORITHM_HEADERSTACK_H
