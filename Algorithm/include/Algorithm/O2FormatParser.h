// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALGORITHM_O2FORMATPARSER_H
#define ALGORITHM_O2FORMATPARSER_H

/// @file   O2FormatParser.h
/// @author Matthias Richter
/// @since  2017-10-18
/// @brief  Parser for the O2 data format

#include "HeaderStack.h"

namespace o2
{

namespace algorithm
{

/**
 * parse an input list and try to interpret in O2 data format
 * O2 format consist of header-payload message pairs. The header message
 * always starts with the DataHeader, optionally there can be more
 * headers in the header stack.
 *
 * The following callbacks are mandadory to be provided, e.g. through lambdas
 * - insert function with signature (const DataHeader&, ptr, size)
 *   auto insertFct = [&] (const auto & dataheader,
 *                         auto ptr,
 *                         auto size) {
 *     // do something with dataheader and buffer
 *   };
 * - getter for the message pointer, e.g. provided std::pair is used
 *   auto getPointerFct = [] (const auto & arg) {return arg.first;};
 * - getter for the message size, e.g. provided std::pair is used
 *   auto getSizeFct = [] (const auto & arg) {return arg.second;};
 *
 * Optionally, also the header stack can be parsed by specifying further
 * arguments. For every header supposed to be parsed, a pair of a dummy object
 * and callback has to be specified, e.g.
 *   // handler callback for MyHeaderStruct
 *   auto onMyHeaderStruct = [&] (const auto & mystruct) {
 *     // do something with mystruct
 *   }; // end handler callback
 *
 *   parseO2Format(list, insertFct, MyHeaderStruct(), onMyHeaderStruct);
 *
 */
template <
  typename InputListT, typename GetPointerFctT, typename GetSizeFctT, typename InsertFctT, // (const auto&, ptr, size)
  typename... HeaderStackTypes                                                             // pairs of HeaderType and CallbackType
  >
int parseO2Format(const InputListT& list,
                  GetPointerFctT getPointer,
                  GetSizeFctT getSize,
                  InsertFctT insert,
                  HeaderStackTypes&&... stackArgs)
{
  const o2::header::DataHeader* dh = nullptr;
  for (auto& part : list) {
    if (!dh) {
      // new header - payload pair, read DataHeader
      dh = o2::header::get<o2::header::DataHeader*>(getPointer(part), getSize(part));
      if (!dh) {
        return -ENOMSG;
      }
      o2::algorithm::dispatchHeaderStackCallback(getPointer(part),
                                                 getSize(part),
                                                 stackArgs...);
    } else {
      insert(*dh, getPointer(part), getSize(part));
      dh = nullptr;
    }
  }
  if (dh) {
    return -ENOMSG;
  }
  return list.size() / 2;
}

} // namespace algorithm

} // namespace o2

#endif // ALGORITHM_O2FORMATPARSER_H
