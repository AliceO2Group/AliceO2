// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAPROCESSOR_H
#define FRAMEWORK_DATAPROCESSOR_H

class FairMQDevice;

namespace o2
{
namespace framework
{

class RootObjectContext;
class MessageContext;
class StringContext;
class ArrowContext;

/// Helper class to send messages from a contex at the end
/// of a computation.
struct DataProcessor {
  static void doSend(FairMQDevice&, RootObjectContext&);
  static void doSend(FairMQDevice&, MessageContext&);
  static void doSend(FairMQDevice&, StringContext&);
  static void doSend(FairMQDevice&, ArrowContext&);
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DATAPROCESSOR_H
