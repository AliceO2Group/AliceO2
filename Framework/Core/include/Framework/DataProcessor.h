// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DATAPROCESSOR_H_
#define O2_FRAMEWORK_DATAPROCESSOR_H_

class FairMQDevice;
class FairMQParts;

namespace o2::framework
{

class MessageContext;
class StringContext;
class ArrowContext;
class RawBufferContext;
class ServiceRegistry;
class DeviceState;

/// Helper class to send messages from a contex at the end
/// of a computation.
struct DataProcessor {
  static void doSend(FairMQDevice&, MessageContext&, ServiceRegistry&);
  static void doSend(FairMQDevice&, StringContext&, ServiceRegistry&);
  static void doSend(FairMQDevice&, ArrowContext&, ServiceRegistry&);
  static void doSend(FairMQDevice&, RawBufferContext&, ServiceRegistry&);
  static void doSend(FairMQDevice&, FairMQParts&&, const char*, unsigned int);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATAPROCESSOR_H_
