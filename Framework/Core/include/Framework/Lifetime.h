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
#ifndef O2_FRAMEWORK_LIFETIME_H_
#define O2_FRAMEWORK_LIFETIME_H_

namespace o2::framework
{

/// Possible Lifetime of objects being exchanged by the DPL.
enum struct Lifetime {
  /// A message which is associated to a timeframe. DPL will wait indefinitely for it by default.
  Timeframe = 0,
  /// Eventually a message whose content is retrieved from CCDB
  Condition = 1,
  /// To be used when data is not produced for every timeframe but
  /// it can be result of a sampling process (like in the case of
  /// QC) or an aggregating one (like in the case of analysis histograms).
  QA = 2,
  Sporadic = 2,
  /// Do not use for now.
  Transient = 3,
  /// A message which is created whenever a Timer expires
  Timer = 4,
  /// A message which is created immediately, with payload / containing a
  /// single value which gets incremented for every / invokation.
  Enumeration = 5,
  /// A message which is created every time a SIGUSR1 is received.
  Signal = 6,
  /// An optional message. When data arrives, if not already part of the data,
  /// a dummy entry will be generated.
  /// This comes handy e.g. to handle Raw Data, since DataDistribution will provide
  /// everything in one go so whatever is expected but not there, for whatever reason
  /// will be substituted with a dummy entry.
  Optional = 7,
  /// An input which is materialised with the contents of some out of band
  /// FairMQ channel.
  OutOfBand = 8,
  /// An input / output which is actually dangling. End-users should not need
  /// to know about this, however this will be used by DPL to mark channels
  /// which at the end of the topological sort result dangling.
  Dangling = 9
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_LIFETIME_H_
