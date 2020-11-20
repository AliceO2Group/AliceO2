// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  Timeframe,
  /// Eventually a message whose content is retrieved from CCDB
  Condition,
  /// Do not use for now
  QA,
  /// Do not use for now.
  Transient,
  /// A message which is created whenever a Timer expires
  Timer,
  /// A message which is created immediately, with payload / containing a
  /// single value which gets incremented for every / invokation.
  Enumeration,
  /// A message which is created every time a SIGUSR1 is received.
  Signal,
  /// An optional message. When data arrives, if not already part of the data,
  /// a dummy entry will be generated.
  /// This comes handy e.g. to handle Raw Data, since DataDistribution will provide
  /// everything in one go so whatever is expected but not there, for whatever reason
  /// will be substituted with a dummy entry.
  Optional
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_LIFETIME_H_
