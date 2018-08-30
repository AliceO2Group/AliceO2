// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_LIFETIMEHELPERS_H
#define FRAMEWORK_LIFETIMEHELPERS_H

#include <functional>
#include <chrono>
#include "Framework/ExpirationHandler.h"
#include "Framework/PartRef.h"

namespace o2
{
namespace framework
{

struct InputRoute;

/// Lifetime handlers are used to manage the cases in which data is not coming
/// from the dataflow, but from some other source or trigger, e.g.,
/// in the case condition data or time based triggers.
struct LifetimeHelpers {
  /// Callback which never expires records. To be used with, e.g.
  /// Lifetime::Timeframe.
  static ExpirationHandler::Checker expireNever();
  /// Callback which always expires records. To be used with, e.g.
  /// Lifetime::Transient.
  static ExpirationHandler::Checker expireAlways();
  /// Callback which expires records with the rate given by @a period, in
  /// milliseconds.
  static ExpirationHandler::Checker expireTimed(std::chrono::milliseconds period);

  /// Does nothing. Use this for cases where you do not want to do anything
  /// when records expire. This is the default behavior for data (which never
  /// expires via this mechanism).
  static ExpirationHandler::Handler doNothing();

  /// Build a fetcher for an object from CCDB when the record is expired.
  /// @a prefix is the lookup prefix in CCDB.
  /// FIXME: actually implement the fetching
  /// FIXME: provide a way to customize the namespace from the ProcessingContext
  static ExpirationHandler::Handler fetchFromCCDBCache(std::string const& prefix);

  /// Create an entry in the registry for histograms on the first
  /// FIXME: actually implement this
  /// FIXME: provide a way to customise the histogram from the configuration.
  static ExpirationHandler::Handler fetchFromQARegistry();

  /// Create an entry in the registry for histograms on the first
  /// FIXME: actually implement this
  /// FIXME: provide a way to customise the histogram from the configuration.
  static ExpirationHandler::Handler fetchFromObjectRegistry();

  /// Enumerate entries on every invokation.  @a route is the route which the
  /// given enumeration refers to. In particular messages created by the
  /// returned ExpirationHandler will have an header which matches the
  /// dataOrigin, dataDescrition and dataSpecification of the given @a route.
  /// The payload of each message will contain an incremental number for each
  /// message being created.
  static ExpirationHandler::Handler enumerate(InputRoute const& route);
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_LIFETIMEHELPERS_H
