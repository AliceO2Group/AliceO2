// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_OUTPUT_H
#define FRAMEWORK_OUTPUT_H

#include "Headers/DataHeader.h"
#include "Framework/Lifetime.h"
#include "Headers/Stack.h"

namespace o2
{
namespace framework
{

/// A concrete description of the output to be created
///
/// Note that header::Stack forbids copy constructor and so it is for Output.
/// As a consequence it can not be used in standard containers. This is however
/// not a limitation which is expected to cause problems because Output is mostly
/// used as rvalue in specifying output route.
struct Output {
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType subSpec = 0;
  enum Lifetime lifetime = Lifetime::Timeframe;
  header::Stack metaHeader = {};

  Output(header::DataOrigin o, header::DataDescription d) : origin(o), description(d) {}

  Output(header::DataOrigin o, header::DataDescription d, header::DataHeader::SubSpecificationType s)
    : origin(o), description(d), subSpec(s)
  {
  }

  Output(header::DataOrigin o, header::DataDescription d, header::DataHeader::SubSpecificationType s, Lifetime l)
    : origin(o), description(d), subSpec(s), lifetime(l)
  {
  }

  Output(header::DataOrigin o, header::DataDescription d, header::DataHeader::SubSpecificationType s, Lifetime l,
         header::Stack&& stack)
    : origin(o), description(d), subSpec(s), lifetime(l), metaHeader(std::move(stack))
  {
  }

  Output(const Output&& rhs)
    : origin(rhs.origin),
      description(rhs.description),
      subSpec(rhs.subSpec),
      lifetime(rhs.lifetime),
      metaHeader(std::move(rhs.metaHeader))
  {
  }

  bool operator==(const Output& that)
  {
    return origin == that.origin && description == that.description && subSpec == that.subSpec &&
           lifetime == that.lifetime;
  };
};

} // namespace framework
} // namespace o2
#endif
