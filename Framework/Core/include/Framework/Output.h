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
#ifndef FRAMEWORK_OUTPUT_H
#define FRAMEWORK_OUTPUT_H

#include "Headers/DataHeader.h"
#include "Framework/Lifetime.h"
#include "Headers/Stack.h"

namespace o2::framework
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
  header::Stack metaHeader = {};

  Output(header::DataOrigin o, header::DataDescription d) : origin(o), description(d) {}

  Output(header::DataOrigin o, header::DataDescription d, header::DataHeader::SubSpecificationType s)
    : origin(o), description(d), subSpec(s)
  {
  }

  Output(header::DataOrigin o, header::DataDescription d, header::DataHeader::SubSpecificationType s, header::Stack&& stack)
    : origin(o), description(d), subSpec(s), metaHeader(std::move(stack))
  {
  }

  Output(header::DataHeader const& header)
    : origin(header.dataOrigin), description(header.dataDescription), subSpec(header.subSpecification)
  {
  }

  Output(const Output&) = delete;

  Output(Output&& rhs)
    : origin(rhs.origin),
      description(rhs.description),
      subSpec(rhs.subSpec),
      metaHeader(std::move(rhs.metaHeader))
  {
  }

  Output& operator=(const Output&) = delete;

  Output& operator=(Output&& rhs)
  {
    origin = rhs.origin;
    description = rhs.description;
    subSpec = rhs.subSpec;
    metaHeader = std::move(rhs.metaHeader);
    return *this;
  }

  bool operator==(const Output& that) const
  {
    return origin == that.origin && description == that.description && subSpec == that.subSpec;
  }
};

} // namespace o2
#endif
