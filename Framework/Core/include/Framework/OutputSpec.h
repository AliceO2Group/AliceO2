// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_OUTPUTSPEC_H
#define FRAMEWORK_OUTPUTSPEC_H

#include "Headers/DataHeader.h"
#include "Framework/Lifetime.h"

namespace o2
{
namespace framework
{

struct OutputLabel {
  std::string value;
};

/// A selector for some kind of data being processed, either in
/// input or in output. This can be used, for example to match
/// specific payloads in a timeframe.
struct OutputSpec {
  OutputLabel binding;
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType subSpec = 0;
  enum Lifetime lifetime = Lifetime::Timeframe;

  OutputSpec(OutputLabel const& inBinding, header::DataOrigin inOrigin, header::DataDescription inDescription,
             header::DataHeader::SubSpecificationType inSubSpec, enum Lifetime inLifetime = Lifetime::Timeframe)
    : binding{ inBinding },
      origin{ inOrigin },
      description{ inDescription },
      subSpec{ inSubSpec },
      lifetime{ inLifetime }
  {
  }

  OutputSpec(header::DataOrigin inOrigin, header::DataDescription inDescription,
             header::DataHeader::SubSpecificationType inSubSpec, enum Lifetime inLifetime = Lifetime::Timeframe)
    : binding{ OutputLabel{ "" } },
      origin{ inOrigin },
      description{ inDescription },
      subSpec{ inSubSpec },
      lifetime{ inLifetime }
  {
  }

  OutputSpec(OutputLabel const& inBinding, header::DataOrigin inOrigin, header::DataDescription inDescription,
             enum Lifetime inLifetime = Lifetime::Timeframe)
    : binding{ inBinding }, origin{ inOrigin }, description{ inDescription }, subSpec{ 0 }, lifetime{ inLifetime }
  {
  }

  OutputSpec(header::DataOrigin inOrigin, header::DataDescription inDescription,
             enum Lifetime inLifetime = Lifetime::Timeframe)
    : binding{ OutputLabel{ "" } },
      origin{ inOrigin },
      description{ inDescription },
      subSpec{ 0 },
      lifetime{ inLifetime }
  {
  }

  bool operator==(const OutputSpec& that)
  {
    return origin == that.origin && description == that.description && subSpec == that.subSpec &&
           lifetime == that.lifetime;
  };
};

} // namespace framework
} // namespace o2
#endif
