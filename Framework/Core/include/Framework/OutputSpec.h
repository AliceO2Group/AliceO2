// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_OUTPUTSPEC_H_
#define O2_FRAMEWORK_OUTPUTSPEC_H_

#include "Headers/DataHeader.h"
#include "Framework/Lifetime.h"

namespace o2::framework
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
             header::DataHeader::SubSpecificationType inSubSpec, enum Lifetime inLifetime = Lifetime::Timeframe);

  OutputSpec(header::DataOrigin inOrigin, header::DataDescription inDescription,
             header::DataHeader::SubSpecificationType inSubSpec, enum Lifetime inLifetime = Lifetime::Timeframe);

  OutputSpec(OutputLabel const& inBinding, header::DataOrigin inOrigin, header::DataDescription inDescription,
             enum Lifetime inLifetime = Lifetime::Timeframe);

  OutputSpec(header::DataOrigin inOrigin, header::DataDescription inDescription,
             enum Lifetime inLifetime = Lifetime::Timeframe);

  bool operator==(OutputSpec const& that) const;

  friend std::ostream& operator<<(std::ostream& stream, OutputSpec const& arg);
};

} // namespace o2::framework
#endif
