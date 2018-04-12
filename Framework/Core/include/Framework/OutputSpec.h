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

namespace o2 {
namespace framework {

/// A selector for some kind of data being processed, either in
/// input or in output. This can be used, for example to match
/// specific payloads in a timeframe.
struct OutputSpec {
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType subSpec = 0;
  enum Lifetime lifetime = Lifetime::Timeframe;

  bool operator==(const OutputSpec& that)
  {
    return origin == that.origin && description == that.description && subSpec == that.subSpec &&
           lifetime == that.lifetime;
  };
};

}
}
#endif
