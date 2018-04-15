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

namespace o2 {
namespace framework {

/// A concrete description of the output to be created
struct Output {
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType subSpec = 0;
  enum Lifetime lifetime = Lifetime::Timeframe;

  bool operator==(const Output& that)
  {
    return origin == that.origin && description == that.description && subSpec == that.subSpec &&
           lifetime == that.lifetime;
  };
};

}
}
#endif
