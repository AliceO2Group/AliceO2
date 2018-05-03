// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_LIFETIME_H
#define FRAMEWORK_LIFETIME_H

namespace o2 {
namespace framework {

/// Possible Lifetime of objects being exchanged by the DPL.
/// FIXME: currently only Timeframe behaves as expected.
enum struct Lifetime {
  Timeframe,
  Condition,
  QA,
  Transient
};

}
}
#endif
