// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DEVICEEXECUTION_H
#define FRAMEWORK_DEVICEEXECUTION_H

#include <vector>

namespace o2 {
namespace framework {

/// This  represent one  single  execution of  a Device.  It's  meant to  hold
/// information which  can change between  one execution  of a Device  and the
/// other, e.g. its pid or the arguments it is started with.
struct DeviceExecution {
  std::vector<char *> args;
};

}
}
#endif
