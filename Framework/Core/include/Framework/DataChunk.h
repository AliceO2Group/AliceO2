// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATACHUNK_H
#define FRAMEWORK_DATACHUNK_H

namespace o2 {
namespace framework {

/// Simple struct to hold a pointer to the actual FairMQMessage.
/// In principle this could be an iovec...
struct DataChunk {
  char *data;
  size_t size;
};

}
}

#endif // FRAMEWORK_DATACHUNK_H
