// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_ROOTSERIALIZATIONSUPPORT_H_
#define O2_FRAMEWORK_ROOTSERIALIZATIONSUPPORT_H_

#include "Framework/TMessageSerializer.h"
#include <TClass.h>

namespace o2::framework
{

/// Helper class to abstract the presence of ROOT at the eyes of the
/// compiler.
struct RootSerializationSupport {
  using TClass = ::TClass;
  using FairTMessage = o2::framework::FairTMessage;
  using TObject = ::TObject;
};

};     // namespace o2::framework
#endif // O2_FRAMEWORK_ROOTSERIALIZATIONSUPPORT_H_
