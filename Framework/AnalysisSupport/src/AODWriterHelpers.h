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

#ifndef O2_FRAMEWORK_AODROOTWRITERHELPERS_H_
#define O2_FRAMEWORK_AODROOTWRITERHELPERS_H_

#include "Framework/AlgorithmSpec.h"
#include <Monitoring/Monitoring.h>

namespace o2::framework::writers
{

struct AODWriterHelpers {
  static AlgorithmSpec getOutputObjHistWriter(ConfigContext const& context);
  static AlgorithmSpec getOutputTTreeWriter(ConfigContext const& context);
  static AlgorithmSpec getMetadataCollector(ConfigContext const& context);
};

} // namespace o2::framework::writers

#endif // O2_FRAMEWORK_AODROOTWRITERHELPERS_H_
