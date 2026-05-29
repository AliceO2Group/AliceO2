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

#ifndef O2_FRAMEWORK_COMMONLABELS_H
#define O2_FRAMEWORK_COMMONLABELS_H

#include "Framework/DataProcessorLabel.h"

namespace o2::framework
{

// Label to disable forwarding/advertising of DomainInfoHeader (oldest possible outputs)
// When present on a DataProcessor, no DomainInfoHeader messages will be sent downstream.
const extern DataProcessorLabel suppressDomainInfoLabel;

// Label to allow multiple DataProcessorSpecs with the same name in the topology.
// When present, duplicate specs with matching inputs and outputs will be deduplicated
// with a warning instead of causing a fatal error.
const extern DataProcessorLabel allowDuplicatesLabel;

} // namespace o2::framework

#endif // O2_FRAMEWORK_COMMONLABELS_H
