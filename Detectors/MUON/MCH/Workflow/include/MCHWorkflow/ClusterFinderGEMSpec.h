// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterFinderGEMSpec.h
/// \brief Definition of a data processor to run the GEM MLEM cluster finder
///
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_CLUSTERFINDERGEMSPEC_H_
#define O2_MCH_CLUSTERFINDERGEMSPEC_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace mch
{

o2::framework::DataProcessorSpec getClusterFinderGEMSpec();

} // end namespace mch
} // end namespace o2

#endif // O2_MCH_CLUSTERFINDERGEMSPEC_H_
