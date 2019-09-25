// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackTPC.cxx
/// \brief Implementation of the TPC track
/// \author Thomas Klemenz, TU Muenchen, thomas.klemenz@tum.de

#include "DataFormatsTPC/TrackTPC.h"

using namespace o2::tpc;

void TrackTPC::resetClusterReferences(int nClusters)
{
  mNClusters = short(nClusters);
  mClusterReferences.resize(nClusters + (nClusters + 1) / 2);
}
