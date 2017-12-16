// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.cxx
/// \brief Cluster structure for TPC clusters

#include "TPCReconstruction/Cluster.h"

ClassImp(o2::TPC::ClusterTimeStamp);
ClassImp(o2::TPC::Cluster);

using namespace o2::TPC;

//________________________________________________________________________
std::ostream& Cluster::print(std::ostream& out) const {
//std::ostream &Cluster::Print(std::ostream &output) const
//{
  out << "TPC Cluster in CRU [" << mCRU << "], pad row ["
         << mRow << "] with charge/maxCharge " << mQ << "/" << mQmax
         << " and coordinates (" << mPadMean << ", " << getTimeStamp() << ")"
         << " and width (" << mPadSigma << ", " << getTimeStampError() << ")";
  return out;
}


