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
/// \brief Implementation of the ITSMFT cluster

#include "DataFormatsITSMFT/CompCluster.h"
#include <cassert>
#include <iostream>

using namespace o2::itsmft;

std::ostream& operator<<(std::ostream& stream, const CompCluster& cl)
{
  stream << " row: " << cl.getRow() << " col: " << cl.getCol()
         << " pattID " << cl.getPatternID() << " [flag: " << cl.getFlag() << "] ";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const CompClusterExt& cl)
{
  stream << " chip: " << cl.getChipID() << ((const CompCluster&)cl);
  return stream;
}

//______________________________________________________________________________
void CompCluster::print() const
{
  // print itself
  std::cout << *this << "\n";
}

//______________________________________________________________________________
void CompClusterExt::print() const
{
  // print itself
  std::cout << *this << "\n";
}

//______________________________________________________________________________
void CompCluster::sanityCheck()
{
  // check self-consistency
  static_assert(NBitsRow + NBitsCol + NBitsPattID + 1 < 8 * sizeof(mData), "mData is too short to fit all fields");
}
