// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/algorithms/ClusterFinder.h>
#include <gpucf/debug/utils.h>


namespace gpucf
{
    
class PeakCountTest : public ClusterFinder
{
    
public:

   PeakCountTest(ClusterFinderConfig, ClEnv); 

   bool run(
           const Array2D<float> &, 
           const Array2D<unsigned char> &, 
           const Array2D<char> &);

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
