// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterFinderGEM.cxx
/// \brief Definition of a class to reconstruct clusters with the original MLEM algorithm
///
/// The original code is in AliMUONClusterFinderMLEM and associated classes.
/// It has been re-written in an attempt to simplify it without changing the results.
///
/// \author Philippe Pillot, Subatech


#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
// GG
#include <iostream>


#include <FairMQLogger.h>

#include "MCHClustering/ClusterDump.h"


namespace o2
{
namespace mch
{

//_________________________________________________________________________________________________
ClusterDump::ClusterDump(const char *str, int mode_)
{
  fileName= str;
  mode = mode_;
  if (mode == 1) {
    dumpFiles[0].open( str, std::fstream::out | std::fstream::app | std::ios_base::binary );
  }
}

//_________________________________________________________________________________________________
ClusterDump::~ClusterDump(){
    // The dump file is not close a the end of processing
    if (mode == 1) {
        std::cout << "Close the file ??????????????"  << std::endl;  
        dumpFiles[0].close();  
    }
}

//_________________________________________________________________________________________________
void ClusterDump::flush(){
    if (mode==1) {
        dumpFiles[0].flush();  
    }
}

void ClusterDump::dumpFloat32( int ifile, long size, const float_t * data) {
  if (mode==1) {
    dumpFiles[ifile].write( (char *) &size, sizeof(long));
    dumpFiles[ifile].write( (char *) data, sizeof(float)*size );
  }
}

void ClusterDump::dumpFloat64( int ifile, long size, const double_t * data) {
  if (mode==1) {
    dumpFiles[ifile].write( (char *) & size, sizeof(long) );
    dumpFiles[ifile].write( (char *) data, sizeof(double)*size );
  }
}

void ClusterDump::dumpInt32( int ifile, long size, const int32_t * data) {
  if (mode==1) {
    dumpFiles[ifile].write( (char *) &size, sizeof(long));
    dumpFiles[ifile].write( (char *) data, sizeof(int32_t)*size );
  }
}

void ClusterDump::dumpUInt32( int ifile, long size, const uint32_t * data) {
  if (mode==1) {
    dumpFiles[ifile].write( (char *) &size, sizeof(long));
    dumpFiles[ifile].write( (char *) data, sizeof(uint32_t)*size );
  }
}

void ClusterDump::dumpInt16( int ifile, long size, const int16_t * data) {
  if (mode==1) {
    dumpFiles[ifile].write( (char *) &size, sizeof(long));
    dumpFiles[ifile].write( (char *) data, sizeof(int16_t)*size );
  }
}

} // namespace mch
} // namespace o2
