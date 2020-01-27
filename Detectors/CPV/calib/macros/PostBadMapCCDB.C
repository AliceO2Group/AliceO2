// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TRandom.h"
#include "CCDB/CcdbApi.h"
#include "CPVCalib/BadChannelMap.h"
#include "CPVBase/Geometry.h"
#endif
void PostBadMapCCDB()
{

  //Post test bad map for CPV to test CCDB

  o2::ccdb::CcdbApi ccdb;
  std::map<std::string, std::string> metadata; // do we want to store any meta data?
  ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation

  auto o2cpvBM = new o2::cpv::BadChannelMap();

  int nBad = 50;

  for (int i = 0; i < nBad; i++) {
    unsigned short channelID = gRandom->Uniform(56 * 64 * 3.5); //Random bad channels in 3.5 CPV modules
    o2cpvBM->addBadChannel(channelID);
  }

  ccdb.storeAsTFileAny(o2cpvBM, "CPV/BadMap", metadata, 1, 1670700184549); // one year validity time
}