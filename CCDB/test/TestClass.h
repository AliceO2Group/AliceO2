// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CCDB_TESTCLASS
#define O2_CCDB_TESTCLASS

// a private test class to be stored inside the CCDB; just used in a unit test
// the header is not public
struct TestClass {
  double mD = 1.;
};

#endif
