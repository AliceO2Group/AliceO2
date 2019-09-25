// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace o2;
#pragma link C++ namespace o2::mch;
#pragma link C++ namespace o2::mch::test;

#pragma link C++ class o2::mch::test::Dummy;
#pragma link C++ class o2::mch::Detector + ;
#pragma link C++ class o2::mch::Hit + ;
#pragma link C++ class std::vector < o2::mch::Hit> + ;
#pragma link C++ class o2::base::DetImpl < o2::mch::Detector> + ;
#pragma link C++ class o2::mch::Digit + ;
#pragma link C++ class std::vector < o2::mch::Digit> + ;

#pragma link C++ function o2::mch::createGeometry;
#pragma link C++ function o2::mch::getSensitiveVolumes;

#pragma link C++ function o2::mch::test::createStandaloneGeometry;
#pragma link C++ function o2::mch::test::drawGeometry;
#pragma link C++ function o2::mch::test::getRadio;
#pragma link C++ function o2::mch::test::showGeometryAsTextTree;
#pragma link C++ function o2::mch::test::setVolumeVisibility;
#pragma link C++ function o2::mch::test::setVolumeColor;

#endif
