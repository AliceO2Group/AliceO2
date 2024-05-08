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

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::fwdalign::MatrixSparse + ;
#pragma link C++ class o2::fwdalign::MatrixSq + ;
#pragma link C++ class o2::fwdalign::MillePede2 + ;
#pragma link C++ class o2::fwdalign::MillePedeRecord + ;
#pragma link C++ class std::vector < o2::fwdalign::MillePedeRecord> + ;
#pragma link C++ class o2::fwdalign::MilleRecordReader + ;
#pragma link C++ class o2::fwdalign::MilleRecordWriter + ;
#pragma link C++ class o2::fwdalign::MinResSolve + ;
#pragma link C++ class o2::fwdalign::RectMatrix + ;
#pragma link C++ class o2::fwdalign::SymBDMatrix + ;
#pragma link C++ class o2::fwdalign::SymMatrix + ;
#pragma link C++ class o2::fwdalign::VectorSparse + ;

#endif
