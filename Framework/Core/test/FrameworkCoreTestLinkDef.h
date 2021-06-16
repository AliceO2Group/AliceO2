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

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::test::TriviallyCopyable + ;
#pragma link C++ class o2::test::Base + ;
#pragma link C++ class o2::test::Polymorphic + ;
#pragma link C++ class o2::test::SimplePODClass + ;
#pragma link C++ class std::vector < o2::test::TriviallyCopyable > +;
#pragma link C++ class std::vector < o2::test::Polymorphic > +;

#pragma link C++ class StepTHn + ;
#pragma link C++ class StepTHnT < TArrayF> + ;
#pragma link C++ class StepTHnT < TArrayD> + ;
#pragma link C++ typedef StepTHnF;
#pragma link C++ typedef StepTHnD;
