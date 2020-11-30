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

#pragma link C++ class o2::utils::TreeStream + ;
#pragma link C++ class o2::utils::TreeStreamRedirector + ;
#pragma link C++ class o2::utils::RootChain + ;
#pragma link C++ class o2::utils::RngHelper;
#pragma link C++ class o2::utils::MemFileHelper + ;
#pragma link C++ class o2::utils::RootSerializableKeyValueStore::SerializedInfo + ;
#pragma link C++ class map < string, o2::utils::RootSerializableKeyValueStore::SerializedInfo*> + ;
#pragma link C++ class o2::utils::RootSerializableKeyValueStore + ;

#endif
