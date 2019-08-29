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

#pragma link C++ class o2::ccdb::IdPath + ;
#pragma link C++ class o2::ccdb::IdRunRange + ;
#pragma link C++ class o2::ccdb::ConditionId + ;
#pragma link C++ class o2::ccdb::ConditionMetaData + ;
#pragma link C++ class o2::ccdb::Condition + ;
#pragma link C++ class o2::ccdb::Storage + ;
#pragma link C++ class o2::ccdb::StorageFactory + ;
#pragma link C++ class o2::ccdb::Manager + ;
#pragma link C++ class o2::ccdb::StorageParameters + ;
#pragma link C++ class o2::ccdb::LocalStorage + ;
#pragma link C++ class o2::ccdb::LocalStorageFactory + ;
#pragma link C++ class o2::ccdb::LocalStorageParameters + ;
#pragma link C++ class o2::ccdb::FileStorage + ;
#pragma link C++ class o2::ccdb::FileStorageFactory + ;
#pragma link C++ class o2::ccdb::FileStorageParameters + ;
#pragma link C++ class o2::ccdb::GridStorage + ;
#pragma link C++ class o2::ccdb::GridStorageFactory + ;
#pragma link C++ class o2::ccdb::GridStorageParameters + ;
#pragma link C++ class o2::ccdb::XmlHandler + ;
#pragma link C++ class o2::ccdb::CcdbApi + ;
#pragma link C++ class o2::ccdb::BasicCCDBManager + ;

/// for the unit test
#pragma link C++ class TestClass + ;
#pragma link C++ class o2::TObjectWrapper < TestClass> + ;

#endif
