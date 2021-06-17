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

#pragma link C++ class o2::detectors::AlignParam + ;
#pragma link C++ class std::vector < o2::detectors::AlignParam> + ;

#pragma link C++ class o2::detectors::DetID + ;
#pragma link C++ class o2::detectors::MatrixCache < o2::math_utils::Transform3D> + ;
#pragma link C++ class o2::detectors::MatrixCache < o2::math_utils::Rotation2Df_t> + ;
#pragma link C++ class o2::detectors::DetMatrixCache + ;
#pragma link C++ class o2::detectors::DetMatrixCacheIndirect + ;

#pragma link C++ class o2::detectors::SimTraits + ;

#pragma link C++ class o2::base::NameConf + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::base::NameConf> + ;

#pragma link C++ class o2::ctf::CTFHeader + ;
#pragma link C++ class o2::ctf::Registry + ;
#pragma link C++ class o2::ctf::Block < uint32_t> + ;
#pragma link C++ class o2::ctf::Block < uint16_t> + ;
#pragma link C++ class o2::ctf::Block < uint8_t> + ;
#pragma link C++ class o2::ctf::Metadata + ;
#pragma link C++ class o2::ctf::ANSHeader + ;

#endif
