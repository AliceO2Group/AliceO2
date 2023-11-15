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

/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::dataformats::TimeStamp < float> + ;
#pragma link C++ class o2::dataformats::TimeStamp < double> + ;
#pragma link C++ class o2::dataformats::TimeStamp < int> + ;
#pragma link C++ class o2::dataformats::TimeStamp < Float16_t > + ;
#pragma link C++ class o2::dataformats::TimeStampWithError < float, float> + ;
#pragma link C++ class o2::dataformats::TimeStampWithError < double, double> + ;
#pragma link C++ class o2::dataformats::TimeStampWithError < int, int> + ;

#pragma link C++ class o2::dataformats::EvIndex < int, int> + ;
#pragma link C++ class o2::dataformats::RangeReference < int, int> + ;
#pragma link C++ class o2::dataformats::RangeReference < uint32_t, uint16_t> + ;
#pragma link C++ class o2::dataformats::RangeReference < o2::dataformats::EvIndex < int, int>, int> + ;

#pragma link C++ class o2::dataformats::RangeRefComp < 4> + ; // reference to a set with 15 entries max (ITS clusters)
#pragma link C++ class o2::dataformats::RangeRefComp < 5> + ; // reference to a set with 15 entries max (ZDC BCData)
#pragma link C++ class o2::dataformats::RangeRefComp < 6> + ; // reference to a set with 63 entries max (FV0 Digit)

#pragma link C++ class o2::dataformats::RangeRefComp < 8> + ; //

#pragma link C++ class std::vector < o2::dataformats::RangeReference < int, int>> + ;

#pragma link C++ class o2::dataformats::AbstractRef < 25, 5, 2> + ;

#pragma link C++ class o2::InteractionRecord + ;
#pragma link C++ class o2::InteractionTimeRecord + ;
#pragma link C++ class o2::BunchFilling + ;
#pragma link C++ class std::pair < long, o2::BunchFilling> + ;

#pragma link C++ class o2::math_utils::detail::Bracket < o2::InteractionRecord> + ;
#pragma link C++ class o2::dataformats::IRFrame + ;
#pragma link C++ class std::vector < o2::dataformats::IRFrame> + ;

#pragma link C++ class o2::dataformats::FlatHisto1D < float> + ;
#pragma link C++ class o2::dataformats::FlatHisto1D < double> + ;
#pragma link C++ class o2::dataformats::FlatHisto1D_f + ;
#pragma link C++ class o2::dataformats::FlatHisto1D_d + ;

#pragma link C++ class o2::dataformats::FlatHisto2D < float> + ;
#pragma link C++ class o2::dataformats::FlatHisto2D < double> + ;
#pragma link C++ class o2::dataformats::FlatHisto2D_f + ;
#pragma link C++ class o2::dataformats::FlatHisto2D_d + ;

#pragma link C++ class o2::dataformats::Pair < float, float> + ;
#pragma link C++ class std::vector < o2::dataformats::Pair < float, float>> + ;

#pragma link C++ class o2::dataformats::Triplet < float, float, float> + ;
#pragma link C++ class std::vector < o2::dataformats::Triplet < float, float, float>> + ;

#include "CommonDataFormat/TFIDInfo.h"
#pragma link C++ class o2::dataformats::TFIDInfo + ;
#pragma link C++ class std::vector < o2::dataformats::TFIDInfo> + ;

#endif
