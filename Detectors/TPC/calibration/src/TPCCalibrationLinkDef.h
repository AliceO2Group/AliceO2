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

#pragma link C++ class o2::tpc::CalibRawBase;
#pragma link C++ class o2::tpc::CalibPedestal;
#pragma link C++ class o2::tpc::CalibPedestalParam +;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::CalibPedestalParam> + ;
#pragma link C++ class o2::tpc::CalibPulser;
#pragma link C++ class o2::tpc::CalibPulserParam +;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::CalibPulserParam> + ;
#pragma link C++ class o2::tpc::CalibTreeDump;
#pragma link C++ class o2::tpc::DigitDump;
#pragma link C++ class o2::tpc::DigitDumpParam;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::DigitDumpParam> + ;
#pragma link C++ class o2::tpc::CalibPadGainTracks;
#pragma link C++ class o2::tpc::FastHisto<float> +;
#pragma link C++ class o2::tpc::FastHisto<unsigned int> +;
#pragma link C++ class o2::tpc::IDCGroup +;
#pragma link C++ class o2::tpc::IDCGroupHelperRegion +;
#pragma link C++ class o2::tpc::IDCGroupHelperSector +;
#pragma link C++ struct o2::tpc::ParameterIDCGroup;
#pragma link C++ struct o2::tpc::ParameterIDCCompression;
#pragma link C++ class o2::tpc::IDCAverageGroup +;
#pragma link C++ class o2::tpc::IDCFactorization +;
#pragma link C++ struct o2::tpc::IDCDelta<float> +;
#pragma link C++ struct o2::tpc::IDCDelta<short> +;
#pragma link C++ struct o2::tpc::IDCDelta<char> +;
#pragma link C++ struct o2::tpc::IDCDeltaCompressionFactors +;
#pragma link C++ struct o2::tpc::IDCDeltaContainer<float> +;
#pragma link C++ struct o2::tpc::IDCDeltaContainer<short> +;
#pragma link C++ struct o2::tpc::IDCDeltaContainer<char> +;
#pragma link C++ class o2::tpc::IDCDeltaCompressionHelper<short> +;
#pragma link C++ class o2::tpc::IDCDeltaCompressionHelper<char> +;
#pragma link C++ struct o2::tpc::IDCZero +;
#pragma link C++ struct o2::tpc::IDCOne +;
#pragma link C++ struct o2::tpc::OneDIDC +;
#pragma link C++ class o2::tpc::OneDIDCAggregator +;
#pragma link C++ struct o2::tpc::FourierCoeff +;
#pragma link C++ struct o2::tpc::ParameterIDCGroupCCDB +;
#pragma link C++ class o2::tpc::RobustAverage +;
#pragma link C++ class o2::tpc::IDCFourierTransform +;
#pragma link C++ class o2::tpc::IDCCCDBHelper<float> +;
#pragma link C++ class o2::tpc::IDCCCDBHelper<short> +;
#pragma link C++ class o2::tpc::IDCCCDBHelper<char> +;

#endif
