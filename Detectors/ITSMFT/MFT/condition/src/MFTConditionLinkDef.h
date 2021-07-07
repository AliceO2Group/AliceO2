#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ struct std::pair < uint64_t, double> + ;
#pragma link C++ struct o2::mft::MFTDCSinfo + ;
#pragma link C++ class std::unordered_map < o2::dcs::DataPointIdentifier, o2::mft::MFTDCSinfo> + ;

#endif
