
#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::Base::Detector+;
#pragma link C++ class o2::Base::Track::TrackParBase+;
#pragma link C++ class o2::Base::Track::TrackPar+;
#pragma link C++ class o2::Base::Track::TrackParCov+;
#pragma link C++ class o2::Base::TrackReference+;

// this is used for the test only, should it be separate LinkDef?
#pragma link C++ class o2::Base::ContVec<o2::Base::Track::TrackPar,int>-;
#pragma link C++ class vector<o2::Base::Track::TrackPar>+;

#endif
