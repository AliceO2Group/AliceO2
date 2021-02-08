#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::calibration::TimeSlot < o2::itsmft::CompClusterExt > +;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::itsmft::CompClusterExt, o2::itsmft::NoiseMap > +;

#endif
