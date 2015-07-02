#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;


#pragma link C++ class AliceO2::TPC::Detector+;
#pragma link C++ class AliceO2::TPC::Point+;

#pragma link C++ class AliTPCRF1D-;                    // 1D Response Function (used for Time Response Function)
#pragma link C++ class AliTPCPRF2D-;                   // 2D Pad Response Function

#pragma link C++ class AliH2F+;                        // Additional functionality to 2D Histogram (used in Draw padResponse func)
                                                       // --- remove it, check miminal code needed for drawing
#pragma link C++ class AliDetectorParam+;              // Base class for AliTPCParam (before also used for TRD)
#pragma link C++ class AliTPCParam+;                   // Parameterize the Geometry, Diffusion, ResponseFunction, Default HV, ...
                                                       // Base class for AliTPCParamSR
#pragma link C++ class AliTPCParamSR-;                 // SR = Straight Rows
                                                       // --- In principle only 1 class of (AliDetectorParam, AliTPCParam,
                                                       //     AliTPCParamSR) is needed - can be merged, but breaks OCDB
#pragma link C++ class AliTPCROC+;                     // Geometry for 1 ROC (ReadOutChamber) - hardcoded
                                                       // --- (possible) duplication of AliTPCParam

#pragma link C++ class AliLog+;
#pragma link C++ class AliMathBase+;

#endif
