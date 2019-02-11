#ifdef __CINT__
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ global gAlice;
#pragma link C++ global gMC;
 
#pragma link C++ enum VertexSmear_t;
#pragma link C++ enum VertexSource_t;

#pragma link C++ class  AliGenerator+;
#pragma link C++ class  AliVertexGenerator+;
#pragma link C++ class  AliRun-;
#pragma link C++ class  AliModule+;
#pragma link C++ class  AliDetector+;
#pragma link C++ class  AliDigit+;
#pragma link C++ class  AliHit+;
#pragma link C++ class  AliLego+;
#pragma link C++ class  AliLegoGenerator+;
#pragma link C++ class  AliLegoGeneratorXYZ+;
#pragma link C++ class  AliLegoGeneratorPhiZ+;
#pragma link C++ class  AliLegoGeneratorEta+;
#pragma link C++ class  AliLegoGeneratorEtaR+;
#pragma link C++ class  AliDigitNew+;
#pragma link C++ class  AliGeometry+;
#pragma link C++ class  AliRecPoint+;
#pragma link C++ class  AliHitMap+;
#pragma link C++ class  AliRndm+;
#pragma link C++ class  AliDebugVolume+;
#pragma link C++ class  AliConfig+;
#pragma link C++ class  AliDigitizer+;
#pragma link C++ class  AliDigitizationInput+;
#pragma link C++ class  AliStream+;
#pragma link C++ class  AliMergeCombi+;
#pragma link C++ class  AliGausCorr+;
#pragma link C++ class  AliLoader+;
#pragma link C++ class  AliDataLoader+;
#pragma link C++ class  AliBaseLoader+;
#pragma link C++ class  AliObjectLoader+;
#pragma link C++ class  AliTreeLoader+;
#pragma link C++ class  AliRunLoader+;
#pragma link C++ class  AliReconstructor+;
#pragma link C++ class  AliMC+;
#pragma link C++ class  AliSimulation+;
#pragma link C++ class  AliReconstruction+;
#pragma link C++ class  AliRecoInputHandler+;
#pragma link C++ class  AliVertexGenFile+;
#pragma link C++ class  AliVertexer+;

#pragma link C++ class AliTriggerDetector+;
#pragma link C++ class AliCentralTrigger+;
#pragma link C++ class AliTriggerUtils+;

#pragma link C++ class AliGeomManager+;
#pragma link C++ class AliAlignObj+;
#pragma link C++ class AliAlignObjParams+;
#pragma link C++ class AliAlignObjMatrix+;
#pragma link C++ class AliMisAligner+;

#pragma link C++ class AliTrackFitter+;
#pragma link C++ class AliTrackFitterRieman+;
#pragma link C++ class AliTrackFitterKalman+;
#pragma link C++ class AliTrackFitterStraight+;
#pragma link C++ class AliTrackResiduals+;
#pragma link C++ class AliTrackResidualsChi2+;
#pragma link C++ class AliTrackResidualsFast+;
#pragma link C++ class AliTrackResidualsLinear+;
#pragma link C++ class AliAlignmentTracks+;

#pragma link C++ class  AliRieman;

#pragma link C++ class AliTriggerDetector+;
#pragma link C++ class AliCentralTrigger+;
#pragma link C++ class AliCTPRawStream+;
#pragma link C++ class AliSignalProcesor+;
#pragma link C++ class  AliHelix+;
#pragma link C++ class  AliCluster+;
#pragma link C++ class  AliCluster3D+;
#pragma link C++ class  AliTracker+;
#pragma link C++ class  AliTrackleter+;
#pragma link C++ class  AliV0+;
#pragma link C++ class  AliKink+;

#pragma link C++ class  AliSelectorRL+;

#pragma link C++ class AliSurveyObj+;
#pragma link C++ class AliSurveyPoint+;
#pragma link C++ class AliSurveyToAlignObjs+;

#pragma link C++ class AliFstream+;
#pragma link C++ class AliCTPRawData+;

#pragma link C++ class AliQADataMaker+;
#pragma link C++ class AliQADataMakerSim+;
#pragma link C++ class AliQADataMakerRec+;
#pragma link C++ class AliCorrQADataMakerRec+;
#pragma link C++ class AliGlobalQADataMaker+;
#pragma link C++ class AliQAManager+;
#pragma link C++ class AliQAChecker+;
#pragma link C++ class AliCorrQAChecker+;
#pragma link C++ class AliGlobalQAChecker+;
#pragma link C++ class AliQACheckerBase+;
#pragma link C++ class AliQAThresholds+;
#pragma link C++ class AliMillepede+;

#pragma link C++ class AliPlaneEff+;

#pragma link C++ class AliTriggerRunScalers+;
#pragma link C++ class AliGRPPreprocessor+;
#pragma link C++ class AliGRPRecoParam+;

#pragma link C++ class AliRelAlignerKalman+;

#pragma link C++ class AliESDTagCreator+;

#pragma link C++ class AliGRPObject+;

#pragma link C++ class AliQAv1+;

#pragma link C++ class AliRunInfo+;
#pragma link C++ class AliEventInfo+;
#pragma link C++ class AliDetectorRecoParam+;
#pragma link C++ class AliRecoParam+;

#pragma link C++ class AliMillePede2+;
#pragma link C++ class AliMillePedeRecord+;
#pragma link C++ class AliMinResSolve+;
#pragma link C++ class AliMatrixSparse+;
#pragma link C++ class AliVectorSparse+;
#pragma link C++ class AliMatrixSq+;
#pragma link C++ class AliSymMatrix+;
#pragma link C++ class AliSymBDMatrix+;
#pragma link C++ class AliRectMatrix+;
#pragma link C++ class AliParamSolver+;

#pragma link C++ class AliGRPManager+;
#pragma link C++ class AliDCSArray+; 	 
#pragma link C++ class AliLHCReader+;
#pragma link C++ class AliCTPTimeParams+;
#pragma link C++ class AliCTPInputTimeParams+;

#pragma link C++ class AliLHCDipValT<Double_t>+; 	 
#pragma link C++ class AliLHCDipValT<Int_t>+; 	 
#pragma link C++ class AliLHCDipValT<Float_t>+; 	 
#pragma link C++ class AliLHCDipValT<Char_t>+; 	 
#pragma link C++ class AliLHCData+;
#pragma link C++ class AliLHCClockPhase+;

#pragma link C++ class AliLTUConfig+;

#pragma link C++ class AliTransportMonitor+;
#pragma link C++ class AliTransportMonitor::AliTransportMonitorVol+;
#pragma link C++ struct AliTransportMonitor::AliTransportMonitorVol::AliPMonData+;

#pragma link C++ class AliParamList+;

#pragma link C++ typedef AliLHCDipValD; 	 
#pragma link C++ typedef AliLHCDipValI; 	 
#pragma link C++ typedef AliLHCDipValF; 	 
#pragma link C++ typedef AliLHCDipValC;

#pragma link C++ class AliMCGenHandler+;
#pragma link C++ class  AliHLTVEventInputHandler+;

#pragma link C++ class AliHLTSimulation+;

#pragma link C++ class AliLumiTools;
#pragma link C++ class AliLumiRef+;


#endif
