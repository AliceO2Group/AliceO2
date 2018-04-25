include						config_common.mak

TARGET						= libO2TPCCATracking
TARGETTYPE					= LIB

CXXFILES					+= interface/AliHLTTPCCAO2Interface.cxx \
						   $(HLTCA_STANDALONE_CXXFILES) \
						   $(HLTCA_MERGER_CXXFILES)

ALLDEP						+= config_common.mak

DEFINES						+= HLTCA_TPC_GEOMETRY_O2 HLTCA_BUILD_O2_LIB
