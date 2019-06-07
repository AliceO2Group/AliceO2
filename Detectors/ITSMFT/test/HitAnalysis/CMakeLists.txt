o2_add_library(HitAnalysis
               SOURCES
               src/HitAnalysis.cxx
               PUBLIC_LINK_LIBRARIES
               FairRoot::Base
               O2::ITSMFTSimulation
               O2::ITSBase)
o2_target_root_dictionary(HitAnalysis HEADERS include/HitAnalysis/HitAnalysis.h)
