o2_add_library(PWGDQCore
               SOURCES
               PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore)

o2_target_root_dictionary(PWGDQCore
              HEADERS CutsLibrary.h HistogramsLibrary.h
              LINKDEF PWGDQCoreLinkDef.h)

o2_add_dpl_workflow(table-maker
                    SOURCES tableMaker.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(table-reader
                    SOURCES tableReader.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(dilepton-ee
                    SOURCES dileptonEE.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(dilepton-mumu
                    SOURCES dileptonMuMu.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(table-maker-pp
                    SOURCES tableMaker_pp.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                    COMPONENT_NAME Analysis)
                  
o2_add_dpl_workflow(dq-filter-pp
                    SOURCES filterPP.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                    COMPONENT_NAME Analysis)                    

o2_add_dpl_workflow(table-maker-muon-pp
                    SOURCES tableMakerMuon_pp.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                    COMPONENT_NAME Analysis)
