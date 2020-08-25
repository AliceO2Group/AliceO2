o2_add_dpl_workflow(table-maker
                    SOURCES tableMaker.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(table-reader
                    SOURCES tableReader.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore
                    COMPONENT_NAME Analysis)
