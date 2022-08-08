o2_add_executable(
                  proxy
                  COMPONENT_NAME ctp
                  SOURCES src/ctp-proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow
                  O2::DataFormatsCTP)
o2_add_executable(
                  qc-proxy
                  COMPONENT_NAME ctp
                  SOURCES src/ctp-qc-proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow
                  O2::DataFormatsCTP)
