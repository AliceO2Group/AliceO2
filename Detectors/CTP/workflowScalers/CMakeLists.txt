o2_add_executable(
                  proxy
                  COMPONENT_NAME ctp
                  SOURCES src/ctp-proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow
                  O2::DataFormatsCTP)
