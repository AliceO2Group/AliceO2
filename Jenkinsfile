#!groovy

node {
  stage "Verify author"
  def power_users = [
    "Barthelemy",
    "MohammadAlTurany",
    "PatrykLesiak",
    "bovulpes",
    "dberzano",
    "iouribelikov",
    "ktf",
    "matthiasrichter",
    "mkrzewic",
    "mpuccio",
    "rbx",
    "sawanzel",
    "wiechula"
  ]
  echo "Changeset from " + env.CHANGE_AUTHOR
  if (power_users.contains(env.CHANGE_AUTHOR)) {
    currentBuild.displayName = "Testing ${env.BRANCH_NAME} from ${env.CHANGE_AUTHOR}"
    echo "PR comes from power user. Testing"
  } else {
    currentBuild.displayName = "Feedback needed for ${env.BRANCH_NAME} from ${env.CHANGE_AUTHOR}"
    input "Do you want to test this change?"
  }
  currentBuild.displayName = "Testing ${env.BRANCH_NAME} from ${env.CHANGE_AUTHOR}"

  stage "Build AliceO2"
  def test_script = '''
      rm -fr alibuild alidist
      git clone https://github.com/alisw/alibuild
      git clone -b IB/v5-08/o2 https://github.com/alisw/alidist
      x=`date +"%s"`
      WORKAREA=/build/workarea/pr/`echo $(( $x / 3600 / 24 / 7))`

      # Make sure we have only one builder per directory
      CURRENT_SLAVE=unknown
      while [[ "$CURRENT_SLAVE" != '' ]]; do
        WORKAREA_INDEX=$((WORKAREA_INDEX+1))
        CURRENT_SLAVE=$(cat $WORKAREA/$WORKAREA_INDEX/current_slave 2> /dev/null || true)
        [[ "$CURRENT_SLAVE" == "$NODE_NAME" ]] && CURRENT_SLAVE=
      done

      mkdir -p $WORKAREA/$WORKAREA_INDEX
      echo $NODE_NAME > $WORKAREA/$WORKAREA_INDEX/current_slave

      alibuild/aliBuild --work-dir $WORKAREA/$WORKAREA_INDEX               \
                        --reference-sources /build/mirror                  \
                        --debug                                            \
                        --jobs 16                                          \
                        --remote-store rsync://repo.marathon.mesos/store/  \
                        --disable DDS                                      \
                        -d build O2 || BUILDERR=$?

      rm -f $WORKAREA/$WORKAREA_INDEX/current_slave
      if [ ! "X$BUILDERR" = X ]; then
        exit $BUILDERR
      fi
    '''

  currentBuild.displayName = "Testing ${env.BRANCH_NAME}"
  parallel(
    "slc7": {
      node ("slc7_x86-64-large") {
        dir ("O2") {
          checkout scm
        }
        withEnv (["CHANGE_TARGET=${env.CHANGE_TARGET}"]) {
          sh test_script
        }
      }
    }
  )
}
