# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

alias enter='alienv enter O2/latest-dev-o2'

function sync-alice() {
  rsync -arvz -e'ssh -p 2020' --progress --delete ed@alihlt-gw-prod.cern.ch:/home/ed/jsons /home/ed
}

function query-alice() {
  while true
  do
    sync-alice
    sleep 2
  done
}

function o2eve() {
  o2-eve -o -d /home/ed/jsons
}

export ALIBUILD_WORK_DIR=~/alice/sw
eval "`alienv shell-helper`"
