#! /bin/bash

environment_scripts=$(dirname "$0")
. $environment_scripts/install_mmposewholebody.sh
. $environment_scripts/install_mediapipe.sh
. $environment_scripts/install_openpifpaf.sh