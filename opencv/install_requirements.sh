

if grep -s -q "Mendel" /etc/os-release; then
  MENDEL_VER="$(cat /etc/mendel_version)"
  if [[ "$MENDEL_VER" == "1.0" || "$MENDEL_VER" == "2.0" || "$MENDEL_VER" == "3.0" ]]; then
    echo "Your version of Mendel is not compatible with OpenCV."
    echo "You must upgrade to Mendel 4.0 or higher."
    exit 1
  fi
  sudo apt install python3-opencv
elif grep -s -q "Raspberry Pi" /sys/firmware/devicetree/base/model; then
  RASPBIAN=$(grep VERSION_ID /etc/os-release | sed 's/VERSION_ID="\([0-9]\+\)"/\1/')
  echo "Raspbian Version: $RASPBIAN"
  if [[ "$RASPBIAN" -ge "10" ]]; then
    # Lock to version due to bug: https://github.com/piwheels/packages/issues/59
    sudo pip3 install opencv-contrib-python==4.1.0.25
    sudo apt-get -y install libjasper1 libhdf5-1* libqtgui4 libatlas-base-dev libqt4-test
  else
    echo "For Raspbian versions older than Buster (10) you have to build OpenCV yourself"
    echo "or install the unofficial opencv-contrib-python package."
    exit 1
  fi
else
  sudo apt install python3-opencv
fi
