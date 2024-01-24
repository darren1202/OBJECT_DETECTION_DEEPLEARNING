

if grep -s -q "Mendel" /etc/os-release; then
  echo "Installing DevBoard specific dependencies"
  sudo apt-get install -y python3-pygame
else
  sudo apt-get install -y libsdl-image1.2-dev libsdl-ttf2.0-dev libatlas-base-dev
  sudo pip3 install pygame
fi

