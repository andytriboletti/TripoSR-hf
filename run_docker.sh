docker run --privileged --network=host --gpus all \
  -v /mnt/c:/mnt/c \
  -v /mnt/c/Users/Andy/OneDrive/Documents/GitHub/midnightcoffee-python:/workspace1 \
  -v /mnt/c/Users/Andy/OneDrive/Documents/GitHub/openspace-python:/workspace2 \
  -v /mnt/c/Users/Andy/OneDrive/Documents/GitHub/wordcraft3d-python:/workspace3 \
  -v /mnt/c/Users/Andy/OneDrive/Documents/GitHub/wizardwriter-python:/workspace4 \
  -v /mnt/c/Users/Andy/OneDrive/Documents/GitHub/lunarlounge-python:/workspace5 \
  -v /mnt/e/hf-cache:/cache:Z \
  --user $(id -u):$(id -g) \
  --interactive --tty triposr /bin/bash
