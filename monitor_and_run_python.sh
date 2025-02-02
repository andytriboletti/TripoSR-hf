tmux new-session -d 'cd workspace1 && python upper.py' \; \
  split-window -h 'cd workspace2 && python upper.py' \; \
  split-window -v 'cd workspace3 && python upper.py' \; \
  split-window -v 'cd workspace4 && python upper.py' \; \
  select-layout tiled \; \
  attach
