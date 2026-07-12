# Commands in operations

## Training on cloud

launch a session for training or other tasks (to avoid thread being killed by terminal)

```bash
tmux new -s train
```

get back to normal bash

```
ctrl+b, then d
```

reopen (attach) tmux

```bash
tmux attach -t train
```

check all running sessions in the background

```bash
tmux list-sessions
```

move the checkpoints to persistent volume every 30min

```bash
while true; do rsync -av --remove-source-files /mlfield/checkpoints/ /workspace/store/; sleep 1800; done
```
