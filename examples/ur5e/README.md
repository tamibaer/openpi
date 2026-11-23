# PI0.5
Start Policy Server:
```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_ur5e_lora --policy.dir=checkpoints/pi05_ur5e_lora/blocks_v1/19999
```

Run Policy on UR5e Client:
```bash
~/venvs/ur5e_client/bin/python examples/ur5e/ur5e_policy_client.py --prompt "pick up the red block and put it in the blue box"
```

# PI0-FAST
Start Robot:
```bash
ros2 launch robot_table_control bringup_real_robot.launch.py use_admittance:=true
```

Start Cameras:
```bash
ros2 launch robot_table_control cameras.launch.py
```
Start SSH Tunnel: 
```bash
ssh -N -L 8000:localhost:8000 bergerhofft@<server-ip>
```

Start Policy Server (on the cluster):
```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_ur5e_full_v2 --policy.dir=checkpoints/pi0_fast_ur5e_full_v2/v1/19999
```

Start Client:
```bash
~/venvs/ur5e_client/bin/python examples/ur5e/policy_client_pi0_fast.py     --prompt "pick up the green cube and put it in the cardboard box"
```