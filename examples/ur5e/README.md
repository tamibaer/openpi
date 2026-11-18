Start Policy Server:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_ur5e_lora --policy.dir=checkpoints/pi05_ur5e_lora/blocks_v1/19999
```

Run Policy on UR5e Client:

```bash
~/venvs/ur5e_client/bin/python examples/ur5e/ur5e_policy_client.py --prompt "pick up the red block and put it in the blue box"
```