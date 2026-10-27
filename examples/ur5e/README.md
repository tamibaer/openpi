Start Policy Server:

```bash
uv run scripts/serve_policy.py policy:checkpoint     
    --policy.config=pi05_ur5e_lora     
    --policy.dir=checkpoints/pi05_ur5e_lora/blocks_v1/19999
```

Run Policy on UR5e Client:

```bash
~/venvs/ur5e_client/bin/python examples/ur5e/ur5e_policy_client.py 
    --mode stream --max-chunks 200   
    --slowdown 2 
    --exec-steps 8 
    --max-step 0.06 
    --max-total 10   
    --prompt "pick up the strawberry and put it in the blue box"
```