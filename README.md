# Reproducing Experiments

## Requirements

- Python **3.9**
- Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## MuJoCo Experiments

To reproduce the results on **MuJoCo** environments:

1. **Launch the acceleration stage**:

   ```bash
   python acceleration.py --env <environment_name>
   ```

2. **Launch the fine-tune stage**:

   ```bash
   python online_finetune.py --env <environment_name>
   ```

You can modify model or training parameters directly inside the corresponding Python files (`acceleration.py`, `online_finetune.py`) or in the configuration file `sh_config.yaml`.

---

## ManiSkill (Image-based) Experiments

To reproduce the results on **image-based ManiSkill** tasks:

1. Install all required packages as described in the official [ManiSkill installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html).

2. Use the following scripts:

   - **Acceleration stage**:
     ```bash
     python rgb_acceleration.py --env_id <environment_name>
     ```

   - **Fine-tune stage**:
     ```bash
     python rgb_online_finetune.py --env_id <environment_name>
     ```