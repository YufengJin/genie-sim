# Genie Sim 3.0 User Guide

## 1. Overview

Genie Sim is the simulation platform from AgiBot. It provides developers with a complete toolchain for digital asset generation, scene generalization, data collection, and automated evaluation. Its core module, Genie Sim Benchmark is a standardized tool dedicated to establishing the most accurate and authoritative evaluation for embodied intelligence.

The platform integrates 3D reconstruction with visual generation to create a high-fidelity simulation environment. It pioneers LLM-driven technology to generate vast simulation scenes and evaluation metrics in minutes. The evaluation system covers 200+ tasks across 100,000+ scenarios to build a comprehensive capability profile for models. Genie Sim also opens over 10,000 hours simulation dataset including real-world robot operation scenarios.

The platform will significantly accelerate model development, reduce reliance on physical hardware, and empower innovation in embodied intelligence. Simulation assets, data, and code are fully open source.

---

## 2. Quick Start

### 2.1 Requirements


|         | Minimum                                    | Tested                          |
| ------- | ------------------------------------------ | ------------------------------- |
| OS      | Ubuntu 22.04/24.04                         | Ubuntu 22.04                    |
| Sim     | NVIDIA Isaac Sim 5.1.0                     | NVIDIA Isaac Sim 5.1.0          |
| CPU     | Intel Core i7 (7th Generation) AMD Ryzen 5 | Intel Core i7 (12th Generation) |
| RAM     | 32GB                                       | 64GB                            |
| GPU     | GeForce RTX 4080                           | GeForce RTX 4090D               |
| Driver  | 580.65.06                                  | 550.120 + CUDA 12.4             |
| Storage | 50GB SSD                                   | 1TB NVMe SSD                    |


Reference: [https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)

### 2.2 Download

**Genie Sim**

```bash
git clone https://github.com/AgibotTech/genie_sim.git
```

**Genie Sim Assets**

Please visit the link below and follow the instructions to download Genie Sim assets, and put the assets into `genie_sim/source/geniesim/assets`:

[https://modelscope.cn/datasets/agibot_world/GenieSimAssets](https://modelscope.cn/datasets/agibot_world/GenieSimAssets)

> **NOTE:** Assets will be uploaded on Huggingface soon.

### 2.3 Installation

#### 2.3.1 Docker Container (recommended)

1. **Develop using docker container**
  - Install docker following Isaac Sim Documentation
2. **Prepare docker image**

> **NOTE:** Genie Sim Benchmark dropped support for curobo since 3.0

```bash
cd genie_sim
genie_sim$ docker build -f ./scripts/dockerfile -t registry.a  # create docker image from dockerfile
```

1. **Launch docker container and run the demo**

Make sure GenieSimAssets is downloaded at `genie_sim/source/geniesim/assets`

```bash
cd genie_sim
genie_sim$ ./scripts/start_gui.sh       # start a new container in repo root
genie_sim$ ./scripts/into.sh            # open a new terminal, into container
```

```bash
/geniesim/main$ omni_python source/geniesim/app/app.py --conf  # inside container, run the demo
```

#### 2.3.2 (Optional) Use Genie Sim as Python Module

> **NOTE:** This is for `import geniesim` from other workspace, geniesim is tested only in conda python3.11

```bash
cd genie_sim
genie_sim$ conda create --name geniesim python=3.11   # prepare conda env
genie_sim$ conda activate geniesim
genie_sim$ python -m pip install -e source/./          # cd in genie_sim root dir
```

#### 2.3.3 Host Machine

1. We **STRONGLY** recommend developers using our tested docker container environment for development.
2. If you wish to use your own environment:
  - For docker users, please refer to the dockerfile we provide and install the dependencies that we list.
  - For conda users, please refer to the geniesim as python module section.

#### 2.3.4 Developer Guide

##### 2.3.4.1 Enable pre-commit hooks for collaboration (optional)

> **NOTE:** Make sure dependencies in `requirements.txt` are properly installed, the pre-commit hook will be triggered once `git commit` is involved.

1. Install and setup pre-commit to enable auto file-formatter, python / json / yaml etc.

```bash
genie_sim$ pip install pre-commit        # install pre-commit to your python env
genie_sim$ pre-commit install            # enable pre-defined pre-commit-hooks within repo
```

1. Trigger file-formatter for all tracked files

```bash
genie_sim$ pre-commit run --all-files
```

---

## 3. Tutorials

### 3.1 Benchmark Evaluation

#### 3.1.1 Run Inference

1. **Code Setup**

```bash
genie_sim$ ./scripts/infer_setup.sh
```

1. **Download checkpoints of the demo tasks**

Four checkpoints of demo tasks are released and can be downloaded to specific local path on [https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset) under `checkpoints` as shown in the chart below.


| DEMO_TASK_NAME | LOCAL_PATH                                     |
| -------------- | ---------------------------------------------- |
| select_color   | `genie_sim/openpi/checkpoints/select_color/`   |
| recognize_size | `genie_sim/openpi/checkpoints/recognize_size/` |
| grasp_targets  | `genie_sim/openpi/checkpoints/grasp_targets/`  |
| organize_items | `genie_sim/openpi/checkpoints/organize_items/` |


1. **Docker setup for inference**
  a. Pull the base docker image
   b. Build the docker image for openpi
   Reference: [https://github.com/PhysicalIntelligence/openpi/blob/main/docs/docker.md](https://github.com/PhysicalIntelligence/openpi/blob/main/docs/docker.md)
   c. Create docker container - `pi05_infer`
2. **Start inference service**

The inference service is deployed on localhost by default. You can also start a service on other machines and set the corresponding IP and port in benchmark configuration files for distributed communication.

```bash
genie_sim$ ./scripts/start_infer.sh
# start inference
# uv run --no-sync scripts/serve_policy.py --host='0.0
```

#### 3.1.2 Run Benchmark

1. **Run docker container**

```bash
genie_sim$ ./scripts/start_gui.sh
genie_sim$ ./scripts/into.sh
```

1. **Run a demo benchmark task**

Execute the following command under genie_sim root directory in docker container:

```bash
/geniesim/main$ c  # demo select_color
/geniesim/main$ omni_python source/geniesim/app/app.py --conf  # demo size_recognize
/geniesim/main$ omni_python source/geniesim/app/app.py --conf  # demo grasp_targets
/geniesim/main$ omni_python source/geniesim/app/app.py --conf  # demo organize_items
```

1. **Run all benchmark tasks**
  a. Enter project root directory in container: `cd /geniesim/main`
   b. Set simulation config at path: `/geniesim/main/scripts/run_tasks.sh`
   c. Set OpenAI configs for LLM/VLM Usage
   Export the following environment variables in the container:

  | Var Name | Default Setting                                      | Usage                     |
  | -------- | ---------------------------------------------------- | ------------------------- |
  | BASE_URL | `https://dashscope.aliyuncs.com/compatible-mode/v1/` | /                         |
  | API_KEY  | Your Api Key                                         | /                         |
  | MODEL    | qwen3-max                                            | for evaluation generation |
  | VL_MODEL | qwen3-vl-plus                                        | for auto scoring          |

   d. Start all simulations
   Start all simulation task in series with command:
   e. Stop all simulations
   Stop the whole evaluation process with command:
   f. Collect Scores
   Collect all task scores, evaluation results will be stored at `./output/task_scores.csv`:
   g. Plot evaluation results with command:

#### 3.1.3 Record

Camera images from robots and environments can be recorded and converted to webm format with following instructions:

1. The `enable_ros` and `record` config should be set `True` in task config file `{task_name}.yml` at `source/geniesim/config/`
2. Before simulation starts, run the script in the container to record the simulation process:

```bash
/geniesim/main$ ./scripts/start_auto_record.sh
```

1. The recording process auto exits after simulation is finished.
2. Press `Ctrl+C` to early stop and convert videos.

Images and videos are recorded at path `./output/recording_data/`

#### 3.1.4 Model Training

1. Compute norm states

```bash
openpi$ uv run python scripts/compute_norm_states.py --config
```

1. Set train config at `openpi/src/openpi/training/config.py`
2. Start training

```bash
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.97
uv sync
uv run python scripts/train.py select_color --exp-name select
```

---

### 3.2 Scene Generator

Scene Generator is an LLM-driven pipeline which takes in the natural language from the user, then performs RAG retrieval from `geniesim.assets` module and python coding for scene_graph, finally returns the "scene language".

#### 3.2.1 Features

1. **Promptable** - Chat within one window with continuous context.
2. **Scene Graph** - Structural scene graph and text annotation, LLM-friendly, benchmark-task-friendly.
3. **Controllable Edit** - High precision in spatial space, and code-level controllability.
4. **Generalization** - Semantic, spatial joint generalization through many aspects.
5. **Customize** - RAG enhanced LLM is training free and supports user's own LLM.
6. **GenieSim Ready** - The output scene_graph and scene layout assets are integrated seamlessly into `geniesim.benchmark`.

#### 3.2.2 Deploy

The generator module is located at `{GENIESIM_REPO}/source/geniesim/generator`

You **MUST** prepare your own `OPENAI_API_BASE_URL` and `OPENAI_API_KEY` before running the generator pipeline.

1. **Tested Models**


| Model Provider | Model Name                        |
| -------------- | --------------------------------- |
| ChatGPT        | gpt-5                             |
| Gemini         | gemini-pro-latest                 |
| DeepSeek       | deepseek-chat / deepseek-reasoner |
| QWEN           | qwen3-max                         |


Different models may have big differences in the output, and they're all input sensitive. Models with think mode or reasoner model may think longer and consume more tokens, this depends on how complex the scenario is. For text embedding model for asset searching, we recommend QWEN: `text-embedding-v4` with 2048 dimensions.

1. **Launch generator pipeline**
  a. Configure your embedding model in `./source/geniesim/generator/server/text_embedding_config.json` by providing `base_url`, `api_key` and `model`. If you use QWEN's embedding models (e.g. `text-embedding-v4`), set `dashscope_mode` to `true` to use dashscope api, otherwise set it to `false` to use openai client. Optionally, you can change `dimension` (default 2048) according to your model.
   b. (Optional) The provided cached database is built by QWEN: `text-embedding-v4` with 2048 dimensions. If you have a different model config, delete the `chroma_db` directory in the assets folder, then it will be automatically rebuilt according to your embedding model after launching the generator.
  > If the provided cached database can not be read, it will be rebuilt automatically at the cost of about 20 minutes and some tokens.
  >  c. Launch generator
  > When docker compose up for the first time, it might be slow due to network turbulence.
  >
  > If rebuilding is necessary, it might take 5~15 mins and cost some tokens.
  - Test open-webui by visiting [http://localhost:8080](http://localhost:8080), and the UI should show up properly.
  - Test MCP-server by visiting [http://localhost:8765/assets-agent/docs](http://localhost:8765/assets-agent/docs). Click the "Try it Out" button, text something in the keyword field, then "Execute" and see if the server returns properly.
2. **Setting up the geniesim webui configs**
  a. First **MAKE SURE** add your `OPENAI_API_BASE_URL` and `OPENAI_API_KEY` to `generator/config/openwebui.json`.
   b. Go to "Admin Panel (bottom left corner)" → "setting" → "database"
   Or visit [http://localhost:8080/admin/settings/db](http://localhost:8080/admin/settings/db)
      - Click "Import Config from JSON File"
      - Import `geniesim/generator/config/openwebui.json`
   c. Go to "Admin Panel (bottom left corner)" → "Functions"
   Or visit [http://localhost:8080/admin/functions](http://localhost:8080/admin/functions)
      - Click `Import` (top right)
      - Import `geniesim/generator/config/function-save_code_to_file.json` and refresh with F5
      - Enable the imported function and set it **global**
   d. Go to "workspace" → "models"
   Or visit [http://localhost:8080/workspace/models](http://localhost:8080/workspace/models)
      - Click "Import"
      - Import `geniesim/generator/config/geniesim-assets.json`
      - Import `geniesim/generator/config/geniesim-scenegen.json`
   e. Edit the imported models to use your own LLM
      - Click the `"..."` for more options on the right side of each model, select `"edit"`
      - In the `Base Model (From)` section, select the LLM you wish to use
      - Go to the page bottom, click `Save & Update`, done
3. **(Optional) Customizations**
  a. If you wish to customize API Connection in Open WebUI:
   Once Open WebUI is running:
  1. Go to the ⚙️ **Admin Settings**.
  2. Navigate to **Connections > OpenAI > Manage** (look for the wrench icon).
  3. Click ➕ **Add New Connection**.
    - Standard / Compatible → Azure OpenAI
    - Use this for OpenAI, DeepSeek, OpenRouter, LocalAI, FastChat, Helicone, LiteLLM, etc.
    - **Connection Type:** External
    - **URL:** `https://api.openai.com/v1` (or your provider's endpoint)
    - **API Key:** Your secret key (usually starts with `sk-...`)
     More troubleshooting at [https://docs.openwebui.com/](https://docs.openwebui.com/)
4. **LICENSE:** open-webui is open-sourced at GitHub under multiple licenses. For complete and updated licensing details, please see the LICENSE files.

#### 3.2.3 Start Scene Generation

1. Start a new chat, use "GenieSim Assistant" to first analyze your scene.
  Minimal example:
  > *I'd like to have an abundant fancy meal with drinks on a round table*
  > **Notice:**
  >
  > 1. The pipeline is only as smart as the Large Language Model connected to it.
  > 2. For the rich, varied, complicated scenarios, you'll need at least Gemini-3 or better.
  > 3. If the generated code fails to run or the scene isn't assembled as expected, more prompt engineering work should be done before code generation. See controllable scene generalization below.
2. Check the assets retrieval results, before starting scene generation.
  a. By continuing chatting with "GenieSim Assistant", you can edit the retrieval results.
   b. Once confirmed, switch to "GenieSim Generator".
   c. Say "*generate the scene*" to start generation.
3. Scene generation usually takes time depending on LLM and scene complexity.
  a. The context is maintained within the same chat window. You can make adjustments via dialogue, and be aware of the context limit of your model.
   b. Finally, save the results to `geniesim/generator/LLM_RESULT.py` by clicking the blue icon *(Save Codes to File)*, then a green bubble will show up indicating usda file is saved successfully.
4. Run generator to assemble the `scene_graph` and `scene usd assets`.
  a. The generator assembler uses the local built version of docker image `registry.agibot.com/genie-sim/open_source:latest`, make sure you've run through the installation section.
   b. We provide a real-time viewer, which can automatically generate and load the usda file if `geniesim/generator/LLM_RESULT.py` changes. Run following commands to start viewer:
  > Trigger once via commandline, run `isaac-sim@user:/geniesim/main$ ./scripts/run_generator.sh`
  >  c. During editing, click the *Save Codes to File* button to trigger the result preview, and the scene will be generated and loaded into IsaacSim.

#### 3.2.4 Mass-scale scene layout generalization

1. Here's the instruction to enable support for massive joint generalization:
  a. First explicitly define the generalization rules, and be specific about the details.
   b. Use structural `Markdown` format and carefully edit the prompt, lower the temperature of your model if *necessary*.
   c. If you wish to generate a scene with the assets list you provide, you can directly start the chat window with "GenieSim Generator" to skip the assets retrieval part.
   d. The underlying randomness uses random seeds to calculate random numbers. If your program fails to generate different scenes after each run, please check the system random interface.
2. **Prompt Demo 1** — *provide object list and rules of randomization*
  > I'd like to create a simple scene using "table_000" with the following specifications:
  >
  > - Randomly select **two distinct objects** from my list.
  > - Place them **near the center of the table**, slightly closer to the operator for easy access.
  > - Add a small amount of **random yaw rotation** (e.g., uniformly distributed between 0° and 360°) to each object to give the arrangement a more natural, less staged appearance.
  > - Ensure the two objects **do not intersect or collide** — maintain clear separation between them.
  >
  > The object list is: `["apple", "blocks", "cola", "facecleaner", "sprite"]`
3. **Prompt Demo 2** — *provide object list with natural language instructions*
  > I'd like to create a simple scene using "table_000" with the following setup:
  >
  > - Randomly select one drink can from my object list.
  > - Build a champagne tower using multiple copies of the chosen drink can.
  > - Place a single apple on top of the tower.
  > - Ensure all objects are neatly arranged on the tabletop.
  >
  > The available object list is: `["apple", "blocks", "cola", "facecleaner", "sprite"]`
4. **Prompt Demo 3** — *provide object list and layout parameters*
  > I'd like to create a simple scene using "table_000" with the following setup:
  >
  > - Randomly select 9 items from my object list.
  > - Arrange the selected objects in a 3×3 grid on the table, with each grid cell measuring 20 cm per side.
  > - Apply a small amount of random variation to each object's position: ±2 cm (normally distributed) along both the X and Y axes.
  > - Also randomize each object's orientation by applying a uniformly distributed yaw rotation between 0° and 360°.
  >
  > The object list consists of: `["benchmark_building_blocks_001", "benchmark_building_blocks_002", ..., "benchmark_building_blocks_026"]` (i.e., items numbered from 1 to 30).
5. If anything in the output goes unexpected, chat like normal to tell LLM to fix it.
6. For massive generalization, execute the generated results multiple times. Each generated scene will be available at output path with index and random seed.

---

### 3.3 Synthetic Data Collection

#### 3.3.1 Teleoperation

Support joystick control for robot waist, left/right end effector and base movement.


| No. | Function                       |
| --- | ------------------------------ |
| 1   | move base of the robot         |
| 2   | control left gripper           |
| 3   | backtracking action            |
| 4   | reset left arm and right arm   |
| 5   | enable left arm pose tracking  |
| 6   | start recording                |
| 7   | control the waist of the robot |
| 8   | control right gripper          |
| 9   | reset body and head            |
| 10  | enable right arm pose tracking |


##### 3.3.1.1 Pico Setup

1. Connect to the same **LAN** as the computer.
2. Start **AIDEA Vision App** in resource library.
3. Choose **Wireless Connection** and enter the **IP** of the computer.

> **Note:** The AIDEA Vision App has been uploaded; you can access it here:
> [https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset/tree/master/app.APK](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset/tree/master/app.APK)
> Installation for the Pico Enterprise Edition HMD can be performed via a computer. When the HMD is connected via USB, its internal storage becomes accessible as a standard drive on the computer. The APK file can then be installed by copying it directly to this drive.

##### 3.3.1.2 Launch Setup

1. **Data Collection**

Execute the following command under genie_sim root directory outside the docker container:

```bash
genie_sim$ ./scripts/autoteleop.sh
```

If you want to change tasks, please modify the `task_name` and `sub_task_name` fields in `./source/geniesim/config/teleop.yaml`.

1. **Data Post Process**

Execute the following command under genie_sim root directory outside the docker container:

```bash
genie_sim$ ./scripts/autoteleop_post_process.sh {TASK_NAME}
# for example
genie_sim$ ./scripts/autoteleop_post_process.sh open_door
```

> **Note:** To improve collection efficiency, data collection and data post-processing can be run in parallel.

#### 3.3.2 Automated Collection

##### 3.3.2.1 Installation

Make sure the `SIM_ASSETS` environment variable is set properly to the Genie Sim Assets path, and then enter `data_collection` directory:

```bash
export SIM_ASSETS=YOUR_PATH_TO_ASSETS  # e.g. ~/genie_sim/source/geniesim/assets
cd genie_sim/source/data_collection
genie_sim/source/data_collection$ sudo chmod -R +x ./scripts
```

**Docker installation**

Make sure the image in 2.3.1 is built, and run the following code to build the data collection image:

```bash
genie_sim/source/data_collection$ docker build -f ./dockerfile ...
```

> **Note:** For cuRobo installation, the dockerfile is configured for RTX 4090D by default. If you're using a different GPU model, you need to modify the `TORCH_CUDA_ARCH_LIST` environment variable in the dockerfile. 50 series GPU (SM_120) may NOT be able to install or run cuRobo, this needs a compatibility update by the cuRobo team.

##### 3.3.2.2 Data Collection

1. **One-Click Data Collection (Recommended)**

Use `run_data_collection.sh` to start data collection in one command.

```bash
genie_sim/source/data_collection$ ./scripts/run_data_collection.sh [OPTIONS]
```

**Options:**

- `--headless` — Run in headless mode for better performance (default: false)
- `--no-record` — Disable recording (default: record enabled)
- `--task TASK_PATH` — Task template path (e.g. `tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json`)
- `--standalone` — Run in standalone mode (only save logs, no terminal output) (default: false)
- `--container-name NAME` — Container name (default: `data_collection_open_source`)
- `--help, -h` — Show help message

**Examples:**

```bash
# Run with default task in GUI mode
genie_sim/source/data_collection$ ./scripts/run_data_collection.sh

# Run with default task in GUI mode with custom task
genie_sim/source/data_collection$ ./scripts/run_data_collection.sh --task ...

# Run in headless mode with custom task
genie_sim/source/data_collection$ ./scripts/run_data_collection.sh --headless --task ...

# Run in standalone headless mode (logs only, no terminal output)
genie_sim/source/data_collection$ ./scripts/run_data_collection.sh --standalone --headless

# Run without recording
genie_sim/source/data_collection$ ./scripts/run_data_collection.sh --no-record
```

**Logs:** Logs are saved to `logs/{TASK_NAME}/` directory:

- `run_data_collection_sh.log` — Script output
- `container.log` — Container logs
- `data_collector_server.log` — Data collector server logs (if available)
- `run_data_collection.log` — Run data collection application logs (if available)

**Outputs:** Outputs are saved to `recording_data/[{TASK_NAME}_{INDEX}]/` directory.

1. **Interactive Mode**

Use `start_gui.sh` to launch an interactive container for debugging or development.

```bash
genie_sim/source/data_collection$ ./scripts/start_gui.sh [ACTION] [CONTAINER_NAME]
```

**Actions:**

- `run` (default) — Create and run a new container
- `exec` — Enter an existing container
- `start` — Start a stopped container
- `restart` — Restart a container

**Parameters:**

- `ACTION` — One of: `exec`, `start`, `restart`, `run` (default: `run`)
- `CONTAINER_NAME` — Container name (default: `data_collection_open_source`)

**Examples:**

```bash
# Create and run a new container (default)
genie_sim/source/data_collection$ ./scripts/start_gui.sh run

# Enter an existing container
genie_sim/source/data_collection$ ./scripts/start_gui.sh exec

# Start a stopped container
genie_sim/source/data_collection$ ./scripts/start_gui.sh start

# Restart a container
genie_sim/source/data_collection$ ./scripts/start_gui.sh restart
```

**Running Services Inside Container:**

After entering the container using `exec`, you need to start two services in separate terminals:

**Terminal 1** — Start the container and run data collector server:

```bash
# Enter the container
genie_sim/source/data_collection$ ./scripts/start_gui.sh exec

# Inside container, start data collector server
isaac-sim@user:/geniesim/main/data_collection$ python scripts/...
```

**Terminal 2** — Enter the same container and run data collection application:

```bash
# Enter the same container (in a new terminal)
genie_sim/source/data_collection$ ./scripts/start_gui.sh exec

# Inside container, run main application
isaac-sim@user:/geniesim/main/data_collection$ python scripts/...
```

> **Note:** Both terminals need to exec into the same container. Make sure the container is running before executing these commands.

For more info about local installation and task configuration, see `source/data_collection/README.md`.

---

### 3.4 Scene Reconstruction

#### 3.4.1 Data Introduction

1. **Data Acquisition**

Collect data using Skyland Innovation's MetaCam handheld 3D laser scanner.

1. **Data Format**

```
├── camera
│   └── left
│       ├── 1763982134704786000.jpg   # fisheye image
│       ├── ...
│       ├── ...
│       ├── 1763982135704717000.jpg
├── colorized.las                      # color point cloud data
├── info
│   ├── calibration.json               # Calibration information
│   ├── device_info.json
│   └── rtk_info.json
└── transforms.json                    # camera pose info, camera pose is CG coordinate
```

#### 3.4.2 Environment Installation

`House.zip` contains the demo data required to run the scene reconstruction examples in this guide.

#### 3.4.3 Run Scene Reconstruction

Run the following command inside the container to implement scene reconstruction:

```bash
# Pull codebase
git clone https://github.com/AgibotTech/genie_sim.git
cd genie_sim/source/scene_reconstruction

# Build docker image
docker build . -t docker-image-name

# Unzip House.zip and launch container
docker run --rm --gpus all -it --shm-size=32g -v $(pwd):/House

# Download weights of Difix3d and place them in the genie_sim directory
# hf_model/
# ├── LICENSE.txt
# ├── NOTICE
# ├── README.md
# ├── model_index.json
# ├── scheduler
# ├── text_encoder
# ├── tokenizer
# ├── unet
# └── vae
```

```bash
cd /root/third_party/gsplat/examples
sh real2sim_environment_entrypoint.sh /mnt 1
# /mnt -> data_path   1 -> Enable DIfix3D (0 to disable)
```

To convert PLY file to USDZ and import into Isaac Sim, follow the "Converting PLY Files to USDZ" section in the NVIDIA 3DGRUT repository (nv-tlabs/3dgrut):

```bash
python -m threedgrut.export.scripts.ply_to_usd path/to/your/model
```

---

### 3.5 AgiBot World Challenge Reasoning to Action Tasks (ICRA)

#### 3.5.1 Run Baseline Model Inference

1. **Docker Image Preparation**

Baseline model checkpoints and inference scripts are integrated into one unified docker image, which can be obtained by following steps:

On the "My submission" page of the Test Server, click "Get Registry Token" to obtain the docker login credentials, then pull the image through the following command. (Note that the validity period of the voucher is 1 hour)

```bash
docker pull sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra...
```

1. **Start model inference**

Start docker container will automatically launch the inference service. Adjust `XLA_PYTHON_CLIENT_MEM_FRACTION` according to your machine GPU memory to run both inference and simulation on one machine.

```bash
docker run -it --network=host --gpus all -e XLA_PYTHON_CLIENT_MEM_FRACTION=...
```

The following info in terminal indicates the successful launch of the Inference service:

```
INFO:websockets.server:server listening on 0.0.0.0:8999
```

#### 3.5.2 Run ICRA tasks

1. Start simulation docker container

```bash
genie_sim$ ./scripts/start_gui.sh
```

1. After starting inference service, enter the docker container to launch all ICRA tasks.
  a. **Config VLM checker** — Set OpenAI configs for VLM auto scoring, see Section 3.1.2-3-c
   Two tasks, `scoop_popcorn` and `clean_the_desktop` are evaluated by VLM while other tasks are evaluated by rules. The absence of VLM configuration will not affect the simulation run, but the evaluation will be missing.
   b. Launch ICRA tasks
2. Scores will be automatically collected when all tasks are finished. It can also be triggered by command below:

```bash
# Use default directory: output/benchmark
/geniesim/main$ python3 scripts/stat_average.py

# Specify a custom directory
/geniesim/main$ python3 scripts/stat_average.py /path/to/your/dir
```

1. Evaluation process will be automatically terminated when SIGINT (`Ctrl+C`) signal is received.

#### 3.5.3 Integrate Your Own Policy

Prepare the docker image of your policy, including dependency, inference scripts and checkpoints etc.

1. Inference and simulation are separated into two containers, where websockets serve as the communication tool. Consequently, inference must be implemented in the form of a websocket service. An example can be found at `scripts/serve_policy.py` in the inference container.
2. **Observations that can be obtained from simulation environments**

**Observation Structure:**

```python
payload = {
    "state": ,          # robot joint state
    "eef": {
        "left": ,       # left_eef
        "right": ,      # right_eef
    },
    "images": {
        "top_head": ,   # rgb image from head
        "hand_left": ,  # rgb image from left wrist
        "hand_right": , # rgb image from right wrist
    },
    "depth": {
        "top_head": ,   # depth from head camera
        "hand_left": ,  # depth from left wrist camera
        "hand_right": , # depth from right wrist camera
    },
    "prompt": ,         # task_instruction
    "task_name":        # task_name
}
```

**Observation Info:**


| key               | type    | shape     | comments                                                                                 |
| ----------------- | ------- | --------- | ---------------------------------------------------------------------------------------- |
| state             | nparray | (32,)     | 0-7 left arm; 7-14 right arm; 14-15 left gripper; 15-16 right gripper; 16-21 waist joint |
| left_eef          | list    | 7         | x, y, z, qw, qx, qy, qz                                                                  |
| right_eef         | list    | 7         | x, y, z, qw, qx, qy, qz                                                                  |
| images-top_head   | nparray | (3, H, W) | rgb image from head camera                                                               |
| images-hand_left  | nparray | (3, H, W) | rgb image from left wrist camera                                                         |
| images-hand_right | nparray | (3, H, W) | rgb image from right wrist camera                                                        |
| depth-top_head    | nparray | (H, W)    | depth from head camera                                                                   |
| depth-hand_left   | nparray | (H, W)    | depth from left wrist camera                                                             |
| depth-hand_right  | nparray | (H, W)    | depth from right wrist camera                                                            |
| prompt            | string  | /         | task instructions                                                                        |
| task_name         | string  | /         | task name                                                                                |


1. **Commands that simulation environments receive**

**Command structure:**

```python
result = {
    # action chunk
    "actions": [
        [cmd],
        [cmd],
        [cmd],
        ...
    ]
}
```

**Command types:** Two types of commands are supported:

- **Abs joint** — For abs joint command, the command order is identical to joint state order specified above.
- **Abs eef** — For abs eef command, the command order is as follows:


| Index Range | Name                                  | Meaning       |
| ----------- | ------------------------------------- | ------------- |
| 0-6         | xyzrpy                                | Left arm eef  |
| 6-12        | xyzrpy                                | Right arm eef |
| 12-13       | idx41_gripper_l_outer_joint1          | Left gripper  |
| 13-14       | idx81_gripper_r_outer_joint1          | Right gripper |
| 14-19       | idx01_body_joint1 - idx05_body_joint5 | Waist         |


Tools are provided to convert ik/fk for robot at path `source/geniesim/utils/g2_ikfk_converter.py`, check the script for detailed usage.

#### 3.5.4 Submit Your Policy

- **Obtain Credentials:** You can obtain your Docker login credentials by clicking the "Get Registry Token" button on the My Submissions page of the Test Server. (Note that the validity period of the voucher is 1 hour)
- **Build the Image:** Integrate all necessary components — including dependencies, model checkpoints, and inference scripts — into a single Docker image.
- **Configure Port and Entrypoint:**
  1. **Port Requirement:** Your inference service must be configured to listen on port 8999.
  2. **Startup:** Ensure the Docker image is configured to start the inference container directly via an `ENTRYPOINT` or `CMD` instruction, without requiring any additional commands at runtime.
- **Tag and Push:** Tag your image according to the registry endpoint and namespace provided. For example, if your registry endpoint is `registry.test.com` and your namespace is `test1`, your image should be tagged and pushed as follows:
  1. Tag format: `registry.test.com/test1/{image_name}:tag`
  2. Command: `docker push registry.test.com/test1/{image_name}:tag`
- **Specify Model Type:** Choose your model type `abs_joint` or `abs_pose` on the submission page according to your model.

**Troubleshooting & Optimization:**

- **Credential Expiration:** If the image is too large to complete the upload within the credential's validity period, simply refresh your login credentials and re-initiate the push. Docker's layer-based architecture ensures that previously successful layers are preserved; only the remaining layers will be transmitted.
- **Image Optimization:** If the push continues to fail, please optimize the Docker image to reduce its footprint. We recommend:
  - Cleaning up build caches and unnecessary dependencies.
  - Implementing multi-stage builds to minimize the final image size.

