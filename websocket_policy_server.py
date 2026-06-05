import asyncio
import dataclasses
import logging
import time
import traceback

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
import wandb
import websockets.asyncio.server
import websockets.frames

from serving.utils import Packer, unpackb
from serving.action_decoder import ActionDecoder
from experiments.robot.openvla_utils import get_vla_latent_action
from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action

logger = logging.getLogger(__name__)

LATENT_ACTION_DETOKENIZE = [f"<ACT_{i}>" for i in range(32)]


@dataclasses.dataclass
class PolicyServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


class WebsocketPolicyServer:
    """WebSocket server that wraps the full UniVLA inference pipeline.

    The server is *stateful per episode*: it tracks the latent-action history
    and action-decoder temporal buffer.  Clients must send a ``{"command": "reset"}``
    message between episodes so the server can clear that state.

    Expected wire protocol (msgpack-serialized dicts):

    Client → Server:
        Observation   ``{"full_image": np.ndarray, "state": np.ndarray, "task_description": str}``
        Reset         ``{"command": "reset"}``
        Exit          ``{"command": "exit"}``

    Server → Client:
        Metadata      (sent once on connection)
        Action        ``{"action": np.ndarray}``
        Reset ack     ``{"status": "reset_done"}``
        Exit ack      ``{"status": "shutting_down"}``
    """

    def __init__(
        self,
        config: PolicyServerConfig,
        model,
        processor,
        action_decoder: ActionDecoder,
        *,
        pretrained_checkpoint: str,
        unnorm_key: str,
        center_crop: bool = True,
        model_family: str = "openvla",
        metadata: dict | None = None,
        process_request=None,
    ) -> None:
        self._host = config.host
        self._port = config.port

        # Inference components
        self._model = model
        self._processor = processor
        self._action_decoder = action_decoder
        self._pretrained_checkpoint = pretrained_checkpoint
        self._unnorm_key = unnorm_key
        self._center_crop = center_crop
        self._model_family = model_family

        # Cached action normalization stats (constant across episodes)
        action_norm_stats = model.get_action_stats(unnorm_key)
        self._action_mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        self._action_high = np.array(action_norm_stats["q99"])
        self._action_low = np.array(action_norm_stats["q01"])

        # Per-episode state
        self._prev_hist_action: list[str] = [""]

        # Server plumbing
        self._metadata = metadata or {}
        self._process_request = process_request
        self._stop_event = asyncio.Event()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        # Monitoring
        self._wandb_enable = bool(self._metadata.get("wandb_enable", False))
        self._trace_enable = bool(self._metadata.get("trace_enable", False))
        self._drop_first_n_frames = int(self._metadata.get("drop_first_n_frames", 1))
        self._infer_cnt = 0

        if self._trace_enable:
            self._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
                on_trace_ready=lambda prof: prof.export_chrome_trace(
                    f"tmp/trace_schedule_{prof.step_num}.json"
                ),
            )

    # ------------------------------------------------------------------
    # Episode state management
    # ------------------------------------------------------------------

    def _reset_episode(self) -> None:
        self._action_decoder.reset()
        self._prev_hist_action = [""]

    # ------------------------------------------------------------------
    # Client format conversion
    # ------------------------------------------------------------------

    def _convert_client_format(self, raw_msg: dict) -> dict:
        """Convert client input format to server expected format.
        
        Supports both LeRobot-style and native LIBERO-style data:
        
        LeRobot-style:
            observation.images.*: HWC uint8 ndarray
            observation.state: (D,) float32 ndarray  
            task: ndarray (needs conversion to string)
        
        Native LIBERO-style:
            observation.image.full: HWC uint8 ndarray
            observation.state: (D,) float32 ndarray
            task_description: str
        
        Server expected format:
            observation.image.full: HWC uint8 ndarray
            observation.state: (D,) float32 ndarray
            task_description: str
        """
        converted = {}
        
        # 1. 处理 observation（支持嵌套和扁平两种格式）
        if "observation" in raw_msg:
            obs = raw_msg["observation"]
            
            # 方式A: 嵌套格式 (LeRobot-style) -> observation.images.*
            if isinstance(obs, dict) and "images" in obs:
                images = obs["images"]
                if isinstance(images, dict):
                    # 找到第一个图像
                    for img_key in sorted(images.keys()):
                        img_val = images[img_key]
                        if isinstance(img_val, np.ndarray) and img_val.ndim == 3:
                            converted["observation.image.full"] = img_val
                            break
            
            # 方式B: 扁平格式 (native LIBERO-style) -> observation.image.full
            elif isinstance(obs, dict) and "observation.image.full" in obs:
                img = obs["observation.image.full"]
                if isinstance(img, np.ndarray):
                    converted["observation.image.full"] = np.squeeze(img)
            
            # 处理状态（去掉batch维度）
            if "observation.state" in obs:
                state = obs["observation.state"]
            elif "state" in obs:
                state = obs["state"]
            else:
                state = None
            
            if state is not None:
                if isinstance(state, np.ndarray):
                    state = np.squeeze(state)
                converted["observation.state"] = state
        
        # 2. 处理任务描述（支持 ndarray 和 str 两种格式）
        if "task_description" in raw_msg:
            # 原生 LIBERO 格式（已经是字符串）
            converted["task_description"] = str(raw_msg["task_description"])
        elif "task" in raw_msg:
            # LeRobot 格式（需要转换）
            task = raw_msg["task"]
            if isinstance(task, np.ndarray):
                if task.size > 0:
                    if task.dtype.kind in ('U', 'S', 'O'):
                        task_description = str(task.flat[0])
                    else:
                        task_description = str(task)
                else:
                    task_description = ""
            else:
                task_description = str(task)
            converted["task_description"] = task_description
        
        # 3. 处理其他顶层 ndarray（去掉batch维度，跳过嵌套dict）
        for key, value in raw_msg.items():
            if key not in ["observation", "task", "task_description"]:
                if isinstance(value, np.ndarray):
                    value = np.squeeze(value)
                    converted[key] = value
        
        return converted

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _predict_action(self, obs_dict: dict) -> np.ndarray:
        """Run the full inference pipeline and return a ready-to-execute action."""
        task_description = obs_dict["task_description"]
        observation = {
            "full_image": obs_dict["observation.image.full"],
            "state": obs_dict["observation.state"],
        }

        t0 = time.perf_counter()

        latent_action, visual_embed, generated_ids = get_vla_latent_action(
            self._model,
            self._processor,
            self._pretrained_checkpoint,
            observation,
            task_description,
            self._unnorm_key,
            center_crop=self._center_crop,
            hist_action=self._prev_hist_action[-1],
        )

        # Update latent-action history
        hist_action = ""
        for token_id in generated_ids[0]:
            hist_action += LATENT_ACTION_DETOKENIZE[token_id.item() - 32001]
        self._prev_hist_action.append(hist_action)

        # Decode latent action → 7-DoF action with temporal ensemble
        action = self._action_decoder(
            latent_action, visual_embed,
            self._action_mask, self._action_low, self._action_high,
        )

        # Post-process gripper
        action = normalize_gripper_action(action, binarize=True)
        if self._model_family == "openvla":
            action = invert_gripper_action(action)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Monitoring
        self._infer_cnt += 1
        if self._wandb_enable and self._infer_cnt > self._drop_first_n_frames:
            wandb.log({"infer_cost_ms": elapsed_ms})
        if self._infer_cnt % 50 == 0:
            logger.info("inference step %d  (%.1f ms)", self._infer_cnt, elapsed_ms)

        return action

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()

        # Send server metadata on connect
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = unpackb(await websocket.recv(), raw=False)

                if isinstance(msg, dict) and msg.get("command") == "reset":
                    self._reset_episode()
                    await websocket.send(packer.pack({"status": "reset_done"}))

                elif isinstance(msg, dict) and msg.get("command") == "exit":
                    logger.info("Received exit command from %s", websocket.remote_address)
                    await websocket.send(packer.pack({"status": "shutting_down"}))
                    self._stop_event.set()

                elif isinstance(msg, dict):
                    # Convert client format to server expected format
                    converted_msg = self._convert_client_format(msg)
                    action = self._predict_action(converted_msg)
                    await websocket.send(packer.pack({"action": action}))

                else:
                    logger.warning("Unexpected message type: %s", type(msg))

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        serve_kwargs: dict = dict(compression=None, max_size=None)
        if self._process_request is not None:
            serve_kwargs["process_request"] = self._process_request

        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            **serve_kwargs,
        ) as server:
            logger.info("WebSocket policy server listening on ws://%s:%s", self._host, self._port)
            await self._stop_event.wait()
            logger.info("Shutdown event received. Server exiting.")
