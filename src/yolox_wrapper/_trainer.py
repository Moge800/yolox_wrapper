# ty: ignore[unresolved-import]
"""YOLOX トレーニング内部実装モジュール

このモジュールは yolox_wrapper.wrapper.YOLOX.train() から呼ばれる内部実装です。
直接使う場合は YOLOXModelCreator クラスを使用してください。

順次エポック学習の仕組み:
    epoch_schedule=[100, 200, 300] の場合:
        Stage 0: max_epoch=100, resume=False → stage_0_epoch100_ckpt.pth
        Stage 1: max_epoch=200, resume=True  → epoch 101 から再開
        Stage 2: max_epoch=300, resume=True  → epoch 201 から再開
"""

import argparse
import io
import shutil
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .dataset import _MODEL_CONFIGS

# ---------------------------------------------------------------------------
# ログリダイレクター
# ---------------------------------------------------------------------------


class _LogRedirector:
    """sys.stdout を一時的にコールバックへリダイレクトするコンテキストマネージャ"""

    def __init__(self, callback: Callable[[str], None] | None) -> None:
        self._callback = callback
        self._original_stdout = None
        self._original_stderr = None

    def __enter__(self) -> "_LogRedirector":
        if self._callback is None:
            return self
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _CallbackStream(self._callback)
        sys.stderr = _CallbackStream(self._callback)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr


class _CallbackStream(io.TextIOBase):
    """write() をコールバックに転送するストリーム"""

    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self._callback = callback

    def write(self, text: str) -> int:
        if text and text.strip():
            self._callback(text.rstrip("\n"))
        return len(text)

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Exp 生成
# ---------------------------------------------------------------------------


def _build_exp(
    num_classes: int,
    data_dir: str,
    input_size: tuple[int, int],
    depth: float,
    width: float,
    max_epoch: int,
    batch_size: int,
    num_workers: int,
    basic_lr_per_img: float,
    train_ann: str,
    val_ann: str,
):
    """動的に YOLOX Exp サブクラスを生成する"""
    try:
        from yolox.exp import Exp as BaseExp
    except ImportError as e:
        raise ImportError(
            "yolox パッケージが必要です。\n"
            "  uv pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git\n"
            "  uv pip install pycocotools-windows  # Windows の場合"
        ) from e

    class YOLOXExp(BaseExp):
        def __init__(self):
            super().__init__()
            self.num_classes = num_classes
            self.depth = depth
            self.width = width
            self.input_size = input_size
            self.test_size = input_size
            self.max_epoch = max_epoch
            self.warmup_epochs = max(1, min(5, max_epoch // 10))
            self.no_aug_epochs = max(1, min(15, max_epoch // 5))
            self.basic_lr_per_img = basic_lr_per_img
            self.data_dir = data_dir
            self.train_ann = train_ann
            self.val_ann = val_ann
            self.train_name = "train"
            self.val_name = "val"
            self.data_num_workers = num_workers
            self.eval_interval = 5
            self.test_conf = 0.01
            self.nmsthre = 0.65

        def get_dataset(self, cache=False, cache_type="ram"):
            from yolox.data import COCODataset, TrainTransform

            return COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
                cache=cache,
                cache_type=cache_type,
                name=self.train_name,
            )

        def get_eval_dataset(self, **kwargs):
            from yolox.data import COCODataset, ValTransform

            return COCODataset(
                data_dir=self.data_dir,
                json_file=self.val_ann,
                img_size=self.test_size,
                preproc=ValTransform(legacy=False),
                name=self.val_name,
            )

        def get_evaluator(
            self, batch_size, is_distributed, testdev=False, legacy=False
        ):
            from yolox.evaluators import COCOEvaluator

            return COCOEvaluator(
                dataloader=self.get_eval_loader(batch_size, is_distributed),
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=self.num_classes,
                testdev=testdev,
                per_class_AP=True,
                per_class_AR=True,
            )

    return YOLOXExp


# ---------------------------------------------------------------------------
# トレーナー
# ---------------------------------------------------------------------------


class _YOLOXTrainer:
    """YOLOX トレーニングを管理するクラス

    通常は YOLOX.train() 経由で使用します。
    """

    def __init__(
        self,
        model_size: str,
        num_classes: int,
        dataset_dir: str,
        output_dir: str,
        input_size: tuple[int, int] = (640, 640),
        batch_size: int = 16,
        device: str = "cpu",
        num_workers: int = 4,
        basic_lr_per_img: float = 0.01 / 64,
    ) -> None:
        if model_size not in _MODEL_CONFIGS:
            raise ValueError(
                f"未対応の model_size: '{model_size}'\n"
                f"使用可能: {list(_MODEL_CONFIGS.keys())}"
            )

        self.model_size = model_size
        self.num_classes = num_classes
        self.dataset_dir = dataset_dir
        self.output_dir = Path(output_dir)
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.basic_lr_per_img = basic_lr_per_img

    def train_sequential(
        self,
        epoch_schedule: list[int],
        on_log: Callable[[str], None] | None = None,
        on_stage_done: Callable[[int, int, str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> str:
        """順次エポック学習を実行する

        Args:
            epoch_schedule: エポック数のリスト (例: [100, 200, 300])
            on_log:         ログ行を受け取るコールバック
            on_stage_done:  ステージ完了時コールバック (stage_idx, epoch, ckpt_path)
            stop_event:     セットされると次のステージ開始前に中断

        Returns:
            最終チェックポイントのパス
        """
        try:
            import torch
            from yolox.core import Trainer
        except ImportError as e:
            raise ImportError(
                "yolox パッケージが必要です。\n"
                "  uv pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git"
            ) from e

        cfg = _MODEL_CONFIGS[self.model_size]
        num_gpu, device_id = self._parse_device(self.device)

        if device_id >= 0:
            torch.cuda.set_device(device_id)

        train_output_dir = str(self.output_dir / "train_output")
        exp_name = f"yolox_{self.model_size}"
        last_ckpt_path = ""

        for i, target_epoch in enumerate(epoch_schedule):
            if stop_event and stop_event.is_set():
                msg = f"[Trainer] 中断 (Stage {i}/{len(epoch_schedule)})"
                if on_log:
                    on_log(msg)
                else:
                    print(msg)
                break

            if on_log:
                on_log(
                    f"[Trainer] ===== Stage {i + 1}/{len(epoch_schedule)} | target epoch: {target_epoch} ====="
                )

            exp_cls = _build_exp(
                num_classes=self.num_classes,
                data_dir=self.dataset_dir,
                input_size=self.input_size,
                depth=cfg["depth"],
                width=cfg["width"],
                max_epoch=target_epoch,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                basic_lr_per_img=self.basic_lr_per_img,
                train_ann="instances_train.json",
                val_ann="instances_val.json",
            )
            exp = exp_cls()
            exp.output_dir = train_output_dir
            exp.exp_name = exp_name

            resume = i > 0
            args = argparse.Namespace(
                experiment_name=exp_name,
                resume=resume,
                ckpt=None,
                start_epoch=None,
                num_machines=1,
                machine_rank=0,
                fp16=False,
                cache=None,
                occupy=False,
                logger="tensorboard",
                save_history_ckpt=True,
            )

            with _LogRedirector(on_log):
                if num_gpu > 1:
                    from yolox.core import launch

                    launch(
                        lambda local_rank, _exp=exp, _args=args: Trainer(
                            _exp, _args
                        ).train(),
                        num_gpus=num_gpu,
                        num_machines=1,
                        machine_rank=0,
                        backend="nccl",
                        dist_url="auto",
                    )
                else:
                    trainer = Trainer(exp, args)
                    trainer.train()

            # スナップショット保存
            last_ckpt = Path(train_output_dir) / exp_name / "last_epoch_ckpt.pth"
            snap = (
                Path(train_output_dir)
                / exp_name
                / f"stage_{i}_epoch{target_epoch}_ckpt.pth"
            )
            if last_ckpt.exists():
                shutil.copy2(last_ckpt, snap)
                last_ckpt_path = str(snap)

            if on_stage_done:
                on_stage_done(i, target_epoch, last_ckpt_path)

        return last_ckpt_path

    def package_model(
        self,
        class_names: dict[int, str],
        checkpoint_path: str | None = None,
        output_model_path: str | None = None,
    ) -> str:
        """チェックポイントを yolox_wrapper.YOLOX が読み込める .pt に変換する

        Args:
            class_names:       クラス名辞書 {0: "cat", 1: "dog"}
            checkpoint_path:   変換元チェックポイント (省略時は自動検索)
            output_model_path: 出力先 (省略時は output_dir/yolox_{model_size}.pt)

        Returns:
            保存した .pt ファイルのパス
        """
        import torch

        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()

        print(f"[Trainer] チェックポイント読み込み: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        cfg = _MODEL_CONFIGS[self.model_size]
        model = self._build_model(cfg["depth"], cfg["width"], len(class_names))

        state_dict = ckpt.get("model", ckpt)
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
        else:
            model = state_dict

        model.eval()

        if output_model_path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_model_path = str(self.output_dir / f"yolox_{self.model_size}.pt")

        torch.save(
            {
                "model": model,
                "names": class_names,
                "nc": len(class_names),
                "input_size": list(self.input_size),
            },
            output_model_path,
        )
        print(f"[Trainer] モデルを保存しました: {output_model_path}")
        return output_model_path

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def _parse_device(self, device: str) -> tuple[int, int]:
        """デバイス文字列から (GPU 数, デバイス ID) を返す"""
        if device == "cpu":
            return 0, -1
        if device == "cuda":
            return 1, 0
        if device.startswith("cuda:"):
            return 1, int(device.split(":")[1])
        return 0, -1

    def _find_best_checkpoint(self) -> str:
        train_dir = self.output_dir / "train_output" / f"yolox_{self.model_size}"
        for name in ["best_ckpt.pth", "last_epoch_ckpt.pth"]:
            p = train_dir / name
            if p.exists():
                return str(p)
        raise FileNotFoundError(
            f"チェックポイントが見つかりません: {train_dir}\n"
            "先に train_sequential() を実行してください。"
        )

    def _build_model(self, depth: float, width: float, num_classes: int):
        try:
            from yolox.models import YOLOPAFPN, YOLOXHead
            from yolox.models import YOLOX as _YOLOXModel
        except ImportError as e:
            raise ImportError("yolox パッケージが必要です。") from e

        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
        head = YOLOXHead(num_classes, width, in_channels=in_channels)
        return _YOLOXModel(backbone, head)
