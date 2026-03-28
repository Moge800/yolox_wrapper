# -*- coding: utf-8 -*-
"""YOLOX Wrapper エントリポイント

GUI 起動:
    uv run main.py

ヘッドレス使用例:
    from yolox_wrapper import YOLOX

    model = YOLOX("l")
    model.train(data="data.yaml", epochs=[100, 200, 300], device="cuda:0", batch=16)

    model = YOLOX("yolox_l.pt")
    results = model.predict("image.jpg", conf=0.3)
    model.export(format="onnx")
"""


def main() -> None:
    from yolox_wrapper.gui.app import App
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
