"""Gradio UI untuk model Fashion-MNIST di Modal.

Deploy : modal deploy app.py
Dev    : modal serve app.py
"""
import modal

app = modal.App("fashion-mnist-cnn")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Wheel CPU: model ini kecil, tidak perlu GPU maupun wheel CUDA yang besar.
    .uv_pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("gradio==6.32.0", "fastapi[standard]==0.138.1", "numpy", "pillow")
    .add_local_python_source("model")
    .add_local_file("model.pth", "/root/model.pth")
    .add_local_dir("examples", "/root/examples")
)


# max_containers=1: Gradio menyimpan state sesi di memori, jadi semua request harus ke container yang sama.
@app.function(image=image, max_containers=1, scaledown_window=300)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    from pathlib import Path

    import gradio as gr
    from fastapi import FastAPI

    from model import load_model, predict

    model = load_model("/root/model.pth")

    def classify(img):
        return predict(model, img) if img is not None else None

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Gambar pakaian"),
        outputs=gr.Label(num_top_classes=3, label="Prediksi"),
        examples=sorted(str(p) for p in Path("/root/examples").glob("*.png")),
        title="Klasifikasi Pakaian Fashion-MNIST",
        description=(
            "CNN + Batch Normalization + Dropout (accuracy test 92,76%). Unggah gambar satu pakaian "
            "dengan latar polos, atau pilih contoh dari test set di bawah. Gambar diubah menjadi "
            "grayscale 28×28 sebelum diprediksi."
        ),
        flagging_mode="never",
    )
    return gr.mount_gradio_app(FastAPI(), demo, path="/")
