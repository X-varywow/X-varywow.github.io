

## （1）demo1

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()  
```


## （2）demo2

```python
import gradio as gr


def tts_fn(text, speaker):
    inputs = processor(text=text, return_tensors="pt")
    speaker_embeddings = torch.tensor(embeddings_dataset[speaker]["xvector"]).unsqueeze(0)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write("speech.wav", speech.numpy(), samplerate=16000)
    
    # data, samplerate = sf.read("speech.wav")
    out_path = "speech.wav"
    # print(type(data))
    # data is numpy.ndarray
    return "success", out_path

speakers = [5793, 1605, 3592, 3369, 7355]
lang = ['日本語', '简体中文', 'English', 'Mix']

app = gr.Blocks()
with app:
    with gr.Tab("Text-to-Speech"):
        with gr.Row():
            with gr.Column():
                textbox = gr.TextArea(label="Text",
                                      placeholder="Type your sentence here",
                                      elem_id=f"tts-input")
                speaker = gr.Radio([5793, 1605, 3592, 3369, 7355], label="speakers", info="role")
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                btn = gr.Button("Generate!")
                btn.click(tts_fn,
                          inputs=[textbox, speaker],
                          outputs=[text_output, audio_output])

app.launch() 
```

## （3）使用 css

```python
# 在 gr 组件中 elem_id 即可完成指定
css = """
    #elem-id {
        color: ,,,
    }

"""
with gr.Blocks(css=css) as app:
    pass
app.launch()
```

https://gradio.app/custom-CSS-and-JS/


## （4）超时问题

大多数浏览器在短时间，没收到对 POST 请求的响应时，会报错。

ERROR document ....

这时，可以使用 websocket，添加 .queue 即可

```python
app = gr.Interface(lambda x:x, "image", "image")
app.queue()  # <-- Sets up a queue with default parameters
app.launch()
```

但是报错：WebSocket connection to 'wss://aigc-service.aviagames.net/queue/join' failed: 

不用 websocket 就超时，，POST 504 网关超时


----------------

仿照 sd webui 写个高级自定义的


参考资料：
- [官方文档](https://gradio.app/docs/)
- https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance