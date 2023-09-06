import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import gc
import json
import time
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

from modules import commons
import utils
from data_utils_vits import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models_vits import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import gradio as gr

## SVC
from inference.infer_tool import Svc
import soundfile


# format: show_name : [model, config, speaker]
# 舍弃 
# - 'caster(white female)': ["spk/caster/G_60000.pth", "spk/caster/config.json", "caster"], 日本卡通女声


# 'ryan(white man)': ["spk/MaudPie/G_55000.pth", "spk/MaudPie/config.json", "MaudPie"],
#    'Biden': ["spk/Biden/G_20000.pth", "spk/Biden/config.json", "Biden"],
#    'Obama': ["spk/Obama/G_50000.pth", "spk/Obama/config.json", "Obama"],
#    'Trump': ["spk/Trump/G_18500.pth", "spk/Trump/config.json", "Trump"],
#     'Nate(black man)': ["spk/multi_black/G_8800.pth", "spk/multi_black/config.json", "Nate"],
#   'MaudPie(white woman)': ["spk/MaudPie/G_55000.pth", "spk/MaudPie/config.json", "MaudPie"],

role = {
    'base(white woman)': [],
    
    'Scharphie(white woman)': ["spk/Scharphie/G_13600.pth", "spk/Scharphie/config.json", "Scharphie"],
    
    'Ryan(white man)': ["spk/ryan/G_13600.pth", "spk/ryan/config.json", "ryan"],
    'postal_dude(white man)': ["spk/postal_dude/G_2015.pth", "spk/postal_dude/config.json", "postal_dude"],
    
    'Wilson(black man)': ["spk/multi_black/G_8800.pth", "spk/multi_black/config.json", "Wilson"],
    'Benimana(black man)': ["spk/multi_black/G_8800.pth", "spk/multi_black/config.json", "Benimana"]
}


# def punc(s):
#     res = []
#     for sentence in s.split("."):
#         if sentence != "\n":
#             sentence = ",".join([i.strip() for i in sentence.split(",")])+"."
#             res.append(sentence)
#     return res


def get_text(text):
    text_norm = text_to_sequence(text, ["english_cleaners2"])
    text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

SHOW_TEXT_s = "She just won one hundred dollers, she can cash out within 30 seconds, whenever she likes with no restrictions at all."

SHOW_TEXT = """OMG, she won over one thousand dollers in under one minute, and I just saw her win not one but two new iPhones.
She just won one hundred dollers, she can cash out within 30 seconds, whenever she likes with no restrictions at all.
That's an extra two hundred dollers, that's one hundred percent safe and secure for her to cash out, because this game is an official PayPal partner.
I can't believe how much this game pays, it's like welfare for free.
I'm normally just a Bingo caller, but I'm going to start playing.
Download and play Bingo Clash now."""



debug = True

def tts_fn(text, speed=1, noise_scale=0.667, noise_scale_w=0.8, interval = int(22050*0.5)):
    
    hps = utils.get_hparams_from_file("./configs_vits/ljs_base.json")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    net_g.eval()
    utils.load_checkpoint("./pretrain/pretrained_ljs.pth", net_g, None)
    
    # res = np.array([])
    # text_l = punc(text)
    # for text in text_l:
    #     if debug:
    #         print(text)
    
    stn_tst = get_text(text)
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])

    with torch.no_grad():
        audio = net_g.infer(x_tst, x_tst_lengths,noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=1.0/speed)[0][0,0].data.float().numpy()
        # res = np.append(res, audio)
        # res = np.append(res, np.array([0]*interval))
        
    timestamp = str(int(time.time()))
    filename = "tmp" + timestamp + ".wav"
    wav_path = f"spk/{filename}"
    soundfile.write(wav_path, audio, 22050, format="wav")
    
    # del net_g
    # gc.collect()
    # net_g = None
    # torch.cuda.empty_cache()
    return audio, wav_path

    # def slice_inference(self,
    #                     raw_audio_path,
    #                     spk,
    #                     tran,
    #                     slice_db,
    #                     cluster_infer_ratio,
    #                     auto_predict_f0,
    #                     noice_scale,
    #                     pad_seconds=0.5,
    #                     clip_seconds=0,
    #                     lg_num=0,
    #                     lgr_num =0.75,
    #                     F0_mean_pooling = False,
    #                     enhancer_adaptive_key = 0,
    #                     cr_threshold = 0.05
    #                     ):

def vc_fn(wav_path, speaker, fmp = True, auto_f0 = True):
    
    model_path, config_path, speaker = role[speaker]
    
    # cluster_model_path = cluster_model_path.name if cluster_model_path != None else "",
    # nsf_hifigan_enhance = enhance
    # device=device if device!="Auto" else None
    print(model_path, config_path)
    
    # svc_model = Svc(args.model_path, args.config_path, args.device, args.cluster_model_path,enhance)
    model = Svc(model_path, config_path, None, "logs/44k/kmeans_10000.pt", False) 
    # device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
    

    res = model.slice_inference(wav_path, 
                                   speaker, 
                                   tran = 0,  # vc_transform 变调，不使用
                                   slice_db = -40,    # slice_db 切片阈值
                                   cluster_infer_ratio = 0, # cluster_ratio 聚类模型比例，为 0 不使用
                                   auto_predict_f0 = auto_f0,    # 自动变调预测
                                   noice_scale = 0.4,
                                   # pad_seconds,cl_num,lg_num,lgr_num, enhancer_adaptive_key, cr_threshold
                                   F0_mean_pooling = fmp  # F0 滤波，改善哑音
                                   )
    # soundfile.write(output_file, _audio, model.target_sample, format="wav")
    # del model
    return res

    
def all_in(text, speaker, speed=1, noise_scale=0.667, noise_scale_w=0.8):
    wav, wav_path = tts_fn(text, speed, noise_scale, noise_scale_w)
    if speaker.startswith("base"):
        return "Success", (22050, wav)
    
    res = vc_fn(wav_path, speaker)
    return "Success", (44100, res)
    

if __name__ == "__main__":

    app = gr.Blocks(
        title = "AIGC",
        theme=gr.themes.Base(
        primary_hue = gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New'],
        ),
    )
    
    with app:
        
        with gr.Tabs():
            with gr.TabItem("AIGC_AUDIO v1.2"):
                
                with gr.Row(variant="panel"):
                    with gr.Tabs():
                        
                        with gr.TabItem("文本转语音"):
                            with gr.Row(variant="panel"):
                                with gr.Column():
                                    text  = gr.Textbox(label = "Text",
                                                    placeholder = "Type your sentence here",
                                                    value = SHOW_TEXT_s,
                                                    elem_id = "tts-input")

                                    speaker = speaker = gr.Radio(label="Speaker", choices=list(role.keys()), value="base(white woman)")

                                    speed = gr.Slider(minimum=0.5, maximum=2, value=1.2, step=0.05, label="语速")
                                    noise_scale = gr.Slider(minimum=0.1, maximum=1, value=0.65, step=0.01, label="情感变化")
                                    noise_scale_w = gr.Slider(minimum=0.1, maximum=1, value=0.8, step=0.1, label="音素发音长度")
                                    # fmp
                                    # auto_f0

                                with gr.Column():

                                    text_output = gr.Textbox(label="Message")
                                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                                    btn = gr.Button("Generate!" , variant="primary")
                                    btn.click(all_in,
                                            inputs=[text, speaker, speed, noise_scale, noise_scale_w],  # 
                                            outputs=[text_output, audio_output])
                        with gr.TabItem("语音转语音"):
                            gr.Markdown(value="""
                                        <font size=3>语音转语音正在实现中</font>
                                        """)
                            


                with gr.Row(variant="panel"):
                    with gr.Column():
                        gr.Markdown(value="""
                            <font size=2> 说明</font>
                            """)
                        gr.Markdown(value="""
                                    <font size=3>
                                    1. Use "." in the end of all sentences. It means 0.5s pause.

                                    2. Only support English now.

                                    3. You can download the audio by click the right button of "Output Audio"。
                                    
                                    4. Generate audio may take a long time(60s a sentence.)
                                    </font>
                                    """)

                with gr.Row(variant="panel"):
                    with gr.Column():
                        gr.Markdown(value="""
                            <font size=2> 更新记录</font>
                            """)
                        gr.Markdown(value="""
                                    ## V1.2
                                    - 加入音色 ryan(原本的 base(white man)),Scharphie(white woman)
                                    - 优化底层的 tts
                        
                                    ## V1.1
                                    - 加入语音克隆模型
                                    - 加入音色 Wilson Benimana 等
                                    
                                    ## 计划中
                                    - 提供一个稳定的链接
                                    - 提高语音流畅度
                                    - 老人、小孩音色
                                    """)
                        


            with gr.TabItem("AIGC_VIDEO v1.0"):
                with gr.Row(variant="panel"):
                    with gr.Tabs():
                        
                        with gr.TabItem("DCT-NET"):
                            with gr.Row(variant="panel"):
                                with gr.Column():
                                    gr.Markdown(value="""
                                                <font size=3>测试</font>
                                                """)

                                with gr.Column():
                                    gr.Markdown(value="""
                                                <font size=3>测试</font>
                                                """)
                        with gr.TabItem("VToonify"):
                            gr.Markdown(value="""
                                        <font size=3>测试</font>
                                        """)
                            

                with gr.Row(variant="panel"):
                    with gr.Column():
                        gr.Markdown(value="""
                            <font size=2> 更新记录</font>
                            """)
                        gr.Markdown(value="""
                                    ## V1.0
                                    - 部署服务
                                    
                                    ## 计划中
                                    - opencv + live2d 优化
                                    """)



                    
    # Enabling the queue is required for inference times > 60 seconds:
    app.queue().launch(share=False)