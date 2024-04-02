import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import requests
import argparse
import time
from typing import List, Tuple
import cv2
import json
import warnings
import os
import tqdm

warnings.filterwarnings('ignore')

device = torch.device('cuda:0')


def generate(model, video_processor, tokenizer, conv, prompt, video_path,  temperature=0.1,):
    # get the video representation tensor
    video_tensor = video_processor(video_path, return_tensors='pt')[
        'pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16)
                  for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    # deal with prompt
    prompt = ' '.join([DEFAULT_IMAGE_TOKEN] *
                      model.get_video_tower().config.num_frames) + '\n' + prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # TODO: need to check this function because it has nothing to do with the image_token
    # tokenizer prompts
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # define stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(
        keywords, tokenizer, input_ids)

    # inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    return outputs


def print_chatbot(output):
    for char in output:
        print(char, end="", flush=True)
        time.sleep(0.02)
    print('\n')


if __name__ == '__main__':

    # define args parser
    parser = argparse.ArgumentParser(description='Video Summary')

    parser.add_argument(
        '-prompt', default='Generate the video summary', type=str,  help="The user Prompt")
    parser.add_argument('--backend', default='decord',
                        type=str,  help="The video Backend")
    parser.add_argument('--temperature', default=0.1,
                        type=float,  help="The Temperature of the Prompt")
    parser.add_argument('--load4bit', action="store_true")
    parser.add_argument("--object_id", type=str, default="iq__3BCpFm5NquAuzCMqekkZX8rR3hF",
                        help="Content object Id")
    args = parser.parse_args()

    if args.load4bit:
        load_4bit, load_8bit = True, False
    else:
        load_4bit, load_8bit = False, True

    # load the model
    disable_torch_init()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    model_name = get_model_name_from_path(model_path)
    cache_dir = 'cache_dir'
    print('Loading Generative Model')
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)

    # Change to FFMPEG - Uniform --> 'decord', 'eluvio', maybe not necessary
    video_processor = processor['video']
    video_processor.set_transform(args.backend)
    video_processor.config.vision_config.video_decode_backend = args.backend

    # define conv
    ipt = args.prompt
    conv_mode = "llava_v1"

    # clear the outpuot while loading
    os.system("clear")

    # get shots prepared to tag
    shot_src = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "tmp", args.object_id)
    shots = sorted(os.listdir(shot_src))
    res = {}

    # loop the shots
    for shot in tqdm.tqdm(shots):
        _shot_int = int(shot.split("_")[1].split(".")[0])
        conv = conv_templates[conv_mode].copy()
        try:
            output = generate(
                model, video_processor, tokenizer,
                conv, args.prompt,
                os.path.join(shot_src, shot),
                temperature=args.temperature, )
            res[_shot_int] = output
        except:
            res[_shot_int] = ""

    # write result
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "captions", f"{args.object_id}.json"), "w") as f:
        json.dump(res, f)
