import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import requests
import argparse
import time


def downloadVideo(URL):
    FILE_TO_SAVE_AS = "./tmp/video.mp4"   # the name you want to save file as. May be cache this file! (on-disk)
    resp = requests.get(URL)           # making requests to server

    with open(FILE_TO_SAVE_AS, "wb") as f: # opening a file handler to create new file 
        f.write(resp.content) # writing content to file
    return FILE_TO_SAVE_AS

def generate(video_link, prompt, load_4bit = True, load_8bit = False, temperature = 0.1, device = torch.device('cpu')):
    disable_torch_init()

    print('Downloading Video {}'.format(video_link.split('Title')[-1]))
    video = downloadVideo(video_link)
    inp = prompt
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'

    print('Loading Generative Model')
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

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
        print(char, end = "", flush = True)
        time.sleep(0.02)
    print('\n')

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')   

    parser = argparse.ArgumentParser(description = 'Video Summary')
    
    parser.add_argument('-prompt',default = 'Generate the video summary', type = str,  help = "The user Prompt")
    parser.add_argument('-temperature',default = 0.1, type = float,  help = "The Temperature of the Prompt")
    parser.add_argument('-video_link',default = 'https://host-76-74-91-15.contentfabric.io/qlibs/ilib24CtWSJeVt9DiAzym8jB6THE9e7H/q/hq__4h2ZexvodR9BuGKTCFSKugEkb3o7ApzEukgACN5gDSuG35HqVukZfZDn9qhxiGKvNtticVaTon/rep/media_download/default/videovideo_1920x1080_h264@9500000?clip_start=6023.46&clip_end=6033.76&authorization=aessjc4BFDGcaPMtWMfU2WyaVkxv8FzE4jctQ2FxsRcpPWSPLJCNy2z2gaMVtE3HUKLYxMxB7hMw2pBaM5oexZFAH2DYksfUjRSSV5PwgxjpQNgRQGq2ZGCjze4P7Rt7fhm7k1atzUgKFHJqYWcwEjcVZhpa6HPdXEgE4BdzCiN5x4EwzXCAArS9sMx3S2ad2jec18ie4QLK4oATxYXJrjKajVorxwsLv9m5xvrSWQHW27yz3Br2PYMicjR4Q1wydZ6XubrTy91ihyKdAjYsypi9BMPspq6WPoqiFWZar4w9cBC1WoyEL2TYnsoCntnx9DQ3ugjKQzFuHvc3G3u5at7GwobmFE77LgvJUjfzsD4Qve1uUWRPa5hvqMeaQs7wEe5bCT5HKELCei&header-x_set_content_disposition=attachment%3Bfilename%3DTitle+-+Live+Recording+-+UEFA1+-+Wed+2023-10-25+%281920x1080%29+%2801-40-23+-+01-40-33%29.mp4', type = str,  help = "The downloadable Video Link!")
    args = parser.parse_args()  
    output = generate(args.video_link, args.prompt, load_4bit = True, load_8bit = False, temperature = 0.1, device = torch.device('cuda:1'))
    print('OUTPUT')
    print_chatbot(output)