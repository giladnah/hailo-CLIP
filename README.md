# AD GENIE - Personalized Advertisement 
![](ad_genie.gif)
### [Watch our beautifully illustrated demo!](<https://youtu.be/_PN4cdiFKmw>)

## Overview
This project personalizes ads using the CLIP model. 
It runs on Raspberry Pi 5, AI+ HAT (with the Hailo 8 device).
The project is a runtime project which receives an input from a USB camera (of a person wearing a certain outfit) and outputs images of the outfit that fits most the "style" of the person (from a certain database).

This system resembles a content personalization system in public spaces, and is capable of interacting with nearby individuals and tailoring commercial content to their preferences.

The system can be utilized in various public settings, such as shopping malls, street billboards, and bus station displays. In retail settings, it serves as a tool to elevate the looks of shop window displays, attracting and engaging customers to enter the store.

 ## Setup Instructions
- Follow README setup instructions of [CLIP application example](https://github.com/giladnah/hailo-CLIP) 

#### Prerequisites
```bash
pip install -r requirements.txt
```
## Arguments
```bash
clip_app -h
usage: clip_app [-h] [--input INPUT] [--detector {person,face,none}] [--json-path JSON_PATH] [--disable-sync] [--dump-dot]
                [--detection-threshold DETECTION_THRESHOLD] [--show-fps] [--enable-callback] [--callback-path CALLBACK_PATH]
                [--disable-runtime-prompts]

Hailo online CLIP app

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        URI of the input stream. Default is /dev/video0. Use '--input demo' to use the demo video.
  --detector {person,face,none}, -d person
                        Which detection pipeline to use.
  --json-path JSON_PATH
                        Path to JSON file to load and save embeddings. If not set, embeddings.json will be used.
  --disable-sync        Disables display sink sync, will run as fast as possible. Relevant when using file source.
  --dump-dot            Dump the pipeline graph to a dot file.
  --detection-threshold DETECTION_THRESHOLD
                        Detection threshold.
  --show-fps, -f        Print FPS on sink.
  --enable-callback     Enables the use of the callback function.
  --callback-path CALLBACK_PATH
                        Path to the custom user callback file.
  --disable-runtime-prompts
                        When set, app will not support runtime prompts. Default is False.
```

For more information:
```bash
clip_app -h
```

## Example
```bash
clip_app -d person -i /dev/video0 --json-path /home/hailo/hailo-CLIP/data_embdedding.json --disable-runtime-prompts --enable-callback
```

## Additional Notes
- blah blah
- Make sure Your USB camera is configured properly with the Rpi and (its output is in /dev/video0)
-

