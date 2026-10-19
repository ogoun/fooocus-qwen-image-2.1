# HTTP API

Start the studio with `--api` (or `-api`) and it serves only an HTTP API, with
no web UI. The API covers everything the interface can do: generation with
references, poses, styles and LoRA, editing with a mask or annotations,
one-click outpainting, video (MiniMax H3 through FreeVideo: text, frames or
references to video, extending and finishing drafts, storyboards), AI boost,
image description, the pose library, the gallery, saved prompts and settings.
It runs the same code path as the interface: the same presets, on-demand
weight downloads, PNG metadata and output folder. The one thing it does not
do is install the video engine, which is a long interactive download; that is
`python -m fooocus_qwen --setup-video` (see [Video](#video)).

Contents:

- [Starting the API](#starting-the-api)
- [Authentication](#authentication)
- [Conventions](#conventions)
- [Generate](#generate)
- [Edit](#edit)
- [Outpaint](#outpaint)
- [Video](#video)
- [Jobs: progress, queue, cancel](#jobs-progress-queue-cancel)
- [Language model: boost, describe](#language-model-boost-describe)
- [Poses](#poses)
- [Gallery and files](#gallery-and-files)
- [Saved prompts](#saved-prompts)
- [Settings and the model](#settings-and-the-model)
- [Catalogues and status](#catalogues-and-status)
- [Errors](#errors)
- [Examples](#examples)
- [Web UI login](#web-ui-login)

## Starting the API

```powershell
.\run.ps1 --api --api-token "a-long-random-string"
# or directly
.venv\Scripts\python -m fooocus_qwen --api --host 127.0.0.1 --port 7865 --api-token "a-long-random-string"
```

The usual flags apply: `--host`, `--port`, `--lora-dir`, `--no-preload`,
`--no-pin-memory`, `--verbose` (adds an access log). The model starts loading
in the background at startup unless `--no-preload` is given. Precision,
SageAttention and the memory profile come from `user/settings.json`, the same
file the Settings tab writes; the API can change them too
(`PATCH /api/v1/settings`).

Once running:

- the API is at `http://HOST:PORT/api/v1/…`;
- interactive documentation (Swagger UI) is at `http://HOST:PORT/docs`, the
  OpenAPI schema at `/openapi.json`. Swagger has an **Authorize** button for
  the token.

`--api` and the web UI are separate modes: one process serves one or the
other. Run two processes on different ports if you need both, but keep in
mind that each one loads its own copy of the model.

## Authentication

Every endpoint except `GET /api/v1/health` requires a Bearer token when one is
set:

```
Authorization: Bearer a-long-random-string
```

Set the token with `--api-token TOKEN`, or with the `FOOOCUS_QWEN_API_TOKEN`
environment variable. The variable is safer: command-line arguments are
visible to every user of the machine in the process list. A missing or wrong
token gets `401` with `WWW-Authenticate: Bearer`.

Without a token the API is open to anyone who can reach the port. That is
fine on `127.0.0.1`. If you start it on any other address without a token,
the log shows a warning. The default host is `0.0.0.0`, which means every
network interface. Behind the internet, put the API behind HTTPS (a reverse
proxy such as Caddy or nginx), because a Bearer token over plain HTTP can be
read on the way.

`--username`/`--password` protect the web UI, not the API. They are ignored
with `--api`, and the log says so.

## Conventions

- **JSON in, JSON out.** Unknown fields are rejected with `422`, so a typo in
  a field name fails instead of silently becoming the default.
- **Images in** can take three forms:
  - a base64 string with or without a `data:image/…;base64,` prefix, in any
    format the UI takes: PNG, JPEG, WebP, HEIC/HEIF, AVIF, JPEG XL, TIFF
    (16-bit, CMYK), PSD, camera RAW (DNG, CR2/CR3, NEF, ARW, RAF, ORF…), EXR,
    HDR and more — recognised by content; EXIF orientation is applied and
    colour profiles are converted to sRGB (SVG is not accepted here: the UI
    rasterizes it in the browser);
  - `{"pose": "standing_05"}`, a pose skeleton from `GET /poses`;
  - `{"gallery": "2026-10-10/17-31-28.png"}`, an earlier result from
    `GET /gallery` or from a previous response's `path`. A gallery video
    (`….mp4`) stands for its last frame: that is how a next shot starts where
    the previous one ended.

  The size limit is 16384 px on a side and 80 MB of base64.
- **Images out.** Every result is saved as PNG with full metadata in the
  outputs folder (`user/outputs/YYYY-MM-DD/`), exactly like in the interface.
  The response lists the images:

  ```json
  {
    "seed": 7,
    "path": "2026-10-10/17-31-28.png",
    "url": "/api/v1/files/2026-10-10/17-31-28.png",
    "width": 1216,
    "height": 896,
    "parameters": { "...": "the PNG metadata" },
    "base64": "iVBORw0KGgo..."
  }
  ```

  `base64` is the saved PNG byte for byte, metadata included. Send
  `"return_base64": false` to get only the `url`.
- **Waiting.** GPU work goes to one queue and runs one job at a time:
  `/generate`, `/edit`, `/outpaint`, `/video`, `/video/storyboard/run`,
  `/poses/{name}/cover` and `PATCH /settings`. By default the request waits and returns the result.
  Add `?wait=false` to get `202 Accepted` with a job to poll instead (see
  [Jobs](#jobs-progress-queue-cancel)). A disconnected client doesn't cancel
  its job; use `DELETE /jobs/{id}`.
- **Messages.** Results carry a `messages` list in English: skipped LoRA, a
  reference scale chosen automatically, an empty mask, a failed AI boost.
  They are the same notes the interface shows in its status line.
- **Seeds.** `-1` means random. The seed actually used is in each image's
  `seed`; several images get consecutive seeds.

## Generate

`POST /api/v1/generate`

| Field | Default | Meaning |
|---|---|---|
| `prompt` | required | the prompt; reference images are `<image1>`, `<image2>`… in order |
| `negative_prompt` | `""` | takes effect only with `true_cfg_scale` > 1 |
| `styles` | `[]` | style names from `GET /styles` |
| `preset` | `MiddleQuality` | preset name from `GET /presets` |
| `aspect` | `1:1` | `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16` or `reference` (take it from the first reference) |
| `image_number` | `1` | 1–8 |
| `seed` | `-1` | `-1` is random |
| `true_cfg_scale` | `1.0` | 1–10; above 1, each step costs twice as much |
| `use_kv_cache` | `true` | reuse reference attention between steps |
| `reference_scale` | `0` | detail of reference images; `0` picks it by their number, as in the interface |
| `references` | `[]` | up to 10 images (base64, pose, gallery) |
| `loras` | `[]` | `[{"name": "my-style", "weight": 0.8}]`, names from `GET /loras` |
| `boost` | `false` | rewrite the prompt with the language model first; it may also pick the aspect ratio unless you gave one |
| `return_base64` | `true` | include images as base64 |

The response has these fields:

- `images` — the results;
- `prompt` — the prompt actually used, rewritten when `boost` was on;
- `conditions` — which `<imageN>` tag each reference got;
- `messages` — notes, as described in [Conventions](#conventions).

## Edit

`POST /api/v1/edit` edits an image by instruction: inside a mask, at marked
areas, or the whole frame.

| Field | Default | Meaning |
|---|---|---|
| `image` | required | the image to edit |
| `prompt` | required | what to change |
| `mask` | `null` | the area to change. Two forms: a black-and-white image (white means change) or a transparent layer with strokes (opaque means change). Resized to the image if needed |
| `mask_mode` | `mask` if a mask is given, otherwise `none` | `mask`: change only inside the mask, everything outside stays pixel-identical; `region`: crop around the mask for more detail; `annotation`: the instructions are coloured marks drawn on `image` itself, no mask needed; `none`: edit the whole frame |
| `crop` | `null` | `[left, top, right, bottom]` in pixels of `image`. Crops the image and the mask before editing, like the brush's Crop button |
| `mask_grow` | `8` | pixels added around the mask |
| `mask_feather` | `12` | softness of the seam, in pixels |
| `keep_outside` | `true` | paste the original back outside the mask |
| `preset`, `image_number`, `seed`, `references`, `loras`, `boost`, `return_base64` | | as in Generate. References follow the source and mask: with a mask, the first reference is `<image3>`, and `conditions` tells you |

An empty mask edits the whole frame and adds a note to `messages`. With
`mask`/`region` and no mask at all, the request is refused with `400`.

## Outpaint

`POST /api/v1/outpaint` extends the frame in one step with the outpaint LoRA.
There is no mask and no second pass. The LoRA downloads on first use, and the
original is pasted back at full resolution.

| Field | Default | Meaning |
|---|---|---|
| `image` | required | the image to extend |
| `ratio` | — | target aspect ratio; the original stays centred |
| `sides` + `amount` | — | instead of `ratio`: which sides (`left`, `right`, `top`, `bottom`) and by how much of each side (`0.25` = a quarter) |
| `prompt` | `""` | optional: what should appear in the new area |
| `describe` | `false` | let the language model describe the scene first (works better on complex scenes) |
| `crop` | `null` | crop the image first |
| `preset` | `QwenTurbo` | |
| `seed`, `loras`, `return_base64` | | as in Generate |

Give either `ratio` or `sides`. An image that already has the target ratio
gets `400`.

## Video

Video is made by MiniMax H3 through the FreeVideo engine, as on the Video
tab. FreeVideo lives in its own folder and Python environment, so it is
installed separately, from the command line only, because the setup downloads
40–55 GB and asks for the model licence:

```powershell
.venv\Scripts\python -m fooocus_qwen --setup-video
```

There is no install endpoint. Until the engine is ready, `/video` and
`/video/storyboard/run` answer `503` with the reason, and
`GET /api/v1/video/status` tells you what is missing. A video job unloads the
image model first, so the two models never share the GPU. The image model
loads again with the next image job.

### Make a video

`POST /api/v1/video` is a GPU job, like `/generate`.

| Field | Default | Meaning |
|---|---|---|
| `prompt` | required, except with `continue_from`/`refine_from` | the scene in plain English, or a ready H3 prompt. The timing line and reference labels are added automatically. H3 was trained on English descriptions; spoken lines may be in any language |
| `mode` | `t2v` | `t2v` text to video, `i2v` first frame to video, `flf2v` first and last frame, `ref` references (characters, style, voice) |
| `resolution` | `480p` | area class: `480p` (864×480), `576p` (1024×576), `768p` (1344×768, H3's native size). Portrait and square frames keep the same area |
| `aspect` | `16:9` | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9`, or `image` to follow the first frame |
| `seconds` | `5` | 1.625–60. Snapped up to H3's frame grid (17n+5 frames at 24 fps); over 15 s the video is made in parts with seamless continuation |
| `seed` | `-1` | `-1` is random |
| `quality` | `light` | `draft` (half-size first pass, finish it later with `refine_from`), `light`, `medium`, `high`, `max`; slower to the right |
| `first`, `last` | `null` | frames, in any image form from [Conventions](#conventions). `first` is used by `i2v` and `flf2v`, `last` by `flf2v`; frames are cropped to the canvas, not stretched. A gallery video gives its last frame |
| `references` | `[]` | `ref` mode only: up to 9 images, 3 videos and 3 audio files. Each is `{"kind": "image"\|"video"\|"audio", "base64": "…"}` or `{"gallery": "2026-10-19/12-00-00.png"}`; `"name": "voice.mp3"` gives the file type when `kind` is left out. A sound needs at least one image or video |
| `loras` | `[]` | `[{"name": "my-motion", "strength": 0.8}]`, files from `GET /video/loras` (the `loras/video` folder), strength −2…2 |
| `boost` | `false` | rewrite the prompt into an H3 prompt with the language model first. It sees the frames and image references |
| `continue_from` | `null` | a gallery video to extend. The new part continues its last moments. `prompt`, `seconds` and `seed` default to the source's; give them to change the next part |
| `refine_from` | `null` | a gallery `draft` to finish at `light` quality without generating it anew. Other fields are ignored |
| `return_base64` | `false` | also return the MP4 in base64 |

References get the labels the model knows them by, in the order images,
videos, sounds: `<Picture 1>`, `<Video 1>` (with `<Audio 1>` if that video has
sound), then `<Audio N>` for separate sounds. Use them in the prompt:
`"<Picture 1> walks into the room and says hello in the voice of <Audio 1>"`.

The response:

```json
{
  "video": {
    "path": "2026-10-19/12-00-00.mp4",
    "url": "/api/v1/files/2026-10-19/12-00-00.mp4",
    "width": 864, "height": 480, "seconds": 5.167, "seed": 7,
    "parameters": {"prompt": "…", "prompt_h3": "…", "mode": "t2v", "frames": 124, "quality": "light", "engine_output": "…"}
  },
  "prompt": "the H3 prompt actually used",
  "messages": []
}
```

The MP4 is saved in the outputs folder next to the pictures, and its
parameters are saved next to it as a `.json` file with the same name. The
field names match the request (`prompt`, `mode`, `resolution`, `aspect`,
`seed`, `quality`), so the parameters can be sent back to repeat a video.
`messages` notes ignored inputs (a frame in `t2v`, references outside `ref`),
a non-English prompt, and the AI boost result.

While the job runs, its `progress` carries the engine's own estimate:
`fraction` of the whole video (0–1, loading and decoding included),
`remaining_seconds`, the sampling `step` of `steps`, and `stage` (the
engine's label). `DELETE /jobs/{id}` and `POST /interrupt` stop the engine.

### Prompt, storyboard, status

| Endpoint | Body | Answer |
|---|---|---|
| `POST /api/v1/video/boost` | `{"prompt", "mode", "seconds", "aspect", "first", "last", "references"}` | `{"prompt", "message"}`, the H3 prompt. Needs the language model; `502` if it fails |
| `POST /api/v1/video/storyboard` | `{"brief", "seconds": 20 (total, 5–120), "structure", "aspect"}` | `{"shots": [{"seconds", "transition", "title", "prompt"}], "seconds", "message"}`. `structure` is `free`, `problem_action_result`, `promise_action_result_next`, `explainer` or `story` |
| `POST /api/v1/video/storyboard/run` | `{"shots", "chain": true, "first", "resolution", "aspect", "seed", "quality", "loras", "return_base64"}` | a GPU job: every shot in turn, then one video joined with the transitions. With `chain`, each shot starts from the last frame of the previous one; shot N gets seed `seed + N - 1`. Answers `{"video", "shots", "contact_sheet", "seconds"}` |
| `GET /api/v1/video/status` | | `{"installed", "ready", "reason", "install"}` |
| `GET /api/v1/video/loras` | | file names in `loras/video` |

Shot `transition` is the cut into that shot: `cut`, `dissolve`, `fade`,
`wipeleft`, `wiperight` (common names such as `cross dissolve` work too).
Shot `seconds` are snapped to the frame grid. You don't have to use
`/storyboard/run`: you can also send each shot to `/video` yourself, passing
`{"gallery": "<previous shot>.mp4"}` as `first`.

## Jobs: progress, queue, cancel

Any GPU request with `?wait=false` returns `202` and a `Location` header:

```json
{"id": "5d6b…", "kind": "generate", "status": "queued", "queue_position": 0,
 "progress": {"stage": "", "image": 0, "images": 0, "step": 0, "steps": 0,
              "fraction": null, "remaining_seconds": null}, "created": 1760104288.1}
```

| Endpoint | |
|---|---|
| `GET /api/v1/jobs/{id}` | state of a job. `status` is one of `queued`, `running`, `done`, `failed`, `cancelled`. `progress` has these fields: `stage` (`downloading …`, `loading`, `preparing`, `boosting the prompt`, `generating`), `image` of `images`, `step` of `steps`; video jobs also fill `fraction` and `remaining_seconds` (see [Video](#video)). A `done` job carries `result`, the same body the waiting request would have returned. A `failed` job carries `error` |
| `GET /api/v1/jobs` | all jobs the server remembers (the last 200 finished ones and everything queued) |
| `DELETE /api/v1/jobs/{id}` | a queued job is removed. A running one stops after the current step and becomes `cancelled`; a storyboard also skips the shots it has not started |
| `POST /api/v1/interrupt` | stop whatever is running now, image or video, like the Stop button |

Wait mode answers the same way. A failed generation, such as running out of
GPU memory, gets `500` with `detail` and the `job` id. A job cancelled while
the request waited gets `409`.

## Language model: boost, describe

These endpoints need a language model server: the address in
`llm_endpoint.txt`, the Settings tab, or `PUT /llm`. If it is unreachable,
they answer `502` with the reason.

| Endpoint | Body | Answer |
|---|---|---|
| `POST /api/v1/boost` | `{"prompt", "mode": "t2i"\|"edit" (default: edit when references are given), "references": []}` | `{"prompt", "aspect", "images_skipped"}` |
| `POST /api/v1/describe` | `{"image"}` | `{"text"}`, the image described as a prompt |
| `GET /api/v1/llm?check=true` | | `{"configured", "address", "token_set", "reachable", "model"}`. The token itself is never returned |
| `PUT /api/v1/llm` | `{"address", "token", "forget_token"}` | an empty `token` keeps the stored one |
| `GET /api/v1/system-prompts` | | names of the editable system prompts |
| `GET`/`PUT /api/v1/system-prompts/{name}` | `{"text"}` | read or replace a system prompt |

## Poses

| Endpoint | |
|---|---|
| `GET /api/v1/poses` | the catalogue and your own poses: `name`, `title`, `custom`, `schematic`, `skeleton_url`, `preview_url` |
| `GET /api/v1/poses/{name}/skeleton` | the OpenPose skeleton PNG, which is what goes to the model |
| `GET /api/v1/poses/{name}/preview` | the cover, or the skeleton in schematic view |
| `POST /api/v1/poses` | `{"image", "title"}` recognises the pose on a photo (DWPose) and saves it as your own pose. A photo without a person gets `400` |
| `PATCH /api/v1/poses/{name}` | `{"title", "schematic"}` |
| `DELETE /api/v1/poses/{name}` | deletes your own pose, or hides a catalogue pose |
| `POST /api/v1/poses/{name}/cover` | draws the cover with the model (a GPU job) |

To use a pose, pass `{"pose": "<name>"}` among `references` and refer to it
by its tag: `"a knight in the pose of <image1>"`.

## Gallery and files

| Endpoint | |
|---|---|
| `GET /api/v1/gallery?limit=60` | recent images and videos, newest first: `path`, `url`, `kind` (`image` or `video`), `parameters` (PNG metadata, or a video's `.json` next to it) |
| `GET /api/v1/files/{path}` | the file itself: `image/png` or `video/mp4` (with range requests, so players can seek). Only files inside the outputs folder are served |

The `parameters` of a result can be sent back as a new request: field names
match (`prompt`, `negative_prompt`, `styles`, `seed`, `true_cfg_scale`, …,
and for a video `mode`, `resolution`, `aspect`, `quality`). A video's
`prompt` is the text you wrote, `prompt_h3` is what the model got.

## Saved prompts

Saved prompts are the same files as the interface's saved prompts, in
`user/prompts/`.

| Endpoint | |
|---|---|
| `GET /api/v1/prompts` | names |
| `GET /api/v1/prompts/{name}` | the fields that were saved: `{"prompt", "negative_prompt", "styles", "preset", "aspect", "seed", "true_cfg_scale", "loras"}` from the Generate tab or the API, only `{"prompt"}` from the Edit tab and for the shipped **Upscale** |
| `PUT /api/v1/prompts/{name}` | save; returns what was stored |
| `DELETE /api/v1/prompts/{name}` | |

## Settings and the model

| Endpoint | |
|---|---|
| `GET /api/v1/settings` | `precision`, `precisions`, `recommended_precision`, `sage_attention`, `sage_available`, `memory_profile`, `memory_profiles`, `resolved_profile`, `vram_gib` |
| `PATCH /api/v1/settings` | `{"precision", "sage_attention", "memory_profile"}`, any subset. A GPU job: it downloads the weights a new precision needs and reloads the model. SageAttention switches on the fly |
| `POST /api/v1/model/load` | start loading the model in the background |
| `POST /api/v1/model/unload` | free the GPU, after the current job |

## Catalogues and status

| Endpoint | |
|---|---|
| `GET /api/v1/health` | `{"status": "ok"}`, no token needed |
| `GET /api/v1/info` | version, `model_loaded`, GPU memory, presets, aspect ratios, mask modes, outpaint sides, the LoRA folder, settings, and under `video` the video modes, resolutions, aspect ratios, qualities, duration limits, storyboard structures and transitions |
| `GET /api/v1/presets` | `name`, `resolution`, `steps`, `fast`, `default` |
| `GET /api/v1/styles` | style names |
| `GET /api/v1/loras` | files in the LoRA folder: `name`, `usable`, `problem`, `family`, `rank`, `triggers` |

Presets, fastest first:

| Preset | Size and steps | Notes |
|---|---|---|
| `TurboDraft` | 768, 6 steps | |
| `Turbo` | 1024, 6 steps | |
| `QwenTurbo` | 1024, 8 steps | |
| `Turbo4` | 1024, 4 steps | |
| `QwenTurboHD` | 1536, 8 steps | |
| `QwenTurbo2K` | 2048, 8 steps | |
| `LowQuality` | 1024, 16 steps | |
| `MiddleQuality` | 1536, 28 steps | the default |
| `MaxQuality` | 2048, 40 steps | |

A fast preset downloads its weights on first use. That job's `stage` says
`downloading …` while it does.

## Errors

| Code | When |
|---|---|
| `400` | the request can't work: unreadable image, unknown style, crop outside the image, `aspect: reference` without references, nothing to outpaint, a video mode without its frames or references, too many references, a video without engine files to extend |
| `401` | missing or wrong token |
| `404` | no such job, pose, gallery file, video LoRA or saved prompt |
| `409` | the job was cancelled, or is already finished when you try to cancel it |
| `422` | the body doesn't match the schema: unknown field, unknown preset, value out of range |
| `500` | the generation itself failed (out of GPU memory, missing weights); `detail` says why |
| `502` | the language model is unreachable or answered with an error |
| `503` | the video engine is not installed or its setup is unfinished; `detail` says what to run |

Error bodies look like `{"detail": "…"}`.

## Examples

### curl

```bash
TOKEN=a-long-random-string
curl -s http://127.0.0.1:7865/api/v1/generate \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox sitting in fresh snow, golden hour", "preset": "QwenTurbo", "aspect": "4:3", "seed": 7}' \
  | python -c "import sys, json, base64; r = json.load(sys.stdin); open('fox.png', 'wb').write(base64.b64decode(r['images'][0]['base64']))"

# a 5-second video from that picture, then download the MP4
curl -s http://127.0.0.1:7865/api/v1/video \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"prompt": "The fox stands up and trots away through the snow.", "mode": "i2v",
       "first": {"gallery": "2026-10-19/17-31-28.png"}, "aspect": "image", "seconds": 5}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['video']['url'])"
curl -s -H "Authorization: Bearer $TOKEN" -o fox.mp4 http://127.0.0.1:7865/api/v1/files/2026-10-19/17-33-02.mp4
```

### Python

```python
import base64, time, requests

API = "http://127.0.0.1:7865/api/v1"
session = requests.Session()
session.headers["Authorization"] = "Bearer a-long-random-string"

def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()

# 1. Generate in a pose
fox = session.post(f"{API}/generate", json={
    "prompt": "a knight in shining armour in the pose of <image1>",
    "preset": "QwenTurbo", "aspect": "3:4", "references": [{"pose": "standing_05"}],
}).json()["images"][0]

# 2. Edit it inside a mask (white = change)
edited = session.post(f"{API}/edit", json={
    "image": {"gallery": fox["path"]}, "mask": b64("mask.png"),
    "prompt": "a red cape", "preset": "QwenTurbo",
}).json()["images"][0]

# 3. Extend to 16:9, watching progress
job = session.post(f"{API}/outpaint?wait=false", json={
    "image": {"gallery": edited["path"]}, "ratio": "16:9",
}).json()
while job["status"] in ("queued", "running"):
    time.sleep(1)
    job = session.get(f"{API}/jobs/{job['id']}").json()
    print(job["progress"])
open("wide.png", "wb").write(base64.b64decode(job["result"]["images"][0]["base64"]))

# 4. A video from the picture: a quick draft, then the final from the same draft
if session.get(f"{API}/video/status").json()["ready"]:
    job = session.post(f"{API}/video?wait=false", json={
        "prompt": "The knight raises his sword as the wind catches the red cape.",
        "mode": "i2v", "first": {"gallery": edited["path"]}, "aspect": "image",
        "seconds": 6, "quality": "draft", "boost": True,
    }).json()
    while job["status"] in ("queued", "running"):
        time.sleep(2)
        job = session.get(f"{API}/jobs/{job['id']}").json()
        print(job["progress"]["stage"], job["progress"]["fraction"], job["progress"]["remaining_seconds"])
    draft = job["result"]["video"]
    final = session.post(f"{API}/video", json={"refine_from": draft["path"]}).json()["video"]
    open("knight.mp4", "wb").write(session.get(f"http://127.0.0.1:7865{final['url']}").content)

    # 5. Storyboard: brief -> shots -> every shot joined into one film
    shots = session.post(f"{API}/video/storyboard", json={
        "brief": "A knight's morning: wakes up, rides out, reaches the castle", "seconds": 20,
        "structure": "story",
    }).json()["shots"]
    film = session.post(f"{API}/video/storyboard/run", json={
        "shots": shots, "first": {"gallery": edited["path"]}, "quality": "light",
    }).json()
    print(film["video"]["url"], film["contact_sheet"]["url"])
```

## Web UI login

The web UI (the default mode, without `--api`) can ask for a login:

```powershell
.\run.ps1 --username anna --password "secret"
# safer, without the password in the process list:
$env:FOOOCUS_QWEN_PASSWORD = "secret"; .\run.ps1 --username anna
```

The browser shows Gradio's sign-in page first. The session is kept in a
cookie, and Gradio's own endpoints (`/gradio_api/…`), which the page itself
uses, are closed behind the same login. `--username` without a password is
refused at start, because a half-configured login that leaves the interface
open is worse than an error. As with the API token, use HTTPS through a
reverse proxy beyond your local network.
