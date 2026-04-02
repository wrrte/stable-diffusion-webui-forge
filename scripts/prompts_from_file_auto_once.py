import copy
import random
import shlex
import os
import re

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers, errors, sd_models
from modules.processing import Processed, process_images
from modules.shared import state


def process_model_tag(tag):
    info = sd_models.get_closet_checkpoint_match(tag)
    assert info is not None, f'Unknown checkpoint: {tag}'
    return info.name


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag.lower() == "true") else False


prompt_tags = {
    "sd_model": process_model_tag,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag,
    "enable_hr": process_boolean_tag,
    "denoising_strength": process_float_tag,
    "hr_upscaler": process_string_tag,
    "hr_scale": process_float_tag,
    "hr_second_pass_steps": process_int_tag,
    "hr_resize_x": process_int_tag,
    "hr_resize_y": process_int_tag
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue

        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        val = args[pos+1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)
        pos += 2

    return res


def load_prompt_file(file):
    if file is None:
        return None, gr.update(), gr.update(lines=7)
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]
        return None, "\n".join(lines), gr.update(lines=7)


def parse_line_robust(line):
    line = line.strip()
    if not line:
        return {"prompt": ""}
    
    idx = -1
    for tag in prompt_tags.keys():
        found = line.find(f" --{tag}")
        if found != -1:
            if idx == -1 or found < idx:
                idx = found

    if idx != -1:
        prompt_part = line[:idx].strip()
        args_part = line[idx:].strip()
        try:
            args = cmdargs(args_part)
            args["prompt"] = prompt_part
            return args
        except Exception:
            errors.report(f"Error parsing args from {args_part}", exc_info=True)
            return {"prompt": line}
    else:
        if line.startswith("--"):
            try:
                return cmdargs(line)
            except Exception:
                errors.report(f"Error parsing line {line}", exc_info=True)
                return {"prompt": line}
        else:
            return {"prompt": line}


class Script(scripts.Script):
    def title(self):
        return "Prompts from file (Auto-Update only once)"

    def ui(self, is_img2img):
        task_file = gr.Textbox(label="Task File Path", lines=1, elem_id=self.elem_id("task_file"))
        
        checkbox_iterate = gr.Checkbox(label="Iterate seed every line", value=False, elem_id=self.elem_id("checkbox_iterate"))
        checkbox_iterate_batch = gr.Checkbox(label="Use same random seed for all lines", value=False, elem_id=self.elem_id("checkbox_iterate_batch"))
        prompt_position = gr.Radio(["start", "end"], label="Insert prompts at the", elem_id=self.elem_id("prompt_position"), value="start")

        prompt_txt = gr.Textbox(label="List of prompt inputs", lines=1, elem_id=self.elem_id("prompt_txt"))
        file = gr.File(label="Upload prompt inputs", type='binary', elem_id=self.elem_id("file"))

        file.change(fn=load_prompt_file, inputs=[file], outputs=[file, prompt_txt, prompt_txt], show_progress=False)

        prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[prompt_txt], outputs=[prompt_txt], show_progress=False)
        
        return [checkbox_iterate, checkbox_iterate_batch, prompt_position, prompt_txt, task_file]

    def run(self, p, checkbox_iterate, checkbox_iterate_batch, prompt_position, prompt_txt: str, task_file: str):
        
        task_file_path = task_file.strip() if task_file else ""
        task_file_path = task_file_path.strip('"').strip("'")
        
        lines = []

        if task_file_path:
            if os.path.exists(task_file_path):
                with open(task_file_path, "r", encoding="utf-8") as f:
                    # 빈 줄을 필터링하지 않고 유지하도록 변경했습니다.
                    lines = [x.strip() for x in f.readlines()]
            else:
                lines = [x.strip() for x in prompt_txt.splitlines()]
                print(f"Task file not found. Falling back to textbox inputs and will create file at: {task_file_path}")
        else:
            lines = [x.strip() for x in prompt_txt.splitlines()]

        # 내용이 있는 프롬프트가 한 줄도 없는지 확인합니다.
        if not any(line for line in lines):
            print("No prompts found.")
            return Processed(p, [], p.seed, "")

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for line in lines:
            if not line: # 빈 줄인 경우 마커(None)와 함께 저장하여 순서를 유지합니다.
                jobs.append((line, None))
                continue
                
            args = parse_line_robust(line)
            job_count += args.get("n_iter", p.n_iter)
            jobs.append((line, args))

        print(f"Will process {len([j for j in jobs if j[1] is not None])} lines in {job_count} jobs.")
        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        
        uncompleted_jobs = []

        created = False

        for i, (line_str, args) in enumerate(jobs):

            if not line_str or args is None:
                if created or state.interrupted or state.skipped:
                    uncompleted_jobs.append(line_str)
                continue

            if created or state.interrupted or state.skipped:
                uncompleted_jobs.append(line_str)
                continue

            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)

            copy_p.all_prompts = None
            copy_p.all_seeds = None
            copy_p.all_subseeds = None

            for k, v in args.items():
                if k == "sd_model":
                    copy_p.override_settings['sd_model_checkpoint'] = v
                else:
                    setattr(copy_p, k, v)

            if args.get("prompt") and p.prompt:
                if prompt_position == "start":
                    copy_p.prompt = args.get("prompt") + " " + p.prompt
                else:
                    copy_p.prompt = p.prompt + " " + args.get("prompt")

            if args.get("negative_prompt") and p.negative_prompt:
                if prompt_position == "start":
                    copy_p.negative_prompt = args.get("negative_prompt") + " " + p.negative_prompt
                else:
                    copy_p.negative_prompt = p.negative_prompt + " " + args.get("negative_prompt")

            completed_iters = 0

            target_iter = args.get("n_iter", p.n_iter) # 원래 해야 할 총 횟수 저장

            if p.seed == -1:
                copy_p.seed = int(random.randrange(4294967294))

            copy_p.n_iter = 1
            start_job_no = state.job_no

            proc = process_images(copy_p)

            completed_iters += state.job_no - start_job_no

            if  ( completed_iters>=1 ):
                created = True

            images += proc.images
            
            # 매장 나올때마다 정보를 바로바로 누적!
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

            # 완료 횟수를 원래 횟수에서 빼서 남은 횟수 계산
            remaining_iters = target_iter - completed_iters

            if remaining_iters > 0:
                if "--n_iter" in line_str:
                    new_line = re.sub(r'--n_iter\s+\d+', f'--n_iter {remaining_iters}', line_str)
                else:
                    new_line = line_str + f' --n_iter {remaining_iters}'
                uncompleted_jobs.append(new_line)

        if task_file_path:
            try:
                with open(task_file_path, "w", encoding="utf-8") as f:
                    for line in uncompleted_jobs:
                        f.write(line + "\n")
            except Exception as e:
                print(f"Error updating task file: {e}")

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
