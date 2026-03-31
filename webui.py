from __future__ import annotations

import os
import time

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from modules import timer
from modules import initialize_util
from modules import initialize
from threading import Thread
from modules_forge.initialization import initialize_forge
from modules_forge import main_thread


startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize_forge()

initialize.imports()

initialize.check_versions()

initialize.initialize()


def _handle_exception(request: Request, e: Exception):
    error_information = vars(e)
    content = {
        "error": type(e).__name__,
        "detail": error_information.get("detail", ""),
        "body": error_information.get("body", ""),
        "message": str(e),
    }
    return JSONResponse(status_code=int(error_information.get("status_code", 500)), content=jsonable_encoder(content))


def create_api(app):
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api


def api_only_worker():
    from fastapi import FastAPI
    from modules.shared_cmd_options import cmd_opts

    app = FastAPI(exception_handlers={Exception: _handle_exception})
    initialize_util.setup_middleware(app)
    api = create_api(app)

    from modules import script_callbacks
    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(
        server_name=initialize_util.gradio_server_name(),
        port=cmd_opts.port if cmd_opts.port else 7861,
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else ""
    )


def webui_worker():
    from modules.shared_cmd_options import cmd_opts

    launch_api = cmd_opts.api

    from modules import shared, ui_tempdir, script_callbacks, ui, progress, ui_extra_networks

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = ui.create_ui()
        startup_timer.record("create ui")

        if not cmd_opts.no_gradio_queue:
            shared.demo.queue(64)

        gradio_auth_creds = list(initialize_util.get_gradio_auth_creds()) or None

        auto_launch_browser = False
        if os.getenv('SD_WEBUI_RESTARTING') != '1':
            if shared.opts.auto_launch_browser == "Remote" or cmd_opts.autolaunch:
                auto_launch_browser = True
            elif shared.opts.auto_launch_browser == "Local":
                auto_launch_browser = not cmd_opts.webui_is_non_local

        from modules_forge.forge_canvas.canvas import canvas_js_root_path

        app, local_url, share_url = shared.demo.launch(
            share=cmd_opts.share,
            server_name=initialize_util.gradio_server_name(),
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            ssl_verify=cmd_opts.disable_tls_verify,
            debug=cmd_opts.gradio_debug,
            auth=gradio_auth_creds,
            inbrowser=auto_launch_browser,
            prevent_thread_lock=True,
            allowed_paths=cmd_opts.gradio_allowed_path + [canvas_js_root_path],
            app_kwargs={
                "docs_url": "/docs",
                "redoc_url": "/redoc",
                "exception_handlers": {Exception: _handle_exception},
            },
            root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
        )

        startup_timer.record("gradio launch")

        # gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
        # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
        # running web ui and do whatever the attacker wants, including installing an extension and
        # running its code. We disable this here. Suggested by RyotaK.
        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']

        initialize_util.setup_middleware(app)

        progress.setup_progress_api(app)
        ui.setup_ui_api(app)

        if launch_api:
            create_api(app)

        ui_extra_networks.add_pages_to_demo(app)

        startup_timer.record("add APIs")

        with startup_timer.subcategory("app_started_callback"):
            script_callbacks.app_started_callback(shared.demo, app)

        timer.startup_record = startup_timer.dump()
        print(f"Startup time: {startup_timer.summary()}.")

        try:
            while True:
                server_command = shared.state.wait_for_server_command(timeout=5)
                if server_command:
                    if server_command in ("stop", "restart"):
                        break
                    else:
                        print(f"Unknown server command: {server_command}")
        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt, stopping...')
            server_command = "stop"

        if server_command == "stop":
            print("Stopping server...")
            # If we catch a keyboard interrupt, we want to stop the server and exit.
            shared.demo.close()
            break

        # disable auto launch webui in browser for subsequent UI Reload
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')

        print('Restarting UI...')
        shared.demo.close()
        time.sleep(0.5)
        startup_timer.reset()
        script_callbacks.app_reload_callback()
        startup_timer.record("app reload callback")
        script_callbacks.script_unloaded_callback()
        startup_timer.record("scripts unloaded callback")
        initialize.initialize_rest(reload_script_modules=True)


def auto_generate_only_worker(task_file):
    from modules import shared, script_callbacks, sd_models
    from modules_forge import main_entry
    import sys
    import os
    
    script_callbacks.before_ui_callback()
    
    from fastapi import FastAPI
    app = FastAPI()
    script_callbacks.app_started_callback(None, app)
    
    main_entry.refresh_model_loading_parameters()
    
    script_path = os.path.join(os.path.dirname(__file__), "scripts", "prompts_from_file_auto.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("prompts_from_file_auto", script_path)
    auto_script = importlib.util.module_from_spec(spec)
    sys.modules["prompts_from_file_auto"] = auto_script
    spec.loader.exec_module(auto_script)

    from modules.processing import StableDiffusionProcessingTxt2Img
    from modules.shared import opts, state
    import modules.scripts as scripts
    
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt="",
        styles=[],
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        sampler_name="DPM++ 2M",
        scheduler="Automatic",
        batch_size=1,
        n_iter=7,
        steps=30,
        cfg_scale=5.0,
        width=1024,
        height=1536,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=True
    )
    p.scripts = scripts.scripts_txt2img
    p.script_args = tuple([None] * p.scripts.alwayson_scripts_num) if hasattr(p.scripts, "alwayson_scripts_num") else ()
    
    script = auto_script.Script()
    state.job_count = 0
    state.job_no = 0
    print(f"Auto-generating from {task_file}...")
    try:
        script.run(p, checkbox_iterate=False, checkbox_iterate_batch=False, prompt_position="start", prompt_txt="", task_file=task_file)
        print("Auto-generation completed.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during auto-generation: {e}")
    finally:
        import signal
        os.kill(os.getpid(), signal.SIGINT)


def auto_generate_once_only_worker(task_file):
    from modules import shared, script_callbacks, sd_models
    from modules_forge import main_entry
    import sys
    import os
    
    script_callbacks.before_ui_callback()
    
    from fastapi import FastAPI
    app = FastAPI()
    script_callbacks.app_started_callback(None, app)
    
    main_entry.refresh_model_loading_parameters()
    
    script_path = os.path.join(os.path.dirname(__file__), "scripts", "prompts_from_file_auto_once.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("prompts_from_file_auto_once", script_path)
    auto_script = importlib.util.module_from_spec(spec)
    sys.modules["prompts_from_file_auto_once"] = auto_script
    spec.loader.exec_module(auto_script)

    from modules.processing import StableDiffusionProcessingTxt2Img
    from modules.shared import opts, state
    import modules.scripts as scripts
    
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt="",
        styles=[],
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        sampler_name="DPM++ 2M",
        scheduler="Automatic",
        batch_size=1,
        n_iter=7,
        steps=30,
        cfg_scale=5.0,
        width=810,
        height=1080,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=True,

        # --- Hires. fix (Latent Upscaler) 설정 추가 ---
        enable_hr=True,               # Hires. fix 활성화
        hr_upscaler="Latent",         # 업스케일러 종류 ("Latent", "Latent (antialiased)", "Latent (bicubic)", "Latent (nearest-exact)" 등 사용 가능)
        hr_scale=2.0,                 # 업스케일 배율 (예: 2.0이면 1024x1536 -> 2048x3072 로 확대)
        denoising_strength=0.55,      # 디노이징 강도 (Latent의 경우 보통 0.5 ~ 0.75 사이 권장, 너무 낮으면 흐릿하고 높으면 원본과 달라집니다)
        hr_second_pass_steps=20,      # Hires. fix에 사용할 스텝 수 (0으로 두면 기본 steps와 동일하게 작동)
        hr_additional_modules=[],     # Hires. fix 추가 모듈 초기화 (필수)
        # ----------------------------------------------
    )
    p.scripts = scripts.scripts_txt2img
    p.script_args = tuple([None] * p.scripts.alwayson_scripts_num) if hasattr(p.scripts, "alwayson_scripts_num") else ()
    
    script = auto_script.Script()
    state.job_count = 0
    state.job_no = 0
    print(f"Auto-generating (once) from {task_file}...")
    try:
        script.run(p, checkbox_iterate=False, checkbox_iterate_batch=False, prompt_position="start", prompt_txt="", task_file=task_file)
        print("Auto-generation once completed.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during auto-generation once: {e}")
    finally:
        import signal
        os.kill(os.getpid(), signal.SIGINT)


def auto_generate_only(task_file):
    Thread(target=auto_generate_only_worker, args=(task_file,), daemon=True).start()


def auto_generate_once_only(task_file):
    Thread(target=auto_generate_once_only_worker, args=(task_file,), daemon=True).start()


def api_only():
    Thread(target=api_only_worker, daemon=True).start()


def webui():
    Thread(target=webui_worker, daemon=True).start()


if __name__ == "__main__":
    from modules.shared_cmd_options import cmd_opts

    # If auto actions are configured, make sure we imply nowebui
    if getattr(cmd_opts, 'auto_generate', None) is not None or getattr(cmd_opts, 'auto_generate_once', None) is not None:
        cmd_opts.nowebui = True

    def listen_for_interrupt():
        import sys
        from modules import shared
        print("Interrupt listener started. Type 'q' and press Enter in the terminal to interrupt smoothly at any time...")
        for line in sys.stdin:
            if line.strip().lower() == 'q':
                print("\nInterrupt requested via terminal. Finishing current image generation...")
                shared.state.interrupt()

    Thread(target=listen_for_interrupt, daemon=True).start()

    if getattr(cmd_opts, 'auto_generate', None):
        auto_generate_only(cmd_opts.auto_generate)
    elif getattr(cmd_opts, 'auto_generate_once', None):
        auto_generate_once_only(cmd_opts.auto_generate_once)
    elif cmd_opts.nowebui:
        api_only()
    else:
        webui()

    main_thread.loop()
